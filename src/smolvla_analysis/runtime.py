from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
import zarr

from .libero_state import capture_libero_state, validate_libero_round_trip
from .model_inspection import freeze_policy
from .phase2_capture import capture_action_query
from .phase2_storage import write_action_query, write_libero_snapshot
from .phase_labels import label_phase
from .perturbations import apply_perturbation
from .rollout_recorder import RolloutRecorder
from .schema import EPISODE_COLUMNS, STEP_COLUMNS, create_run_directory, unique_run_id


def _git_info(repo: Path) -> dict[str, Any]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, text=True, capture_output=True).stdout
    return {"commit": commit, "dirty": bool(status.strip())}


def _versions() -> dict[str, str]:
    import importlib.metadata

    names = ("lerobot", "hf-libero", "torch", "transformers", "numpy", "mujoco", "zarr", "pyarrow")
    return {name: importlib.metadata.version(name) for name in names}


def _host_environment() -> dict[str, Any]:
    cpu_model = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True, capture_output=True,
    ).stdout.strip().splitlines()
    return {
        "os": platform.platform(), "python": sys.version, "mujoco_gl": os.environ["MUJOCO_GL"],
        "gpu": torch.cuda.get_device_name(0), "gpu_vram_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
        "driver": driver[0] if driver else None, "cuda": torch.version.cuda,
        "cpu": cpu_model, "cpu_logical_count": os.cpu_count(),
        "ram_total_gib": os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30,
    }


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def _asset_path(project: Path, relative: str | Path) -> Path:
    relative = Path(relative)
    candidates = (project / relative, project / "archive/full_experiment" / relative)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Required local asset is absent: {relative}")


def _prepare_libero_runtime_config(project: Path, destination: Path) -> Path:
    package_roots = [Path(entry) / "libero/libero" for entry in sys.path if entry]
    package_root = next(
        (candidate.resolve() for candidate in package_roots if (candidate / "bddl_files").is_dir()),
        None,
    )
    if package_root is None:
        raise FileNotFoundError("Could not locate the active LIBERO package data on sys.path")
    destination.mkdir(parents=True, exist_ok=False)
    payload = {
        "benchmark_root": str(package_root),
        "bddl_files": str(package_root / "bddl_files"),
        "init_states": str(package_root / "init_files"),
        "datasets": str(_asset_path(project, "checkpoints/libero_datasets")),
        "assets": str(_asset_path(project, "checkpoints/libero_assets")),
    }
    (destination / "config.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
    return destination


def load_runtime(config: dict, project: Path):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    checkpoint = _asset_path(project, config["model"]["local_path"])
    assets = _asset_path(project, "checkpoints/smolvlm_processor")
    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint)
    policy_cfg.device = config["model"]["device"]
    policy_cfg.n_action_steps = config["model"]["n_action_steps"]
    policy_cfg.load_vlm_weights = False
    policy_cfg.vlm_model_name = str(assets)
    started = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(checkpoint, config=policy_cfg).to(policy_cfg.device)
    model_load_time = time.perf_counter() - started
    freeze_policy(policy)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=checkpoint,
        preprocessor_overrides={
            "device_processor": {"device": policy_cfg.device},
            "tokenizer_processor": {"tokenizer_name": str(assets)},
        },
    )
    return policy_cfg, policy, preprocessor, postprocessor, model_load_time


def _make_env(suite: str, task_id: int, config: dict, policy_cfg):
    import libero.libero as libero_runtime
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors

    libero_runtime._assets_path_cache = os.environ["SMOLVLA_LIBERO_ASSETS"]
    env_cfg = LiberoEnv(
        task=suite,
        task_ids=[task_id],
        episode_length=config["benchmark"]["max_steps"],
        obs_type="pixels_agent_pos",
        observation_height=360,
        observation_width=360,
        control_mode=config["model"]["control_mode"],
        max_parallel_tasks=1,
    )
    env = make_env(env_cfg, n_envs=1, use_async_envs=False)[suite][task_id]
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, policy_cfg)
    return env_cfg, env, env_preprocessor, env_postprocessor


def _raw_state(observation: dict) -> tuple[list, list, list]:
    robot = observation["robot_state"]
    eef = np.concatenate([robot["eef"]["pos"][0], robot["eef"]["quat"][0]])
    gripper = robot["gripper"]["qpos"][0]
    joints = robot["joints"]["pos"][0]
    return joints.tolist(), eef.tolist(), gripper.tolist()


def _success(info: dict) -> bool:
    final = info.get("final_info")
    if isinstance(final, dict) and "is_success" in final:
        return bool(np.asarray(final["is_success"])[0])
    return False


def _query_seed(base_seed: int, episode_id: str, query_index: int) -> int:
    payload = f"{base_seed}:{episode_id}:{query_index}".encode()
    return int.from_bytes(sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _postprocess_action_chunk(chunk: np.ndarray, device: str, postprocessor, env_postprocessor) -> np.ndarray:
    processed = []
    for action_index in range(chunk.shape[1]):
        action = torch.as_tensor(chunk[:, action_index], dtype=torch.float32, device=device)
        action = postprocessor(action)
        transition = env_postprocessor({"action": action})
        processed.append(transition["action"].detach().cpu().numpy())
    return np.stack(processed, axis=1)


def execute(config: dict, project: Path, command_line: list[str]) -> Path:
    os.environ["MUJOCO_GL"] = config["benchmark"]["mujoco_gl"]
    os.environ["SMOLVLA_LIBERO_ASSETS"] = str(_asset_path(project, "checkpoints/libero_assets"))
    run_id = unique_run_id(config["project"]["run_name"] or "smolvla")
    run_dir = create_run_directory(project / config["project"]["output_root"], run_id)
    runtime_libero_config = _prepare_libero_runtime_config(project, run_dir / "runtime_libero_config")
    os.environ["LIBERO_CONFIG_PATH"] = str(runtime_libero_config)
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    shutil.copy2(project / "reports/model_modules.json", run_dir / "model_modules.json")
    targets = json.loads((run_dir / "model_modules.json").read_text())["resolved_targets"]
    started_at = datetime.now(UTC)
    expected = sum(len(config["benchmark"]["task_ids"]) * config["benchmark"]["episodes_per_task"] for _ in config["benchmark"]["suites"])
    manifest = {
        "run_id": run_id, "timestamp": started_at.isoformat(), "resolved_config": config,
        "git": _git_info(project), "model_repo_id": config["model"]["repo_id"],
        "model_revision": config["model"]["revision"], "dependency_versions": _versions(),
        "command_line": command_line, "seeds": config["benchmark"]["episode_seeds"],
        "perturbation_condition": config["perturbations"]["condition"], "start_time": started_at.isoformat(),
        "end_time": None, "completion_status": "running", "expected_episode_count": expected,
        "completed_episode_count": 0, "warnings": ["Checkpoint config declares 6-D state; serialized normalization and LIBERO runtime use 8-D."],
        "failures": [],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    recorder = RolloutRecorder(run_dir)
    activation_store = (
        zarr.open_group(str(run_dir / "activations.zarr"), mode="w")
        if config["activations"]["enabled"]
        else None
    )
    environment_store = (
        zarr.open_group(str(run_dir / "environment_states.zarr"), mode="w")
        if config["recording"]["save_environment_state"]
        else None
    )
    environment = _host_environment()
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2))
    manifest["environment"] = environment
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    try:
        policy_cfg, policy, preprocessor, postprocessor, model_load_time = load_runtime(config, project)
        for suite in config["benchmark"]["suites"]:
            for task_id in config["benchmark"]["task_ids"]:
                env_cfg, env, env_preprocessor, env_postprocessor = _make_env(suite, task_id, config, policy_cfg)
                max_steps = int(env.call("_max_episode_steps")[0])
                for episode_index, seed in enumerate(config["benchmark"]["episode_seeds"]):
                    episode_id = f"{suite}_task{task_id:02d}_ep{episode_index:03d}_seed{seed}"
                    policy.reset()
                    torch.cuda.reset_peak_memory_stats()
                    observation, _ = env.reset(seed=[seed])
                    frames, states, policy_states, camera1, camera2 = [], [], [], [], []
                    latencies, actions = [], []
                    episode_activation_group = (
                        activation_store.require_group(episode_id) if activation_store is not None else None
                    )
                    episode_environment_group = (
                        environment_store.require_group(episode_id) if environment_store is not None else None
                    )
                    activation_arrays_written = 0
                    environment_states_written = 0
                    query_index = 0
                    active_query_id = None
                    active_query_seed = None
                    active_query_step = None
                    active_activation_reference = None
                    success = False
                    episode_start = time.perf_counter()
                    previous_action = None
                    previous_delta = None
                    task_name = env.get_attr("task")[0]
                    instruction = env.get_attr("task_description")[0]
                    policy_instruction = instruction
                    termination_reason = "timeout"
                    for step in range(max_steps):
                        from lerobot.envs.utils import add_envs_task, preprocess_observation
                        raw_for_state = observation
                        robot_state, eef_state, gripper_state = _raw_state(raw_for_state)
                        object_state = None
                        goal_state = None
                        environment_state_reference = None
                        if episode_environment_group is not None:
                            snapshot = capture_libero_state(env)
                            step_key = f"step_{step:04d}"
                            environment_states_written += write_libero_snapshot(
                                episode_environment_group, step_key, snapshot
                            )
                            environment_state_reference = f"environment_states.zarr/{episode_id}/{step_key}"
                            object_state = {
                                "objects": snapshot.objects,
                                "contacts": list(snapshot.contacts),
                                "grasped_objects": list(snapshot.grasped_objects),
                            }
                            goal_state = {
                                "predicates": list(snapshot.goal_predicates),
                                "success": snapshot.success,
                            }
                            if step == 0:
                                validation = validate_libero_round_trip(env, snapshot, atol=1e-8)
                                manifest.setdefault("state_round_trip", {})[episode_id] = validation
                        policy_observation = preprocess_observation(observation)
                        policy_observation = add_envs_task(env, policy_observation)
                        policy_observation = apply_perturbation(
                            policy_observation, config["perturbations"]["condition"],
                            config["perturbations"]["parameters"],
                        )
                        if step == 0:
                            policy_instruction = policy_observation["task"][0]
                        camera1.append((policy_observation["observation.images.image"][0].numpy() * 255).astype(np.uint8))
                        camera2.append((policy_observation["observation.images.image2"][0].numpy() * 255).astype(np.uint8))
                        policy_observation = env_preprocessor(policy_observation)
                        policy_states.append(policy_observation["observation.state"][0].cpu().numpy())
                        batch = preprocessor(deepcopy(policy_observation))
                        inference_start = time.perf_counter()
                        if not policy._queues["action"] and episode_activation_group is not None:
                            active_query_id = f"query_{query_index:04d}_step_{step:04d}"
                            active_query_seed = _query_seed(config["project"]["seed"], episode_id, query_index)
                            active_query_step = step
                            action, query = capture_action_query(
                                policy,
                                batch,
                                targets,
                                query_id=active_query_id,
                                flow_noise_seed=active_query_seed,
                            )
                            environment_chunk = _postprocess_action_chunk(
                                query.model_action_chunk,
                                policy_cfg.device,
                                postprocessor,
                                env_postprocessor,
                            )
                            activation_arrays_written += write_action_query(
                                episode_activation_group, query, environment_chunk
                            )
                            active_activation_reference = f"activations.zarr/{episode_id}/{active_query_id}"
                            query_index += 1
                        else:
                            if not policy._queues["action"]:
                                active_query_id = None
                                active_query_seed = None
                                active_query_step = step
                                active_activation_reference = None
                            with torch.inference_mode():
                                action = policy.select_action(batch)
                        latency_ms = (time.perf_counter() - inference_start) * 1000
                        action = postprocessor(action)
                        transition = env_postprocessor({"action": action})
                        action_numpy = transition["action"].detach().cpu().numpy()
                        if action_numpy.shape != (1, 7) or not np.isfinite(action_numpy).all():
                            raise RuntimeError(f"Invalid executed action at {episode_id} step {step}: {action_numpy}")
                        executed = action_numpy[0]
                        frames.append(env.envs[0].render())
                        observation, reward, terminated, truncated, info = env.step(action_numpy)
                        success = success or _success(info)
                        done = bool(terminated[0] or truncated[0] or success)
                        delta = np.zeros_like(executed) if previous_action is None else executed - previous_action
                        jerk = np.zeros_like(executed) if previous_delta is None else delta - previous_delta
                        # The policy queue stores normalized model outputs, while
                        # ``executed`` has passed through both action processors.
                        # Apply the same processors to queued actions so each
                        # recorded chunk has one consistent, environment-facing
                        # coordinate system.
                        future_actions = []
                        for queued_action in policy._queues["action"]:
                            queued_action = postprocessor(queued_action)
                            queued_transition = env_postprocessor({"action": queued_action})
                            future_actions.append(queued_transition["action"][0].detach().cpu().numpy())
                        active_chunk = np.stack([executed, *future_actions])
                        normalized_progress = step / max(max_steps - 1, 1)
                        phase, phase_evidence = label_phase(None, normalized_progress, config["phase_thresholds"])
                        queue_age = None if active_query_step is None else step - active_query_step
                        recorder.add_step(
                            dict.fromkeys(STEP_COLUMNS) | {
                                "run_id": run_id, "episode_id": episode_id, "env_step": step,
                                "normalized_progress": normalized_progress, "timestamp": datetime.now(UTC).isoformat(),
                                "task_phase": phase, "robot_state": robot_state, "eef_state": eef_state,
                                "gripper_state": gripper_state, "object_state": object_state,
                                "goal_state": goal_state or {"phase_evidence": phase_evidence},
                                "action_query_id": active_query_id, "flow_noise_seed": active_query_seed,
                                "queue_age": queue_age, "chunk_action_index": queue_age,
                                "predicted_action_chunk": active_chunk.tolist(),
                                "executed_action": executed.tolist(), "action_norm": float(np.linalg.norm(executed)),
                                "action_smoothness": float(np.linalg.norm(delta)), "action_jerk": float(np.linalg.norm(jerk)),
                                "gripper_action": float(executed[6]), "policy_latency_ms": latency_ms,
                                "gpu_memory_mb": torch.cuda.memory_allocated() / 2**20, "uncertainty_features": None,
                                "activation_reference": active_activation_reference,
                                "environment_state_reference": environment_state_reference,
                            }
                        )
                        latencies.append(latency_ms)
                        actions.append(executed)
                        states.append(robot_state)
                        previous_action, previous_delta = executed.copy(), delta.copy()
                        if done:
                            termination_reason = "success" if success else ("terminated" if terminated[0] else "truncated")
                            break
                    frames.append(env.envs[0].render())
                    video_rel = f"videos/{episode_id}.mp4"
                    imageio.mimsave(run_dir / video_rel, frames, fps=30, codec="libx264", quality=7)
                    obs_rel = f"observations/{episode_id}.npz"
                    np.savez_compressed(
                        run_dir / obs_rel, camera1=np.stack(camera1), camera2=np.stack(camera2),
                        robot_state=np.asarray(states, dtype=np.float32), policy_state=np.asarray(policy_states, dtype=np.float32),
                        executed_actions=np.asarray(actions, dtype=np.float32),
                    )
                    latencies_array = np.asarray(latencies)
                    recorder.add_episode(
                        dict.fromkeys(EPISODE_COLUMNS) | {
                            "run_id": run_id, "episode_id": episode_id, "suite": suite, "task_id": task_id,
                            "task_name": task_name, "instruction": policy_instruction, "episode_index": episode_index,
                            "seed": seed, "initial_state_id": episode_index, "condition": config["perturbations"]["condition"],
                            "success": success, "total_steps": len(actions), "termination_reason": termination_reason,
                            "failure_class": None if success else "unknown", "failure_onset_step": None,
                            "model_load_time_s": model_load_time, "wall_time_s": time.perf_counter() - episode_start,
                            "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
                            "latency_mean_ms": float(latencies_array.mean()), "latency_p50_ms": float(np.quantile(latencies_array, .5)),
                            "latency_p95_ms": float(np.quantile(latencies_array, .95)), "video_path": video_rel,
                            "observation_path": obs_rel,
                            "activation_path": (
                                f"activations.zarr/{episode_id}" if episode_activation_group is not None else None
                            ),
                            "environment_state_path": (
                                f"environment_states.zarr/{episode_id}"
                                if episode_environment_group is not None
                                else None
                            ),
                            "infrastructure_failure": False,
                        }
                    )
                    manifest["completed_episode_count"] += 1
                    manifest.setdefault("activation_arrays", {})[episode_id] = activation_arrays_written
                    manifest.setdefault("environment_state_arrays", {})[episode_id] = environment_states_written
                    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
                env.close()
        recorder.finalize()
        manifest["completion_status"] = "complete"
    except Exception as error:
        manifest["completion_status"] = "failed"
        manifest["failures"].append(f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["end_time"] = datetime.now(UTC).isoformat()
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return run_dir
