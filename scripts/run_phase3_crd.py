#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import zarr

from smolvla_analysis.config import load_config
from smolvla_analysis.libero_state import (
    FULL_SIM_DATA_FIELDS,
    capture_libero_state,
    restore_libero_state,
    validate_libero_round_trip,
)
from smolvla_analysis.model_inspection import resolve_pathways
from smolvla_analysis.phase2_capture import capture_action_query, fixed_flow_noise
from smolvla_analysis.phase2_storage import (
    read_libero_snapshot,
    write_action_query,
    write_libero_snapshot,
)
from smolvla_analysis.phase3_crd import (
    CONTINUATION_SCHEDULES,
    DEFAULT_STATE_SPECS,
    FACTOR_CONDITIONS,
    GOAL_CONTRADICTIONS,
    GOAL_INSTRUCTIONS,
    GOAL_PARAPHRASES,
    GOAL_PREDICATES,
    MUJOCO_STATE_ATOL,
    NUMERIC_OBSERVATION_ATOL,
    PIXEL_OBSERVATION_ATOL,
    PROPOSAL_SEEDS,
    BranchSpec,
    StateSpec,
    atomic_write_json,
    continuation_seed,
    certificate_within_tolerance,
    evaluate_common_goals,
    expected_query_ids,
    factor_query_id,
    is_monotonic_branch_source_upgrade,
    is_monotonic_numeric_tolerance_relaxation,
    is_monotonic_state_capture_upgrade,
    iter_branch_specs,
    legacy_cross_instance_branch_ids,
    nested_field_max_abs_differences,
    predicted_archive_init_state,
    query_summary,
    validate_paired_first_plan,
    validate_branch_accounting,
    validate_query_accounting,
)
from smolvla_analysis.runtime import _asset_path, _prepare_libero_runtime_config, load_runtime


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT / "archive/full_experiment/runs/benchmark_400_20260723T021424Z_e0638bea"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the certified two-task Phase 3 CRD smoke matrix.")
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/base.yaml")
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--branch-horizon", type=int, default=150)
    parser.add_argument("--certificate-steps", type=int, default=3)
    parser.add_argument(
        "--stop-after-branches",
        type=int,
        help="Stop after this many total completed branches; use for an in-place resumable smoke.",
    )
    parser.add_argument(
        "--allow-state-contract-upgrade",
        action="store_true",
        help="Resume when full MuJoCo solver/control state is the sole contract addition.",
    )
    parser.add_argument(
        "--allow-branch-source-upgrade",
        action="store_true",
        help="Resume when per-process archive replay is the sole branch-source contract addition.",
    )
    parser.add_argument("--skip-factor-queries", action="store_true")
    parser.add_argument(
        "--audit-branch-id",
        help="Replay one completed branch from a current-process archive reconstruction without changing the ledger.",
    )
    parser.add_argument(
        "--refresh-legacy-cross-instance-branches",
        action="store_true",
        help="Preserve and regenerate only the 30 branches invalidated by the old cross-process restore path.",
    )
    parser.add_argument(
        "--refresh-uncertified-branches",
        action="store_true",
        help="Preserve and regenerate active branches that lack per-process source reconstruction provenance.",
    )
    parser.add_argument(
        "--allow-monotonic-tolerance-migration",
        action="store_true",
        help="Resume an incomplete run only when the numeric certificate tolerance is the sole relaxed field.",
    )
    return parser.parse_args()


def _safe_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _contract(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    source_run = args.source_run.resolve()
    source_manifest = source_run / "manifest.json"
    source_episodes = source_run / "episodes.parquet"
    if not source_manifest.is_file() or not source_episodes.is_file():
        raise FileNotFoundError(f"Incomplete canonical source run: {source_run}")
    return {
        "schema_version": 2,
        "source_run": str(source_run),
        "source_manifest_sha256": _file_sha256(source_manifest),
        "source_episodes_sha256": _file_sha256(source_episodes),
        "model_repo_id": config["model"]["repo_id"],
        "model_revision": config["model"]["revision"],
        "branch_horizon": args.branch_horizon,
        "certificate_steps": args.certificate_steps,
        "certificate_tolerances": {
            "mujoco_state_atol": MUJOCO_STATE_ATOL,
            "numeric_observation_atol": NUMERIC_OBSERVATION_ATOL,
            "pixel_observation_atol": PIXEL_OBSERVATION_ATOL,
        },
        "image_resolution": [256, 256],
        "full_sim_data_fields": list(FULL_SIM_DATA_FIELDS),
        "branch_source_reconstruction": "archive_action_replay_current_process",
        "goals": {
            goal: {
                "predicate": list(GOAL_PREDICATES[goal]),
                "instruction": GOAL_INSTRUCTIONS[goal],
                "paraphrase": GOAL_PARAPHRASES[goal],
                "contradiction": GOAL_CONTRADICTIONS[goal],
            }
            for goal in GOAL_PREDICATES
        },
        "proposal_seeds": list(PROPOSAL_SEEDS),
        "continuation_schedules": {
            str(schedule): [continuation_seed(schedule, index) for index in range(3)]
            for schedule in CONTINUATION_SCHEDULES
        },
        "factor_conditions": list(FACTOR_CONDITIONS),
        "activation_relative_layer_positions": config["activations"]["relative_layer_positions"],
        "state_specs": [spec.__dict__ for spec in DEFAULT_STATE_SPECS],
        "branch_ids": [branch.branch_id for branch in iter_branch_specs()],
        "query_ids": list(expected_query_ids()),
    }


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _initialize_run(args: argparse.Namespace, config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    if args.branch_horizon < 1 or args.certificate_steps < 1:
        raise ValueError("branch-horizon and certificate-steps must be positive")
    contract = _contract(args, config)
    contract_sha = _fingerprint(contract)
    run_dir = args.run_dir
    if run_dir is None:
        run_id = datetime.now(UTC).strftime("phase3_crd_%Y%m%dT%H%M%SZ")
        run_dir = PROJECT / "local/phase3_crd" / run_id
    run_dir = run_dir.resolve()
    archive_root = (PROJECT / "archive").resolve()
    if run_dir == archive_root or archive_root in run_dir.parents:
        raise ValueError("Phase 3 run output must not be placed under immutable archive/")
    source_run = args.source_run.resolve()
    if run_dir == source_run or source_run in run_dir.parents:
        raise ValueError("Phase 3 run output must not be placed inside its canonical source run")
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("contract_sha256") != contract_sha:
            existing_contract = manifest.get("contract", {})
            tolerance_migration = (
                args.allow_monotonic_tolerance_migration
                and is_monotonic_numeric_tolerance_relaxation(existing_contract, contract)
            )
            state_upgrade = (
                args.allow_state_contract_upgrade
                and is_monotonic_state_capture_upgrade(existing_contract, contract)
            )
            branch_source_upgrade = (
                args.allow_branch_source_upgrade
                and is_monotonic_branch_source_upgrade(existing_contract, contract)
            )
            if not (tolerance_migration or state_upgrade or branch_source_upgrade):
                raise ValueError("Existing Phase 3 run has a different scientific contract")
            old_sha = manifest["contract_sha256"]
            if tolerance_migration:
                amendment = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "kind": "monotonic_numeric_observation_tolerance_relaxation",
                    "old_contract_sha256": old_sha,
                    "new_contract_sha256": contract_sha,
                    "old_numeric_observation_atol": existing_contract[
                        "certificate_tolerances"
                    ]["numeric_observation_atol"],
                    "new_numeric_observation_atol": contract["certificate_tolerances"][
                        "numeric_observation_atol"
                    ],
                    "reason": (
                        "Bit-identical cameras and sub-threshold MuJoCo state were accompanied by "
                        "derived gripper-velocity roundoff above the original observation cutoff."
                    ),
                }
            elif state_upgrade:
                amendment = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "kind": "full_simulator_runtime_state_upgrade",
                    "old_contract_sha256": old_sha,
                    "new_contract_sha256": contract_sha,
                    "added_fields": contract["full_sim_data_fields"],
                    "reason": (
                        "A contact-sensitive state exposed divergence from MuJoCo solver warm-start "
                        "and applied-control fields omitted by flattened qpos/qvel snapshots."
                    ),
                }
            else:
                amendment = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "kind": "per_process_archive_reconstruction_upgrade",
                    "old_contract_sha256": old_sha,
                    "new_contract_sha256": contract_sha,
                    "branch_source_reconstruction": contract["branch_source_reconstruction"],
                    "reason": (
                        "A cold cross-instance restore failed before subsequent restores stabilized; "
                        "every process must reconstruct branch roots by exact archived-action replay."
                    ),
                }
            manifest.setdefault("contract_amendments", []).append(amendment)
            manifest["contract"] = contract
            manifest["contract_sha256"] = contract_sha
            manifest["updated_at"] = datetime.now(UTC).isoformat()
            atomic_write_json(manifest_path, manifest)
        return run_dir, manifest
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "contract": contract,
        "contract_sha256": contract_sha,
        "expected_branches": len(contract["branch_ids"]),
        "expected_queries": len(contract["query_ids"]),
        "completed_branches": 0,
        "completed_queries": 0,
        "errors": [],
    }
    atomic_write_json(manifest_path, manifest)
    return run_dir, manifest


def _update_manifest(run_dir: Path, manifest: dict[str, Any], **updates: Any) -> None:
    manifest.update(updates)
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    atomic_write_json(run_dir / "manifest.json", manifest)


def _prepare_libero_config(run_dir: Path) -> Path:
    destination = run_dir / "runtime_libero_config"
    if not destination.exists():
        _prepare_libero_runtime_config(PROJECT, destination)
    os.environ["LIBERO_CONFIG_PATH"] = str(destination)
    os.environ["SMOLVLA_LIBERO_ASSETS"] = str(_asset_path(PROJECT, "checkpoints/libero_assets"))
    return destination


def _make_environment(task_id: int, policy_cfg):
    import libero.libero as libero_runtime
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors

    libero_runtime._assets_path_cache = os.environ["SMOLVLA_LIBERO_ASSETS"]
    env_cfg = LiberoEnv(
        task="libero_goal",
        task_ids=[task_id],
        episode_length=300,
        obs_type="pixels_agent_pos",
        observation_height=256,
        observation_width=256,
        control_mode="relative",
        max_parallel_tasks=1,
    )
    environment = make_env(env_cfg, n_envs=1, use_async_envs=False)["libero_goal"][task_id]
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, policy_cfg)
    return environment, env_preprocessor, env_postprocessor


def _archive_fidelity(
    observation: dict[str, Any],
    policy_observation: dict[str, Any],
    archive: Any,
    step: int,
) -> dict[str, float | bool]:
    live_joint = np.asarray(observation["robot_state"]["joints"]["pos"])[0].astype(np.float32)
    live_main = np.asarray(observation["pixels"]["image"])[0].transpose(2, 0, 1)
    live_wrist = np.asarray(observation["pixels"]["image2"])[0].transpose(2, 0, 1)
    live_policy_state = policy_observation["observation.state"][0].detach().cpu().numpy()
    joint_difference = np.abs(live_joint - archive["robot_state"][step])
    state_difference = np.abs(live_policy_state - archive["policy_state"][step])
    main_difference = np.abs(live_main.astype(np.int16) - archive["camera1"][step].astype(np.int16))
    wrist_difference = np.abs(live_wrist.astype(np.int16) - archive["camera2"][step].astype(np.int16))
    return {
        "joint_max_abs": float(joint_difference.max(initial=0.0)),
        "policy_state_max_abs": float(state_difference.max(initial=0.0)),
        "main_pixel_max_abs": int(main_difference.max(initial=0)),
        "wrist_pixel_max_abs": int(wrist_difference.max(initial=0)),
        "exact": bool(
            np.array_equal(live_joint, archive["robot_state"][step])
            and np.array_equal(live_policy_state, archive["policy_state"][step])
            and not main_difference.any()
            and not wrist_difference.any()
        ),
    }


def _policy_observation(observation: dict[str, Any], instruction: str, env_preprocessor):
    from lerobot.envs.utils import preprocess_observation

    converted = preprocess_observation(observation)
    converted["task"] = [instruction]
    return env_preprocessor(converted)


def _batch_scalar_observation(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _batch_scalar_observation(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value[None]
    return value


def _direct_step(environment, action: np.ndarray):
    scalar = environment.envs[0]
    raw, reward, done, info = scalar._env.step(np.asarray(action, dtype=np.float32))
    return scalar._format_raw_obs(raw), float(reward), bool(done), info


def _run_probe(environment, actions: np.ndarray) -> dict[str, Any]:
    observation = None
    rewards = []
    done_values = []
    for action in actions:
        observation, reward, done, _ = _direct_step(environment, action)
        rewards.append(reward)
        done_values.append(done)
    snapshot = capture_libero_state(environment)
    return {
        "mujoco_state": snapshot.mujoco_state,
        "observation": observation,
        "rewards": rewards,
        "done": done_values,
        "goals": evaluate_common_goals(environment),
    }


def _branch_certificate(environment, source, actions: np.ndarray) -> dict[str, Any]:
    restore_libero_state(environment, source)
    first = _run_probe(environment, actions)
    restore_libero_state(environment, source)
    second = _run_probe(environment, actions)
    differences = nested_field_max_abs_differences(
        first["observation"], second["observation"]
    )
    state_difference = float(
        np.max(np.abs(first["mujoco_state"] - second["mujoco_state"]), initial=0.0)
    )
    passed = bool(
        certificate_within_tolerance(state_difference, differences)
        and first["rewards"] == second["rewards"]
        and first["done"] == second["done"]
        and first["goals"] == second["goals"]
    )
    restore_libero_state(environment, source)
    return {
        "pass": passed,
        "max_abs_mujoco_state_diff": state_difference,
        "max_abs_observation_diff": max(differences.values(), default=0.0),
        "tolerances": {
            "mujoco_state_atol": MUJOCO_STATE_ATOL,
            "numeric_observation_atol": NUMERIC_OBSERVATION_ATOL,
            "pixel_observation_atol": PIXEL_OBSERVATION_ATOL,
        },
        "observation_field_max_abs_diff": differences,
        "rewards_match": first["rewards"] == second["rewards"],
        "done_match": first["done"] == second["done"],
        "goals_match": first["goals"] == second["goals"],
    }


def _resolve_init_state(
    environment,
    env_preprocessor,
    episode_table: pd.DataFrame,
    spec: StateSpec,
    archive,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    task_episodes = episode_table[
        (episode_table["suite"] == "libero_goal")
        & (episode_table["task_id"] == spec.source_task_id)
        & (episode_table["episode_index"] < spec.source_episode_index)
    ]
    hint = predicted_archive_init_state(spec.source_episode_index, int(task_episodes["success"].sum()))
    scalar = environment.envs[0]
    candidates = [hint, *[index for index in range(len(scalar._init_states)) if index != hint]]
    attempts = []
    instruction = str(episode_table.loc[spec.source_episode_id, "instruction"])
    for candidate in candidates:
        scalar.init_state_id = candidate
        observation, _ = environment.reset(seed=[spec.source_seed])
        processed = _policy_observation(observation, instruction, env_preprocessor)
        fidelity = _archive_fidelity(observation, processed, archive, 0)
        attempts.append({"candidate": candidate, **fidelity})
        if fidelity["exact"]:
            return candidate, observation, {"hint": hint, "attempts": attempts}
    best = min(
        attempts,
        key=lambda item: (
            item["joint_max_abs"] + item["policy_state_max_abs"],
            item["main_pixel_max_abs"] + item["wrist_pixel_max_abs"],
        ),
    )
    raise RuntimeError(f"No exact archive init-state match for {spec.source_episode_id}; best={best}")


def _store_state(
    state_store,
    spec: StateSpec,
    snapshot,
    policy_observation: dict[str, Any],
    raw_observation: dict[str, Any],
    source_actions: np.ndarray,
    provenance: dict[str, Any],
) -> None:
    if spec.state_id in state_store:
        raise ValueError(f"Refusing to overwrite state artifact: {spec.state_id}")
    partial_key = f".partial__{spec.state_id}"
    if partial_key in state_store:
        del state_store[partial_key]
    write_libero_snapshot(state_store, partial_key, snapshot)
    group = state_store[partial_key]
    group.create_dataset(
        "policy_state", data=policy_observation["observation.state"][0].cpu().numpy(), overwrite=False
    )
    group.create_dataset(
        "policy_image", data=policy_observation["observation.images.image"][0].cpu().numpy(), overwrite=False
    )
    group.create_dataset(
        "policy_image2", data=policy_observation["observation.images.image2"][0].cpu().numpy(), overwrite=False
    )
    group.create_dataset("source_action_prefix", data=source_actions[: spec.landmark_step], overwrite=False)
    group.create_dataset(
        "source_eef_pos", data=np.asarray(raw_observation["robot_state"]["eef"]["pos"])[0], overwrite=False
    )
    group.attrs["state_spec_json"] = json.dumps(spec.__dict__, sort_keys=True)
    group.attrs["provenance_json"] = json.dumps(provenance, sort_keys=True, default=_safe_json)
    group.attrs["complete"] = True
    state_store.move(partial_key, spec.state_id)


def _validate_persisted_state(
    group,
    spec: StateSpec,
    snapshot,
    policy_observation: dict[str, Any],
    source_actions: np.ndarray,
) -> dict[str, Any]:
    if group.attrs.get("complete") is not True:
        raise RuntimeError(f"Incomplete persisted state: {spec.state_id}")
    if json.loads(group.attrs["state_spec_json"]) != spec.__dict__:
        raise RuntimeError(f"Persisted state specification changed: {spec.state_id}")
    stored_mujoco = np.asarray(group["mujoco_state"][:], dtype=np.float64)
    mujoco_difference = float(
        np.abs(stored_mujoco - snapshot.mujoco_state).max(initial=0.0)
    )
    comparisons = {
        "policy_state_exact": np.array_equal(
            group["policy_state"][:], policy_observation["observation.state"][0].cpu().numpy()
        ),
        "policy_image_exact": np.array_equal(
            group["policy_image"][:], policy_observation["observation.images.image"][0].cpu().numpy()
        ),
        "policy_image2_exact": np.array_equal(
            group["policy_image2"][:], policy_observation["observation.images.image2"][0].cpu().numpy()
        ),
        "source_action_prefix_exact": np.array_equal(
            group["source_action_prefix"][:], source_actions[: spec.landmark_step]
        ),
    }
    if mujoco_difference > MUJOCO_STATE_ATOL or not all(comparisons.values()):
        raise RuntimeError(
            f"Persisted state disagrees with current archive replay for {spec.state_id}: "
            f"mujoco_difference={mujoco_difference}, comparisons={comparisons}"
        )
    return {"mujoco_max_abs_diff": mujoco_difference, **comparisons}


def _reconstruct_source_episode_states(
    environment,
    env_preprocessor,
    episode_table: pd.DataFrame,
    source_run: Path,
    state_store,
    specs: list[StateSpec],
    certificate_steps: int,
) -> dict[str, tuple[Any, dict[str, Any]]]:
    if not specs:
        raise ValueError("At least one state specification is required")
    representative = specs[0]
    episode = episode_table.loc[representative.source_episode_id]
    with np.load(source_run / episode.observation_path) as loaded:
        archive = {name: loaded[name] for name in loaded.files}
    source_actions = np.asarray(archive["executed_actions"], dtype=np.float32)
    resolved, _, resolution = _resolve_init_state(
        environment, env_preprocessor, episode_table, representative, archive
    )
    scalar = environment.envs[0]
    scalar.init_state_id = resolved
    observation, _ = environment.reset(seed=[representative.source_seed])
    instruction = str(episode["instruction"])
    initial_policy = _policy_observation(observation, instruction, env_preprocessor)
    initial_fidelity = _archive_fidelity(observation, initial_policy, archive, 0)
    if not initial_fidelity["exact"]:
        raise RuntimeError(f"Resolved init state did not replay exactly: {representative.source_episode_id}")

    by_step = {spec.landmark_step: spec for spec in specs}
    maximum_step = max(by_step)
    reconstructed = {}
    for action_index in range(maximum_step):
        observation, _, terminated, truncated, _ = environment.step(source_actions[action_index][None])
        if bool(terminated[0] or truncated[0]):
            raise RuntimeError(
                f"Source replay terminated before landmark {action_index + 1}: "
                f"{representative.source_episode_id}"
            )
        step = action_index + 1
        if step not in by_step:
            continue
        spec = by_step[step]
        processed = _policy_observation(observation, instruction, env_preprocessor)
        archive_fidelity = _archive_fidelity(observation, processed, archive, step)
        if not archive_fidelity["exact"]:
            raise RuntimeError(f"Archive replay mismatch at {spec.state_id}: {archive_fidelity}")
        snapshot = capture_libero_state(environment)
        round_trip = validate_libero_round_trip(environment, snapshot, atol=1e-10)
        probe_actions = source_actions[step : step + certificate_steps]
        if len(probe_actions) != certificate_steps:
            raise RuntimeError(
                f"Insufficient archived probe actions at {spec.state_id}: "
                f"expected={certificate_steps}, available={len(probe_actions)}"
            )
        certificate = _branch_certificate(environment, snapshot, probe_actions)
        if not certificate["pass"]:
            raise RuntimeError(f"Branch certificate failed at {spec.state_id}: {certificate}")
        goals = evaluate_common_goals(environment)
        provenance = {
            "resolved_init_state_id": resolved,
            "init_state_resolution": resolution,
            "initial_archive_fidelity": initial_fidelity,
            "landmark_archive_fidelity": archive_fidelity,
            "round_trip": round_trip,
            "branch_certificate": certificate,
            "common_goals": goals,
            "source_actions_sha256": sha256(source_actions.tobytes()).hexdigest(),
            "runtime_reconstruction": "archive_action_replay_current_process",
            "snapshot_mujoco_sha256": sha256(snapshot.mujoco_state.tobytes()).hexdigest(),
        }
        if spec.state_id in state_store:
            provenance["persisted_state_validation"] = _validate_persisted_state(
                state_store[spec.state_id], spec, snapshot, processed, source_actions
            )
        else:
            _store_state(
                state_store,
                spec,
                snapshot,
                processed,
                observation,
                source_actions,
                provenance,
            )
            provenance["persisted_state_validation"] = {"created_in_current_process": True}
        reconstructed[spec.state_id] = (snapshot, provenance)
    return reconstructed


def _state_observation(group, goal: str, factor: str = "original") -> dict[str, Any]:
    image = torch.from_numpy(np.asarray(group["policy_image"][:], dtype=np.float32)).unsqueeze(0)
    image2 = torch.from_numpy(np.asarray(group["policy_image2"][:], dtype=np.float32)).unsqueeze(0)
    instruction = GOAL_INSTRUCTIONS[goal]
    if factor == "paraphrase":
        instruction = GOAL_PARAPHRASES[goal]
    elif factor == "contradiction":
        instruction = GOAL_CONTRADICTIONS[goal]
    elif factor == "main_mean":
        image = image.mean(dim=(2, 3), keepdim=True).expand_as(image).clone()
    elif factor == "wrist_mean":
        image2 = image2.mean(dim=(2, 3), keepdim=True).expand_as(image2).clone()
    elif factor != "original":
        raise ValueError(f"Unknown factor: {factor}")
    return {
        "observation.state": torch.from_numpy(
            np.asarray(group["policy_state"][:], dtype=np.float32)
        ).unsqueeze(0),
        "observation.images.image": image,
        "observation.images.image2": image2,
        "task": [instruction],
    }


def _postprocess_chunk(chunk: np.ndarray, device: str, postprocessor, env_postprocessor) -> np.ndarray:
    result = []
    for action_index in range(chunk.shape[1]):
        action = torch.as_tensor(chunk[:, action_index], dtype=torch.float32, device=device)
        action = postprocessor(action)
        result.append(env_postprocessor({"action": action})["action"].detach().cpu().numpy())
    return np.stack(result, axis=1)


def _ensure_query(
    *,
    query_id: str,
    state_group,
    goal: str,
    factor: str,
    noise_seed: int,
    policy,
    policy_cfg,
    preprocessor,
    postprocessor,
    env_postprocessor,
    targets,
    query_store,
    summary_dir: Path,
) -> np.ndarray:
    summary_path = summary_dir / f"{query_id}.json"
    if query_id in query_store:
        group = query_store[query_id]
        if group.attrs.get("complete") is not True or not summary_path.exists():
            raise RuntimeError(f"Incomplete query artifact requires manual audit: {query_id}")
        if group.attrs.get("query_id") != query_id:
            raise RuntimeError(f"Query identity mismatch in persisted artifact: {query_id}")
        return np.asarray(group["environment_action_chunk"][:], dtype=np.float32)
    partial_key = f".partial__{query_id}"
    if partial_key in query_store:
        del query_store[partial_key]
    observation = _state_observation(state_group, goal, factor)
    batch = preprocessor(deepcopy(observation))
    policy.reset()
    _, query = capture_action_query(
        policy,
        batch,
        targets,
        query_id=query_id,
        flow_noise_seed=noise_seed,
    )
    environment_chunk = _postprocess_chunk(
        query.model_action_chunk, policy_cfg.device, postprocessor, env_postprocessor
    )
    summary = {
        **query_summary(query, environment_chunk),
        "state_id": str(state_group.name).rsplit("/", 1)[-1],
        "goal": goal,
        "factor": factor,
    }
    write_action_query(
        query_store,
        query,
        environment_chunk,
        group_key=partial_key,
    )
    atomic_write_json(summary_path, summary)
    query_store.move(partial_key, query_id)
    policy.reset()
    return environment_chunk


def _effect_state(environment, observation: dict[str, Any] | None, source_group) -> dict[str, Any]:
    snapshot = capture_libero_state(environment)
    bowl = np.asarray(snapshot.objects["akita_black_bowl_1"]["position"], dtype=np.float64)
    source_bowl = np.asarray(
        json.loads(source_group.attrs["metadata_json"])["objects"]["akita_black_bowl_1"]["position"],
        dtype=np.float64,
    )
    if observation is None:
        eef = np.asarray(source_group["source_eef_pos"][:], dtype=np.float64)
    else:
        eef = np.asarray(observation["robot_state"]["eef"]["pos"], dtype=np.float64)
    source_eef = np.asarray(source_group["source_eef_pos"][:], dtype=np.float64)
    return {
        "goals": evaluate_common_goals(environment),
        "bowl_position": bowl.tolist(),
        "bowl_displacement": (bowl - source_bowl).tolist(),
        "bowl_displacement_norm": float(np.linalg.norm(bowl - source_bowl)),
        "eef_position": eef.tolist(),
        "eef_displacement": (eef - source_eef).tolist(),
        "eef_displacement_norm": float(np.linalg.norm(eef - source_eef)),
        "grasped_objects": list(snapshot.grasped_objects),
        "contact_count": len(snapshot.contacts),
    }


def _live_policy_batch(observation, instruction, env_preprocessor, preprocessor):
    vector_observation = _batch_scalar_observation(observation)
    policy_observation = _policy_observation(vector_observation, instruction, env_preprocessor)
    return preprocessor(deepcopy(policy_observation))


def _run_branch(
    branch: BranchSpec,
    source,
    source_provenance: dict[str, Any],
    source_group,
    initial_chunk: np.ndarray,
    environment,
    env_preprocessor,
    env_postprocessor,
    policy,
    policy_cfg,
    preprocessor,
    postprocessor,
    horizon: int,
) -> dict[str, Any]:
    restore_libero_state(environment, source)
    restored_goals = evaluate_common_goals(environment)
    expected_goals = json.loads(source_group.attrs["provenance_json"])["common_goals"]
    if restored_goals != expected_goals:
        raise RuntimeError(f"Goal state changed on restore for {branch.state.state_id}")
    if set(policy._queues) != {"action"}:
        raise RuntimeError(f"Unexpected policy queue contract: {sorted(policy._queues)}")
    policy.reset()
    instruction = GOAL_INSTRUCTIONS[branch.target_goal]
    chunk = np.asarray(initial_chunk, dtype=np.float32)
    if chunk.shape == (1, 50, 7):
        chunk = chunk[0]
    if chunk.shape != (50, 7) or not np.isfinite(chunk).all():
        raise ValueError(f"Invalid initial environment chunk: {chunk.shape}")

    started = time.perf_counter()
    steps = 0
    replan_index = 0
    continuation_seeds = []
    continuation_noise_sha256 = []
    observation = None
    underlying_done = False
    goal_first_step = {goal: (0 if value else None) for goal, value in restored_goals.items()}
    first10_effect = None
    first_plan_effect = None
    target_success = restored_goals[branch.target_goal]

    while steps < horizon and not target_success and not underlying_done:
        active_chunk = chunk
        for action in active_chunk:
            if steps >= horizon:
                break
            observation, _, underlying_done, _ = _direct_step(environment, action)
            steps += 1
            goals = evaluate_common_goals(environment)
            for goal, value in goals.items():
                if value and goal_first_step[goal] is None:
                    goal_first_step[goal] = steps
            if steps == 10:
                first10_effect = _effect_state(environment, observation, source_group)
            target_success = goals[branch.target_goal]
            if target_success or underlying_done:
                break
        if first_plan_effect is None:
            first_plan_effect = _effect_state(environment, observation, source_group)
        if target_success or underlying_done or steps >= horizon:
            break
        seed = continuation_seed(branch.continuation_schedule, replan_index)
        continuation_seeds.append(seed)
        batch = _live_policy_batch(observation, instruction, env_preprocessor, preprocessor)
        noise = fixed_flow_noise(policy, batch, seed)
        noise_array = noise.detach().float().cpu().numpy()
        continuation_noise_sha256.append(
            sha256(np.ascontiguousarray(noise_array).tobytes()).hexdigest()
        )
        with torch.inference_mode():
            model_chunk = policy.predict_action_chunk(batch, noise=noise).detach().cpu().numpy()
        chunk = _postprocess_chunk(
            model_chunk, policy_cfg.device, postprocessor, env_postprocessor
        )[0]
        replan_index += 1

    final_effect = _effect_state(environment, observation, source_group)
    if first10_effect is None:
        first10_effect = final_effect
    if first_plan_effect is None:
        first_plan_effect = final_effect
    terminal_reason = (
        "target_success" if target_success else "underlying_done" if underlying_done else "horizon"
    )
    return {
        "schema_version": 1,
        **branch.metadata(),
        "success": bool(target_success),
        "initial_goal_status": restored_goals,
        "source_reconstruction": {
            "mode": source_provenance["runtime_reconstruction"],
            "snapshot_mujoco_sha256": source_provenance["snapshot_mujoco_sha256"],
            "landmark_archive_fidelity": source_provenance["landmark_archive_fidelity"],
            "branch_certificate": source_provenance["branch_certificate"],
            "persisted_state_validation": source_provenance["persisted_state_validation"],
        },
        "goal_first_step": goal_first_step,
        "steps_executed": steps,
        "first_plan_steps": min(steps, 50),
        "continuation_replans": replan_index,
        "continuation_seeds": continuation_seeds,
        "continuation_noise_sha256": continuation_noise_sha256,
        "terminal_reason": terminal_reason,
        "underlying_done": underlying_done,
        "first10_effect": first10_effect,
        "first_plan_effect": first_plan_effect,
        "final_effect": final_effect,
        "wall_time_s": time.perf_counter() - started,
    }


def _load_branch_payloads(directory: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]


def _compare_branch_payloads(existing: dict[str, Any], replayed: dict[str, Any]) -> dict[str, Any]:
    excluded = {"wall_time_s", "source_reconstruction"}
    existing_semantic = {key: value for key, value in existing.items() if key not in excluded}
    replayed_semantic = {key: value for key, value in replayed.items() if key not in excluded}
    keys = sorted(set(existing_semantic) | set(replayed_semantic))
    changed = [key for key in keys if existing_semantic.get(key) != replayed_semantic.get(key)]
    return {
        "semantic_exact_match": not changed,
        "excluded_fields": sorted(excluded),
        "changed_top_level_fields": changed,
    }


def _persisted_query_ids(query_store, summary_dir: Path) -> list[str]:
    identifiers = []
    for key in sorted(query_store.group_keys()):
        if key.startswith(".partial__"):
            continue
        group = query_store[key]
        if group.attrs.get("complete") is not True:
            raise RuntimeError(f"Query group is not marked complete: {key}")
        if group.attrs.get("query_id") != key:
            raise RuntimeError(f"Query group identity mismatch: key={key}")
        if "environment_action_chunk" not in group:
            raise RuntimeError(f"Query group lacks environment action chunk: {key}")
        if not (summary_dir / f"{key}.json").is_file():
            raise RuntimeError(f"Query group lacks atomic summary: {key}")
        identifiers.append(key)
    return identifiers


def _prepare_branch_refresh(
    run_dir: Path,
    manifest: dict[str, Any],
    branches_dir: Path,
    *,
    kind: str,
    targets: list[str],
    reason: str,
    audit_report: str | None = None,
) -> dict[str, Any]:
    refreshes = manifest.setdefault("branch_refreshes", [])
    entry = next((item for item in refreshes if item.get("kind") == kind), None)
    if entry is None:
        entry = {
            "kind": kind,
            "status": "preparing",
            "created_at": datetime.now(UTC).isoformat(),
            "reason": reason,
            "target_branch_ids": targets,
            "old_sha256": {},
        }
        if audit_report is not None:
            entry["audit_report"] = audit_report
        refreshes.append(entry)
        atomic_write_json(run_dir / "manifest.json", manifest)
    if entry["target_branch_ids"] != targets:
        raise RuntimeError(f"Branch-refresh target set changed for {kind}")
    if entry["status"] == "complete":
        return entry
    if entry["status"] == "preparing":
        backup_dir = run_dir / "superseded_branches" / kind
        backup_dir.mkdir(parents=True, exist_ok=True)
        for branch_id in targets:
            source = branches_dir / f"{branch_id}.json"
            backup = backup_dir / source.name
            if backup.exists():
                if source.exists():
                    raise RuntimeError(f"Ambiguous refresh artifacts for {branch_id}")
                entry["old_sha256"].setdefault(branch_id, _file_sha256(backup))
                continue
            if not source.is_file():
                raise FileNotFoundError(f"Missing legacy branch selected for refresh: {branch_id}")
            digest = _file_sha256(source)
            os.replace(source, backup)
            entry["old_sha256"][branch_id] = digest
            atomic_write_json(run_dir / "manifest.json", manifest)
        entry["status"] = "running"
        entry["started_at"] = datetime.now(UTC).isoformat()
        manifest["status"] = "running"
        manifest["completed_branches"] = len(list(branches_dir.glob("*.json")))
        atomic_write_json(run_dir / "manifest.json", manifest)
    elif entry["status"] != "running":
        raise RuntimeError(f"Unknown branch-refresh status: {entry['status']}")
    return entry


def _prepare_legacy_branch_refresh(
    run_dir: Path,
    manifest: dict[str, Any],
    branches_dir: Path,
) -> None:
    _prepare_branch_refresh(
        run_dir,
        manifest,
        branches_dir,
        kind="legacy_cross_instance_restore",
        targets=list(legacy_cross_instance_branch_ids()),
        reason=(
            "A deterministic audit showed that the old cold cross-process restore reversed a "
            "branch outcome; only branches rooted in the two disk-restored smoke states are affected."
        ),
        audit_report=(
            "reports/phase3_crd/phase3_crd_20260728T021125Z/branch_replay_audits/"
            "task03_ep000_step0050__goal_drawer__proposal_202__continuation_0.json"
        ),
    )


def _prepare_uncertified_branch_refresh(
    run_dir: Path,
    manifest: dict[str, Any],
    branches_dir: Path,
) -> None:
    kind = "legacy_uncertified_branch_roots"
    existing = next(
        (item for item in manifest.get("branch_refreshes", []) if item.get("kind") == kind),
        None,
    )
    if existing is not None:
        targets = list(existing["target_branch_ids"])
    else:
        targets = sorted(
            path.stem
            for path in branches_dir.glob("*.json")
            if "source_reconstruction" not in json.loads(path.read_text())
        )
        if len(targets) != 84:
            raise RuntimeError(f"Expected exactly 84 uncertified active branches, found {len(targets)}")
    _prepare_branch_refresh(
        run_dir,
        manifest,
        branches_dir,
        kind=kind,
        targets=targets,
        reason=(
            "Identical first chunks produced different first-plan effects across continuation schedules; "
            "all active payloads lacking per-process archive-reconstruction provenance are excluded."
        ),
    )


def _finalize_branch_refreshes(run_dir: Path, manifest: dict[str, Any], branches_dir: Path) -> None:
    for entry in manifest.get("branch_refreshes", []):
        if entry.get("status") != "running":
            continue
        targets = entry["target_branch_ids"]
        backup_dir = run_dir / "superseded_branches" / entry["kind"]
        if not all((branches_dir / f"{branch_id}.json").is_file() for branch_id in targets):
            continue
        if not all((backup_dir / f"{branch_id}.json").is_file() for branch_id in targets):
            raise RuntimeError("Branch-refresh backups are incomplete")
        entry["new_sha256"] = {
            branch_id: _file_sha256(branches_dir / f"{branch_id}.json") for branch_id in targets
        }
        entry["status"] = "complete"
        entry["completed_at"] = datetime.now(UTC).isoformat()
        atomic_write_json(run_dir / "manifest.json", manifest)


def main() -> int:
    args = _parse_args()
    if args.audit_branch_id and args.run_dir is None:
        raise ValueError("--audit-branch-id requires --run-dir")
    if args.audit_branch_id and (args.stop_after_branches or not args.skip_factor_queries):
        raise ValueError("Branch audit mode requires --skip-factor-queries and no stop-after limit")
    if args.audit_branch_id and (
        args.refresh_legacy_cross_instance_branches or args.refresh_uncertified_branches
    ):
        raise ValueError("Branch audit and branch refresh modes are mutually exclusive")
    if args.refresh_legacy_cross_instance_branches and args.refresh_uncertified_branches:
        raise ValueError("Run only one branch refresh transaction at a time")
    config = load_config(args.config)
    if config["model"]["device"] != "cuda":
        raise ValueError("The Phase 3 gate requires the checkpoint's CUDA path")
    os.environ.setdefault("MUJOCO_GL", "egl")
    run_dir, manifest = _initialize_run(args, config)
    _prepare_libero_config(run_dir)
    branches = iter_branch_specs()
    branches_dir = run_dir / "branches"
    summaries_dir = run_dir / "query_summaries"
    branches_dir.mkdir(exist_ok=True)
    summaries_dir.mkdir(exist_ok=True)
    state_store = zarr.open_group(str(run_dir / "states.zarr"), mode="a")
    query_store = zarr.open_group(str(run_dir / "queries.zarr"), mode="a")
    if args.refresh_legacy_cross_instance_branches:
        _prepare_legacy_branch_refresh(run_dir, manifest, branches_dir)
    if args.refresh_uncertified_branches:
        _prepare_uncertified_branch_refresh(run_dir, manifest, branches_dir)
    episode_table = pd.read_parquet(args.source_run / "episodes.parquet").set_index("episode_id")
    policy_cfg = policy = preprocessor = postprocessor = None
    environments = {}
    exit_code = 0
    try:
        policy_cfg, policy, preprocessor, postprocessor, model_load_time = load_runtime(config, PROJECT)
        if set(policy._queues) != {"action"}:
            raise RuntimeError(f"Unexpected checkpoint queues: {sorted(policy._queues)}")
        targets = resolve_pathways(policy, config["activations"]["relative_layer_positions"])
        manifest["model_load_time_s"] = model_load_time

        def environment_for(task_id: int):
            if task_id not in environments:
                environments[task_id] = _make_environment(task_id, policy_cfg)
            return environments[task_id]

        completed = _load_branch_payloads(branches_dir)
        validate_branch_accounting(branches, completed)
        completed_ids = {payload["branch_id"] for payload in completed}
        if args.audit_branch_id and args.audit_branch_id not in completed_ids:
            raise ValueError(f"Audit target is not a completed branch: {args.audit_branch_id}")
        runtime_states: dict[str, tuple[Any, dict[str, Any]]] = {}
        for branch in branches:
            if args.audit_branch_id and branch.branch_id != args.audit_branch_id:
                continue
            if not args.audit_branch_id and branch.branch_id in completed_ids:
                continue
            environment, env_preprocessor, env_postprocessor = environment_for(
                branch.state.source_task_id
            )
            if branch.state.state_id not in runtime_states:
                episode_specs = [
                    spec
                    for spec in DEFAULT_STATE_SPECS
                    if spec.source_episode_id == branch.state.source_episode_id
                ]
                runtime_states.update(
                    _reconstruct_source_episode_states(
                        environment,
                        env_preprocessor,
                        episode_table,
                        args.source_run,
                        state_store,
                        episode_specs,
                        args.certificate_steps,
                    )
                )
            source_group = state_store[branch.state.state_id]
            if source_group.attrs.get("complete") is not True:
                raise RuntimeError(f"Incomplete state artifact: {branch.state.state_id}")
            initial_chunk = _ensure_query(
                query_id=branch.query_id,
                state_group=source_group,
                goal=branch.target_goal,
                factor="original",
                noise_seed=branch.proposal_seed,
                policy=policy,
                policy_cfg=policy_cfg,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                env_postprocessor=env_postprocessor,
                targets=targets,
                query_store=query_store,
                summary_dir=summaries_dir,
            )
            source, source_provenance = runtime_states[branch.state.state_id]
            payload = _run_branch(
                branch,
                source,
                source_provenance,
                source_group,
                initial_chunk,
                environment,
                env_preprocessor,
                env_postprocessor,
                policy,
                policy_cfg,
                preprocessor,
                postprocessor,
                args.branch_horizon,
            )
            if args.audit_branch_id:
                existing = json.loads((branches_dir / f"{branch.branch_id}.json").read_text())
                audit = {
                    "schema_version": 1,
                    "run_id": manifest["run_id"],
                    "contract_sha256": manifest["contract_sha256"],
                    "branch_id": branch.branch_id,
                    "comparison": _compare_branch_payloads(existing, payload),
                    "existing": existing,
                    "replayed": payload,
                }
                audit_path = (
                    PROJECT
                    / "reports/phase3_crd"
                    / manifest["run_id"]
                    / "branch_replay_audits"
                    / f"{branch.branch_id}.json"
                )
                atomic_write_json(audit_path, audit)
                print(json.dumps({"output": str(audit_path), **audit["comparison"]}, indent=2))
                return 0
            sibling = next(
                (
                    existing
                    for existing in completed
                    if existing.get("query_id") == branch.query_id
                ),
                None,
            )
            if sibling is not None:
                validate_paired_first_plan(sibling, payload)
            atomic_write_json(branches_dir / f"{branch.branch_id}.json", payload)
            completed.append(payload)
            completed_ids.add(branch.branch_id)
            accounting = validate_branch_accounting(branches, completed)
            query_ids = _persisted_query_ids(query_store, summaries_dir)
            validate_query_accounting(query_ids)
            _update_manifest(
                run_dir,
                manifest,
                status="running",
                completed_branches=accounting["completed"],
                completed_queries=len(query_ids),
            )
            if args.stop_after_branches and accounting["completed"] >= args.stop_after_branches:
                break

        accounting = validate_branch_accounting(branches, completed)
        if accounting["complete"] and not args.skip_factor_queries:
            for state in DEFAULT_STATE_SPECS:
                environment, _, env_postprocessor = environment_for(state.source_task_id)
                source_group = state_store[state.state_id]
                for goal in GOAL_INSTRUCTIONS:
                    for factor in FACTOR_CONDITIONS:
                        query_id = factor_query_id(state.state_id, goal, factor)
                        _ensure_query(
                            query_id=query_id,
                            state_group=source_group,
                            goal=goal,
                            factor=factor,
                            noise_seed=101,
                            policy=policy,
                            policy_cfg=policy_cfg,
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                            env_postprocessor=env_postprocessor,
                            targets=targets,
                            query_store=query_store,
                            summary_dir=summaries_dir,
                        )
        query_ids = _persisted_query_ids(query_store, summaries_dir)
        query_accounting = validate_query_accounting(query_ids)
        final_status = (
            "complete"
            if accounting["complete"] and query_accounting["complete"] and not args.skip_factor_queries
            else "partial"
        )
        if accounting["complete"]:
            _finalize_branch_refreshes(run_dir, manifest, branches_dir)
        _update_manifest(
            run_dir,
            manifest,
            status=final_status,
            completed_branches=accounting["completed"],
            completed_queries=query_accounting["completed"],
            missing_queries=query_accounting["missing"],
            missing_branches=accounting["missing"],
        )
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "status": final_status,
                    "completed_branches": accounting["completed"],
                    "expected_branches": accounting["expected"],
                    "completed_queries": query_accounting["completed"],
                    "expected_queries": query_accounting["expected"],
                },
                indent=2,
            )
        )
    except Exception as error:
        manifest.setdefault("errors", []).append(
            {"timestamp": datetime.now(UTC).isoformat(), "type": type(error).__name__, "message": str(error)}
        )
        _update_manifest(run_dir, manifest, status="failed")
        raise
    finally:
        for environment, _, _ in environments.values():
            environment.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
