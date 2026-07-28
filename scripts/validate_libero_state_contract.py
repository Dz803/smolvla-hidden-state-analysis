#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from smolvla_analysis.libero_state import (
    capture_libero_state,
    restore_libero_state,
    validate_libero_round_trip,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate full LIBERO branch restoration against a MuJoCo-only baseline."
    )
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--branch-steps", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=Path("reports/phase2_state_gate"))
    return parser.parse_args()


def _asset_path(project: Path, relative: str) -> Path:
    for candidate in (project / relative, project / "archive/full_experiment" / relative):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(relative)


def _write_runtime_libero_config(project: Path, config_directory: Path) -> None:
    package_root = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages/libero/libero"
    )
    payload = {
        "benchmark_root": str(package_root),
        "bddl_files": str(package_root / "bddl_files"),
        "init_states": str(package_root / "init_files"),
        "datasets": str(_asset_path(project, "checkpoints/libero_datasets")),
        "assets": str(_asset_path(project, "checkpoints/libero_assets")),
    }
    config_directory.mkdir(parents=True, exist_ok=False)
    (config_directory / "config.yaml").write_text(json.dumps(payload, indent=2) + "\n")


def _make_environment(project: Path, suite: str, task_id: int, config_directory: Path):
    assets = _asset_path(project, "checkpoints/libero_assets")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_directory)
    os.environ["SMOLVLA_LIBERO_ASSETS"] = str(assets)

    import libero.libero as libero_runtime
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env

    libero_runtime._assets_path_cache = str(assets)
    env_config = LiberoEnv(
        task=suite,
        task_ids=[task_id],
        episode_length=50,
        obs_type="pixels_agent_pos",
        observation_height=128,
        observation_width=128,
        control_mode="relative",
        max_parallel_tasks=1,
    )
    return make_env(env_config, n_envs=1, use_async_envs=False)[suite][task_id]


def _branch_actions(count: int) -> list[np.ndarray]:
    base = np.asarray([0.08, -0.05, 0.04, 0.03, -0.02, 0.01, 1.0], dtype=np.float32)
    return [base * np.float32(1.0 - 0.1 * index) for index in range(count)]


def _run_branch(environment, actions: list[np.ndarray]) -> dict[str, Any]:
    observation = None
    rewards = []
    terminated = []
    for action in actions:
        observation, reward, term, trunc, _ = environment.step(action[None, :])
        rewards.append(float(np.asarray(reward)[0]))
        terminated.append(bool(np.asarray(term)[0] or np.asarray(trunc)[0]))
    snapshot = capture_libero_state(environment)
    return {
        "mujoco_state": snapshot.mujoco_state,
        "observation": observation,
        "rewards": rewards,
        "terminated": terminated,
        "gripper_current_action": snapshot.runtime_state["robots"][0].get(
            "gripper_current_action", []
        ),
    }


def _array_differences(left: Any, right: Any, prefix: str = "") -> dict[str, float]:
    if isinstance(left, dict):
        result = {}
        for key in sorted(left):
            result.update(_array_differences(left[key], right[key], f"{prefix}/{key}"))
        return result
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape or not np.issubdtype(left_array.dtype, np.number):
        return {}
    difference = np.abs(left_array.astype(np.float64) - right_array.astype(np.float64))
    return {prefix.lstrip("/"): float(np.max(difference, initial=0.0))}


def _compare_branches(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    observation_differences = _array_differences(left["observation"], right["observation"])
    return {
        "max_abs_mujoco_state_diff": float(
            np.max(np.abs(left["mujoco_state"] - right["mujoco_state"]), initial=0.0)
        ),
        "max_abs_observation_diff": max(observation_differences.values(), default=0.0),
        "observation_field_max_abs_diff": observation_differences,
        "reward_match": left["rewards"] == right["rewards"],
        "termination_match": left["terminated"] == right["terminated"],
        "gripper_current_action_left": left["gripper_current_action"],
        "gripper_current_action_right": right["gripper_current_action"],
    }


def main() -> None:
    args = _parse_args()
    project = Path(__file__).resolve().parents[1]
    os.environ.setdefault("MUJOCO_GL", "egl")
    with tempfile.TemporaryDirectory(prefix="smolvla-libero-state-") as temporary:
        config_directory = Path(temporary) / "config"
        _write_runtime_libero_config(project, config_directory)
        environment = _make_environment(project, args.suite, args.task_id, config_directory)
        try:
            environment.reset(seed=[args.seed])
            source = capture_libero_state(environment)
            immediate_round_trip = validate_libero_round_trip(environment, source, atol=1e-10)
            actions = _branch_actions(args.branch_steps)

            restore_libero_state(environment, source)
            physics_reference = _run_branch(environment, actions)
            environment.envs[0]._env.regenerate_obs_from_state(source.mujoco_state.copy())
            physics_replay = _run_branch(environment, actions)
            physics_only = _compare_branches(physics_reference, physics_replay)

            restore_libero_state(environment, source)
            full_reference = _run_branch(environment, actions)
            restore_libero_state(environment, source)
            full_replay = _run_branch(environment, actions)
            full_runtime = _compare_branches(full_reference, full_replay)

            full_pass = (
                full_runtime["max_abs_mujoco_state_diff"] <= 1e-10
                and full_runtime["max_abs_observation_diff"] == 0.0
                and full_runtime["reward_match"]
                and full_runtime["termination_match"]
            )
            physics_only_exposes_hidden_state = (
                physics_only["max_abs_mujoco_state_diff"] > 1e-10
                or physics_only["max_abs_observation_diff"] > 0.0
                or physics_only["gripper_current_action_left"]
                != physics_only["gripper_current_action_right"]
            )
            report = {
                "schema_version": 1,
                "timestamp": datetime.now(UTC).isoformat(),
                "suite": args.suite,
                "task_id": args.task_id,
                "seed": args.seed,
                "branch_steps": args.branch_steps,
                "action_sequence": [action.tolist() for action in actions],
                "captured_mujoco_state_length": int(source.mujoco_state.size),
                "captured_object_count": len(source.objects),
                "captured_goal_predicate_count": len(source.goal_predicates),
                "captured_runtime_robot_count": len(source.runtime_state.get("robots", [])),
                "immediate_round_trip": immediate_round_trip,
                "physics_only_replay": physics_only,
                "full_runtime_replay": full_runtime,
                "physics_only_exposes_hidden_state": physics_only_exposes_hidden_state,
                "full_runtime_replay_pass": full_pass,
            }
            run_id = datetime.now(UTC).strftime("state_contract_%Y%m%dT%H%M%SZ")
            output = project / args.output_root / run_id
            output.mkdir(parents=True, exist_ok=False)
            (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            print(json.dumps({"output": str(output), **report}, indent=2, sort_keys=True))
            if not full_pass:
                raise SystemExit("Full LIBERO runtime-state replay was not exact")
        finally:
            environment.close()


if __name__ == "__main__":
    main()
