#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from smolvla_analysis.libero_state import capture_libero_state, restore_libero_state
from smolvla_analysis.phase2_storage import read_libero_snapshot
from smolvla_analysis.phase3_crd import (
    atomic_write_json,
    evaluate_common_goals,
    nested_field_max_abs_differences,
)
from smolvla_analysis.runtime import _asset_path


PROJECT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablate full-simulator fields from a certified Phase 3 state restore."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--state-id", default="task04_ep002_step0050")
    parser.add_argument("--probe-steps", type=int, default=3)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _make_environment(run_dir: Path, task_id: int):
    config_dir = run_dir / "runtime_libero_config"
    if not (config_dir / "config.yaml").is_file():
        raise FileNotFoundError(f"Missing run-local LIBERO config: {config_dir}")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    assets = _asset_path(PROJECT, "checkpoints/libero_assets")
    os.environ["SMOLVLA_LIBERO_ASSETS"] = str(assets)

    import libero.libero as libero_runtime
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env

    libero_runtime._assets_path_cache = str(assets)
    env_config = LiberoEnv(
        task="libero_goal",
        task_ids=[task_id],
        episode_length=300,
        obs_type="pixels_agent_pos",
        observation_height=256,
        observation_width=256,
        control_mode="relative",
        max_parallel_tasks=1,
    )
    return make_env(env_config, n_envs=1, use_async_envs=False)["libero_goal"][task_id]


def _direct_step(environment, action: np.ndarray):
    scalar = environment.envs[0]
    raw, reward, done, _ = scalar._env.step(np.asarray(action, dtype=np.float32))
    return scalar._format_raw_obs(raw), float(reward), bool(done)


def _run_probe(environment, actions: np.ndarray) -> dict[str, Any]:
    observation = None
    rewards = []
    done = []
    for action in actions:
        observation, reward, terminal = _direct_step(environment, action)
        rewards.append(reward)
        done.append(terminal)
    snapshot = capture_libero_state(environment)
    return {
        "mujoco_state": snapshot.mujoco_state,
        "observation": observation,
        "rewards": rewards,
        "done": done,
        "goals": evaluate_common_goals(environment),
    }


def _without_fields(snapshot, fields: set[str]):
    runtime = deepcopy(snapshot.runtime_state)
    for field in fields:
        runtime["sim_data"].pop(field, None)
    return replace(snapshot, runtime_state=runtime)


def _compare_condition(environment, source, candidate, actions: np.ndarray) -> dict[str, Any]:
    restore_libero_state(environment, source)
    reference = _run_probe(environment, actions)
    restore_libero_state(environment, candidate)
    replay = _run_probe(environment, actions)
    observation = nested_field_max_abs_differences(
        reference["observation"], replay["observation"]
    )
    return {
        "max_abs_mujoco_state_diff": float(
            np.abs(reference["mujoco_state"] - replay["mujoco_state"]).max(initial=0.0)
        ),
        "max_abs_observation_diff": max(observation.values(), default=0.0),
        "observation_field_max_abs_diff": observation,
        "rewards_match": reference["rewards"] == replay["rewards"],
        "done_match": reference["done"] == replay["done"],
        "goals_match": reference["goals"] == replay["goals"],
    }


def main() -> int:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    state_store = zarr.open_group(str(run_dir / "states.zarr"), mode="r")
    if args.state_id not in state_store or state_store[args.state_id].attrs.get("complete") is not True:
        raise ValueError(f"State is not complete: {args.state_id}")
    state_group = state_store[args.state_id]
    state_spec = json.loads(state_group.attrs["state_spec_json"])
    source = read_libero_snapshot(state_store, args.state_id)
    sim_fields = sorted(source.runtime_state.get("sim_data", {}))
    if not sim_fields:
        raise ValueError("Selected state has no full simulator fields")

    source_run = Path(manifest["contract"]["source_run"])
    observation_path = source_run / "observations" / f"{state_spec['source_episode_id']}.npz"
    with np.load(observation_path) as archive:
        start = int(state_spec["landmark_step"])
        actions = np.asarray(
            archive["executed_actions"][start : start + args.probe_steps], dtype=np.float32
        )
    if len(actions) != args.probe_steps:
        raise ValueError("Archive does not contain the requested probe horizon")

    environment = _make_environment(run_dir, int(state_spec["source_task_id"]))
    try:
        cold_start_full = _compare_condition(environment, source, source, actions)
        conditions = {"full": source, "drop_all_sim_data": _without_fields(source, set(sim_fields))}
        conditions.update(
            {f"drop_{field}": _without_fields(source, {field}) for field in sim_fields}
        )
        results = {
            name: _compare_condition(environment, source, candidate, actions)
            for name, candidate in conditions.items()
        }
    finally:
        environment.close()

    report = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "contract_sha256": manifest["contract_sha256"],
        "state_id": args.state_id,
        "probe_steps": args.probe_steps,
        "sim_fields": sim_fields,
        "cold_start_full": cold_start_full,
        "results": results,
    }
    output = args.output or (
        PROJECT / "reports/phase3_crd" / manifest["run_id"] / "restore_field_ablation.json"
    )
    output = output.resolve()
    atomic_write_json(output, report)
    print(json.dumps({"output": str(output), **report}, indent=2, sort_keys=True))
    if results["full"]["max_abs_mujoco_state_diff"] != 0.0:
        raise RuntimeError("Full-state positive control is not exact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
