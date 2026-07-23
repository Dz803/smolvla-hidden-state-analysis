from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import zarr

from .schema import validate_tables


REQUIRED_RUN_FILES = {"config_resolved.yaml", "manifest.json", "environment.json", "model_modules.json", "episodes.parquet", "steps.parquet"}


def validate_run(run_dir: str | Path, *, require_complete: bool = True) -> list[str]:
    root = Path(run_dir)
    errors = [f"missing file: {name}" for name in sorted(REQUIRED_RUN_FILES) if not (root / name).exists()]
    if errors:
        return errors
    manifest = json.loads((root / "manifest.json").read_text())
    if require_complete and manifest.get("completion_status") != "complete":
        errors.append("manifest is not complete")
    if manifest.get("completed_episode_count") != manifest.get("expected_episode_count"):
        errors.append("completed episode count does not match expected count")
    episodes = pd.read_parquet(root / "episodes.parquet")
    steps = pd.read_parquet(root / "steps.parquet")
    errors.extend(validate_tables(episodes, steps))
    benchmark = manifest.get("resolved_config", {}).get("benchmark", {})
    configured_pairs = {
        (suite, int(task_id))
        for suite in benchmark.get("suites", [])
        for task_id in benchmark.get("task_ids", [])
    }
    recorded_pairs = set(zip(episodes["suite"], episodes["task_id"].astype(int), strict=False))
    if recorded_pairs != configured_pairs:
        errors.append(f"recorded suite/task IDs do not match config: {sorted(recorded_pairs)}")
    expected_episode_keys = {
        (suite, int(task_id), int(seed))
        for suite, task_id in configured_pairs
        for seed in benchmark.get("episode_seeds", [])
    }
    recorded_episode_keys = set(
        zip(episodes["suite"], episodes["task_id"].astype(int), episodes["seed"].astype(int), strict=False)
    )
    if recorded_episode_keys != expected_episode_keys:
        errors.append("recorded suite/task/seed combinations do not match config")
    for _, episode in episodes.iterrows():
        episode_steps = steps.loc[steps["episode_id"] == episode["episode_id"]]
        if len(episode_steps) != int(episode["total_steps"]):
            errors.append(f"step count mismatch: {episode['episode_id']}")

        path = episode["video_path"]
        if pd.notna(path) and not (root / path).exists():
            errors.append(f"missing video: {path}")
        elif pd.notna(path):
            try:
                reader = imageio.get_reader(root / path)
                decoded_frames = sum(1 for frame in reader if frame.ndim == 3 and frame.shape[-1] in {3, 4})
                reader.close()
                if decoded_frames != int(episode["total_steps"]) + 1:
                    errors.append(f"video frame count mismatch: {path}")
            except Exception as exc:
                errors.append(f"unreadable video: {path} ({type(exc).__name__})")

        observation_path = episode["observation_path"]
        if pd.notna(observation_path) and not (root / observation_path).exists():
            errors.append(f"missing observation archive: {observation_path}")
        elif pd.notna(observation_path):
            with np.load(root / observation_path) as archive:
                for key in ("camera1", "camera2", "robot_state", "policy_state", "executed_actions"):
                    if key not in archive:
                        errors.append(f"observation archive missing {key}: {observation_path}")
                    elif len(archive[key]) != int(episode["total_steps"]):
                        errors.append(f"observation length mismatch for {key}: {observation_path}")
                    elif not np.isfinite(archive[key]).all():
                        errors.append(f"non-finite observation values for {key}: {observation_path}")

    for column in ("executed_action", "predicted_action_chunk"):
        for row_index, value in steps[column].items():
            nested_value = value.tolist() if isinstance(value, np.ndarray) else value
            if not np.isfinite(np.asarray(nested_value, dtype=float)).all():
                errors.append(f"non-finite values in steps.{column} at row {row_index}")

    activation_root = root / "activations.zarr"
    references = steps["activation_reference"].dropna().unique()
    if len(references) and not activation_root.exists():
        errors.append("missing activations.zarr")
    elif len(references):
        activation_store = zarr.open_group(str(activation_root), mode="r")
        target_count = len(json.loads((root / "model_modules.json").read_text())["resolved_targets"])
        checked_arrays = 0
        for reference in references:
            group_path = str(reference).removeprefix("activations.zarr/")
            try:
                group = activation_store[group_path]
            except KeyError:
                errors.append(f"missing activation reference: {reference}")
                continue
            arrays = list(group.arrays())
            if len(arrays) != target_count:
                errors.append(f"activation target count mismatch: {reference}")
            for _, array in arrays:
                checked_arrays += 1
                if not np.isfinite(array[:]).all():
                    errors.append(f"non-finite activation array: {reference}/{array.name}")
        recorded_count = sum(manifest.get("activation_arrays", {}).values())
        if checked_arrays != recorded_count:
            errors.append("activation array count does not match manifest")
    if episodes["infrastructure_failure"].fillna(False).any():
        errors.append("run contains infrastructure failures")
    return errors
