from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable
from uuid import uuid4

import numpy as np
import pandas as pd


EPISODE_COLUMNS = (
    "run_id", "episode_id", "suite", "task_id", "task_name", "instruction", "episode_index",
    "seed", "initial_state_id", "condition", "success", "total_steps", "termination_reason",
    "failure_class", "failure_onset_step", "model_load_time_s", "wall_time_s", "peak_gpu_memory_mb",
    "latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "video_path", "observation_path",
    "activation_path", "infrastructure_failure",
)
STEP_COLUMNS = (
    "run_id", "episode_id", "env_step", "normalized_progress", "timestamp", "task_phase",
    "robot_state", "eef_state", "gripper_state", "object_state", "goal_state",
    "predicted_action_chunk", "executed_action", "action_norm", "action_smoothness", "action_jerk",
    "gripper_action", "policy_latency_ms", "gpu_memory_mb", "uncertainty_features",
    "activation_reference",
)
FAILURE_CLASSES = {
    "wrong object", "wrong target", "localization error", "premature gripper closure", "empty grasp",
    "dropped object", "collision", "unstable/oscillating action", "stalled behavior",
    "incorrect placement", "premature termination", "timeout", "simulator/infrastructure error", "unknown",
}


def unique_run_id(prefix: str = "run") -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}_{uuid4().hex[:8]}"


def create_run_directory(root: str | Path, run_id: str) -> Path:
    path = Path(root) / run_id
    path.mkdir(parents=True, exist_ok=False)
    for child in ("observations", "videos", "annotations", "summaries", "plots"):
        (path / child).mkdir()
    return path


def _missing(frame: pd.DataFrame, required: Iterable[str]) -> list[str]:
    return sorted(set(required) - set(frame.columns))


def validate_tables(episodes: pd.DataFrame, steps: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if missing := _missing(episodes, EPISODE_COLUMNS):
        errors.append(f"episodes missing columns: {missing}")
    if missing := _missing(steps, STEP_COLUMNS):
        errors.append(f"steps missing columns: {missing}")
    if errors:
        return errors
    if episodes["episode_id"].duplicated().any():
        errors.append("duplicate episode_id values")
    unknown_steps = set(steps["episode_id"]) - set(episodes["episode_id"])
    if unknown_steps:
        errors.append(f"steps reference unknown episodes: {sorted(unknown_steps)}")
    duplicate_steps = steps.duplicated(["episode_id", "env_step"])
    if duplicate_steps.any():
        errors.append("duplicate episode/env_step pairs")
    for column in ("normalized_progress", "action_norm", "policy_latency_ms", "gpu_memory_mb"):
        values = pd.to_numeric(steps[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            errors.append(f"non-finite values in steps.{column}")
    invalid_failures = set(episodes["failure_class"].dropna()) - FAILURE_CLASSES
    if invalid_failures:
        errors.append(f"unknown failure classes: {sorted(invalid_failures)}")
    return errors

