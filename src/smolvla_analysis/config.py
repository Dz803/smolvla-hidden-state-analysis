from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .perturbations import CONDITIONS


SCHEMA: dict[str, Any] = {
    "project": {"name", "output_root", "run_name", "seed", "overwrite"},
    "model": {
        "repo_id", "revision", "local_path", "device", "dtype", "compile", "control_mode",
        "n_action_steps",
    },
    "benchmark": {
        "name", "suites", "task_ids", "episodes_per_task", "episode_seeds", "max_steps",
        "max_parallel_tasks", "mujoco_gl",
    },
    "recording": {
        "save_video", "save_observations", "save_environment_state", "save_action_chunks",
        "save_executed_actions", "save_latency", "save_gpu_stats", "step_log_format",
    },
    "activations": {
        "enabled", "capture_every_n_env_steps", "pathways", "relative_layer_positions",
        "save_pooled_features", "save_activation_norms", "save_token_statistics",
        "save_full_tokens_for_selected_episodes", "full_token_episode_fraction", "storage_dtype",
        "storage_format",
    },
    "uncertainty": {
        "enabled", "mode", "samples_per_observation", "sample_every_n_env_steps",
        "preserve_original_rollout",
    },
    "perturbations": {"condition", "parameters"},
    "analysis": {"bootstrap_samples", "confidence_level", "group_split_key", "random_seed"},
    "phase_thresholds": {
        "approach_distance_m", "alignment_distance_m", "grasp_distance_m", "lift_height_m",
        "goal_distance_m",
    },
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _read(path: Path, seen: set[Path]) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Configuration inheritance cycle at {path}")
    seen.add(path)
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    parent = raw.pop("extends", None)
    data = _read(path.parent / parent, seen) if parent else {}
    seen.remove(path)
    return _merge(data, raw)


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    unknown_sections = set(config) - set(SCHEMA)
    if unknown_sections:
        raise ValueError(f"Unsupported configuration sections: {sorted(unknown_sections)}")
    missing_sections = set(SCHEMA) - set(config)
    if missing_sections:
        raise ValueError(f"Missing configuration sections: {sorted(missing_sections)}")
    for section, allowed in SCHEMA.items():
        if not isinstance(config[section], dict):
            raise ValueError(f"{section} must be a mapping")
        unknown = set(config[section]) - allowed
        if unknown:
            raise ValueError(f"Unsupported fields in {section}: {sorted(unknown)}")

    project, model, benchmark = config["project"], config["model"], config["benchmark"]
    if project.get("overwrite") is not False:
        raise ValueError("project.overwrite must remain false for immutable runs")
    if model.get("control_mode") != "relative":
        raise ValueError("The checkpoint contract requires relative control")
    if model.get("device") not in {"cuda", "cpu"}:
        raise ValueError("model.device must be cuda or cpu")
    if benchmark.get("mujoco_gl") != "egl":
        raise ValueError("Headless LIBERO evaluation requires benchmark.mujoco_gl=egl")
    if benchmark.get("max_parallel_tasks") != 1:
        raise ValueError("LIBERO evaluation is restricted to max_parallel_tasks=1")
    if benchmark.get("episodes_per_task", 0) <= 0:
        raise ValueError("episodes_per_task must be positive")
    if len(benchmark.get("episode_seeds", [])) != benchmark["episodes_per_task"]:
        raise ValueError("episode_seeds length must equal episodes_per_task")
    positions = config["activations"].get("relative_layer_positions", [])
    if any(not 0 < float(position) <= 1 for position in positions):
        raise ValueError("relative_layer_positions must be in (0, 1]")
    condition = config["perturbations"].get("condition")
    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported perturbation condition: {condition}")
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    return validate_config(_read(Path(path), set()))
