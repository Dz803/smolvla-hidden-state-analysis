from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .libero_state import libero_problem_environment
from .phase2_capture import ActionQueryCapture


GOAL_PREDICATES = {
    "drawer": ("In", "akita_black_bowl_1", "wooden_cabinet_1_top_region"),
    "cabinet": ("On", "akita_black_bowl_1", "wooden_cabinet_1_top_side"),
}
GOAL_INSTRUCTIONS = {
    "drawer": "open the top drawer and put the bowl inside",
    "cabinet": "put the bowl on top of the cabinet",
}
GOAL_PARAPHRASES = {
    "drawer": "Open the upper drawer, then place the bowl inside it.",
    "cabinet": "Place the bowl on the top surface of the cabinet.",
}
GOAL_CONTRADICTIONS = {
    "drawer": (
        "Do not put the bowl on top of the cabinet. "
        "Instead, open the top drawer and put the bowl inside it."
    ),
    "cabinet": (
        "Do not open the top drawer or put the bowl inside it. "
        "Instead, put the bowl on top of the cabinet."
    ),
}
TASK_GOALS = {3: "drawer", 4: "cabinet"}
PROPOSAL_SEEDS = (101, 202, 303, 404)
CONTINUATION_SCHEDULES = (0, 1)
FACTOR_CONDITIONS = ("paraphrase", "contradiction", "main_mean", "wrist_mean")
MUJOCO_STATE_ATOL = 1e-10
NUMERIC_OBSERVATION_ATOL = 1e-10
PIXEL_OBSERVATION_ATOL = 0.0


@dataclass(frozen=True)
class StateSpec:
    state_id: str
    source_task_id: int
    source_episode_id: str
    source_episode_index: int
    source_seed: int
    landmark_step: int

    @property
    def source_goal(self) -> str:
        return TASK_GOALS[self.source_task_id]


DEFAULT_STATE_SPECS = (
    StateSpec("task03_ep000_step0050", 3, "libero_goal_task03_ep000_seed0", 0, 0, 50),
    StateSpec("task03_ep000_step0100", 3, "libero_goal_task03_ep000_seed0", 0, 0, 100),
    StateSpec("task03_ep001_step0050", 3, "libero_goal_task03_ep001_seed1", 1, 1, 50),
    StateSpec("task03_ep001_step0100", 3, "libero_goal_task03_ep001_seed1", 1, 1, 100),
    StateSpec("task03_ep003_step0100", 3, "libero_goal_task03_ep003_seed3", 3, 3, 100),
    StateSpec("task04_ep000_step0050", 4, "libero_goal_task04_ep000_seed0", 0, 0, 50),
    StateSpec("task04_ep001_step0050", 4, "libero_goal_task04_ep001_seed1", 1, 1, 50),
    StateSpec("task04_ep002_step0050", 4, "libero_goal_task04_ep002_seed2", 2, 2, 50),
    StateSpec("task04_ep003_step0050", 4, "libero_goal_task04_ep003_seed3", 3, 3, 50),
    StateSpec("task04_ep004_step0050", 4, "libero_goal_task04_ep004_seed4", 4, 4, 50),
)


@dataclass(frozen=True)
class BranchSpec:
    state: StateSpec
    target_goal: str
    proposal_seed: int
    continuation_schedule: int

    @property
    def query_id(self) -> str:
        return f"{self.state.state_id}__goal_{self.target_goal}__proposal_{self.proposal_seed}"

    @property
    def branch_id(self) -> str:
        return f"{self.query_id}__continuation_{self.continuation_schedule}"

    def metadata(self) -> dict[str, Any]:
        return {
            **asdict(self.state),
            "source_goal": self.state.source_goal,
            "target_goal": self.target_goal,
            "proposal_seed": self.proposal_seed,
            "continuation_schedule": self.continuation_schedule,
            "query_id": self.query_id,
            "branch_id": self.branch_id,
        }


def iter_branch_specs(
    states: Iterable[StateSpec] = DEFAULT_STATE_SPECS,
    *,
    goals: Iterable[str] = tuple(GOAL_PREDICATES),
    proposal_seeds: Iterable[int] = PROPOSAL_SEEDS,
    continuation_schedules: Iterable[int] = CONTINUATION_SCHEDULES,
) -> tuple[BranchSpec, ...]:
    branches = tuple(
        BranchSpec(state, goal, int(proposal_seed), int(schedule))
        for state in states
        for goal in goals
        for proposal_seed in proposal_seeds
        for schedule in continuation_schedules
    )
    identifiers = [branch.branch_id for branch in branches]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Branch identifiers are not unique")
    return branches


def stable_seed(*parts: Any, modulus: int = 2**63 - 1) -> int:
    payload = ":".join(str(part) for part in parts).encode()
    return int.from_bytes(sha256(payload).digest()[:8], "big") % modulus


def continuation_seed(schedule: int, replan_index: int) -> int:
    if schedule not in CONTINUATION_SCHEDULES:
        raise ValueError(f"Unknown continuation schedule: {schedule}")
    if replan_index < 0:
        raise ValueError("replan_index must be non-negative")
    return stable_seed("phase3-continuation", schedule, replan_index)


def factor_query_id(state_id: str, goal: str, factor: str, proposal_seed: int = 101) -> str:
    if goal not in GOAL_PREDICATES:
        raise ValueError(f"Unknown goal: {goal}")
    if factor not in FACTOR_CONDITIONS:
        raise ValueError(f"Unknown factor: {factor}")
    return f"{state_id}__goal_{goal}__factor_{factor}__proposal_{proposal_seed}"


def expected_query_ids(
    states: Iterable[StateSpec] = DEFAULT_STATE_SPECS,
    *,
    include_factors: bool = True,
) -> tuple[str, ...]:
    state_tuple = tuple(states)
    core = {branch.query_id for branch in iter_branch_specs(state_tuple)}
    factors = {
        factor_query_id(state.state_id, goal, factor)
        for state in state_tuple
        for goal in GOAL_PREDICATES
        for factor in FACTOR_CONDITIONS
    }
    identifiers = tuple(sorted(core | factors if include_factors else core))
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Query identifiers are not unique")
    return identifiers


def legacy_cross_instance_branch_ids() -> tuple[str, ...]:
    """Branches produced after the initial smoke from disk-restored task-3 states."""

    affected_states = {"task03_ep000_step0050", "task03_ep000_step0100"}
    valid_smoke = {
        "task03_ep000_step0050__goal_drawer__proposal_101__continuation_0",
        "task03_ep000_step0050__goal_drawer__proposal_101__continuation_1",
    }
    return tuple(
        branch.branch_id
        for branch in iter_branch_specs()
        if branch.state.state_id in affected_states and branch.branch_id not in valid_smoke
    )


def evaluate_common_goals(environment) -> dict[str, bool]:
    problem = libero_problem_environment(environment)
    return {
        name: bool(problem._eval_predicate((predicate[0].lower(), *predicate[1:])))
        for name, predicate in GOAL_PREDICATES.items()
    }


def predicted_archive_init_state(episode_index: int, preceding_successes: int) -> int:
    if episode_index < 0 or preceding_successes < 0 or preceding_successes > episode_index:
        raise ValueError("Invalid historical episode/reset counts")
    return int(episode_index + 2 * preceding_successes)


def certificate_within_tolerance(
    mujoco_state_difference: float,
    observation_differences: dict[str, float],
) -> bool:
    if not np.isfinite(mujoco_state_difference) or mujoco_state_difference > MUJOCO_STATE_ATOL:
        return False
    for name, difference in observation_differences.items():
        tolerance = PIXEL_OBSERVATION_ATOL if name.startswith("pixels/") else NUMERIC_OBSERVATION_ATOL
        if not np.isfinite(difference) or difference > tolerance:
            return False
    return True


def nested_field_max_abs_differences(
    left: Any, right: Any, prefix: str = ""
) -> dict[str, float]:
    """Compare nested observation fields without silently accepting schema changes."""

    name = prefix.lstrip("/") or "<root>"
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            return {name: float("inf")}
        result: dict[str, float] = {}
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}/{key}"
            if key not in left or key not in right:
                result[child.lstrip("/")] = float("inf")
            else:
                result.update(nested_field_max_abs_differences(left[key], right[key], child))
        return result
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        return {name: float("inf")}
    if np.issubdtype(left_array.dtype, np.number) and np.issubdtype(
        right_array.dtype, np.number
    ):
        difference = np.abs(left_array.astype(np.float64) - right_array.astype(np.float64))
        return {name: float(difference.max(initial=0.0))}
    return {name: 0.0 if np.array_equal(left_array, right_array) else float("inf")}


def is_monotonic_numeric_tolerance_relaxation(
    existing_contract: dict[str, Any],
    proposed_contract: dict[str, Any],
) -> bool:
    """Return true only when the numeric certificate tolerance is the sole relaxed field."""

    try:
        existing_tolerance = float(
            existing_contract["certificate_tolerances"]["numeric_observation_atol"]
        )
        proposed_tolerance = float(
            proposed_contract["certificate_tolerances"]["numeric_observation_atol"]
        )
    except (KeyError, TypeError, ValueError):
        return False
    if not np.isfinite(existing_tolerance) or not np.isfinite(proposed_tolerance):
        return False
    if proposed_tolerance < existing_tolerance:
        return False
    normalized = deepcopy(existing_contract)
    normalized["certificate_tolerances"]["numeric_observation_atol"] = proposed_tolerance
    return normalized == proposed_contract


def is_monotonic_state_capture_upgrade(
    existing_contract: dict[str, Any],
    proposed_contract: dict[str, Any],
) -> bool:
    """Return true only when full MuJoCo runtime fields are added to the contract."""

    fields = proposed_contract.get("full_sim_data_fields")
    if "full_sim_data_fields" in existing_contract or not isinstance(fields, list) or not fields:
        return False
    normalized = deepcopy(existing_contract)
    normalized["full_sim_data_fields"] = fields
    return normalized == proposed_contract


def is_monotonic_branch_source_upgrade(
    existing_contract: dict[str, Any],
    proposed_contract: dict[str, Any],
) -> bool:
    """Return true only when per-process archive replay is added as the branch source."""

    mode = proposed_contract.get("branch_source_reconstruction")
    if "branch_source_reconstruction" in existing_contract or mode != "archive_action_replay_current_process":
        return False
    normalized = deepcopy(existing_contract)
    normalized["branch_source_reconstruction"] = mode
    return normalized == proposed_contract


def action_statistics(action_chunk: np.ndarray) -> dict[str, float]:
    chunk = np.asarray(action_chunk, dtype=np.float64)
    if chunk.ndim != 2 or chunk.shape[1] != 7:
        raise ValueError(f"Expected an Hx7 action chunk, got {chunk.shape}")
    return {
        "plan_rms": float(np.sqrt(np.mean(np.square(chunk)))),
        "plan_translation_rms": float(np.sqrt(np.mean(np.square(chunk[:, :3])))),
        "plan_rotation_rms": float(np.sqrt(np.mean(np.square(chunk[:, 3:6])))),
        "plan_gripper_abs_mean": float(np.mean(np.abs(chunk[:, 6]))),
        "plan_temporal_std": float(np.mean(np.std(chunk, axis=0))),
        "plan_first10_rms": float(np.sqrt(np.mean(np.square(chunk[:10])))),
    }


def query_summary(query: ActionQueryCapture, environment_action_chunk: np.ndarray) -> dict[str, Any]:
    layers = sorted({record.layer_index for record in query.activations})
    if not layers:
        raise ValueError("Query capture contains no activation layers")
    layer = layers[-1]
    vlm = query.activation_stack("vlm", layer)[0, 0]
    mask = np.asarray(query.prefix_pad_mask[0], dtype=bool)
    expert = query.activation_stack("action_expert", layer)[-1, 0]
    head = np.asarray(query.action_head_inputs[-1, 0], dtype=np.float64)
    velocity = np.stack([record.velocity for record in query.denoising]).astype(np.float64)[:, 0]
    chunk = np.asarray(environment_action_chunk, dtype=np.float64)
    if chunk.ndim == 3:
        chunk = chunk[0]
    return {
        "schema_version": 1,
        "query_id": query.query_id,
        "flow_noise_seed": int(query.flow_noise_seed),
        "flow_noise_sha256": query.flow_noise_sha256,
        "layer": layer,
        "vlm_valid_token_count": int(mask.sum()),
        "vlm_pooled_norm": float(np.linalg.norm(vlm[mask].mean(axis=0))),
        "vlm_token_std": float(np.mean(np.std(vlm[mask].astype(np.float64), axis=0))),
        "expert_final_pooled_norm": float(np.linalg.norm(expert.mean(axis=0))),
        "expert_final_token_std": float(np.mean(np.std(expert.astype(np.float64), axis=0))),
        "action_head_input_rms": float(np.sqrt(np.mean(np.square(head)))),
        "velocity_executed_rms": float(np.sqrt(np.mean(np.square(velocity[..., :7])))),
        "velocity_padding_rms": float(np.sqrt(np.mean(np.square(velocity[..., 7:])))),
        "denoising_executed_path": float(
            np.sum(np.sqrt(np.mean(np.square(np.diff(velocity[..., :7], axis=0)), axis=(1, 2))))
        ),
        **action_statistics(chunk),
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate_paired_first_plan(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any]:
    """Fail closed when two continuation branches do not share one exact prefix."""

    left_query = left.get("query_id")
    right_query = right.get("query_id")
    if not left_query or left_query != right_query:
        raise ValueError(
            f"Paired branches must reference one query: left={left_query!r}, right={right_query!r}"
        )
    schedules = {left.get("continuation_schedule"), right.get("continuation_schedule")}
    if schedules != set(CONTINUATION_SCHEDULES):
        raise ValueError(
            f"Paired branches for {left_query} have invalid continuation schedules: {schedules}"
        )
    invariant_fields = (
        "initial_goal_status",
        "source_reconstruction",
        "first10_effect",
        "first_plan_steps",
        "first_plan_effect",
    )
    mismatches = [field for field in invariant_fields if left.get(field) != right.get(field)]
    if mismatches:
        raise ValueError(
            f"Paired first-plan invariant failed for {left_query}: fields={mismatches}"
        )
    return {
        "query_id": left_query,
        "continuation_schedules": sorted(int(item) for item in schedules),
        "exact": True,
    }


def validate_branch_accounting(
    expected: Iterable[BranchSpec], completed_payloads: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    expected_ids = {branch.branch_id for branch in expected}
    completed = list(completed_payloads)
    completed_ids = [str(payload.get("branch_id")) for payload in completed]
    duplicates = sorted({identifier for identifier in completed_ids if completed_ids.count(identifier) > 1})
    unexpected = sorted(set(completed_ids) - expected_ids)
    missing = sorted(expected_ids - set(completed_ids))
    if duplicates or unexpected:
        raise ValueError(f"Invalid branch ledger: duplicates={duplicates}, unexpected={unexpected}")
    return {
        "expected": len(expected_ids),
        "completed": len(completed_ids),
        "missing": missing,
        "complete": not missing,
    }


def validate_query_accounting(
    completed_ids: Iterable[str],
    *,
    states: Iterable[StateSpec] = DEFAULT_STATE_SPECS,
    include_factors: bool = True,
) -> dict[str, Any]:
    expected = set(expected_query_ids(states, include_factors=include_factors))
    completed = [str(identifier) for identifier in completed_ids]
    duplicates = sorted({identifier for identifier in completed if completed.count(identifier) > 1})
    unexpected = sorted(set(completed) - expected)
    missing = sorted(expected - set(completed))
    if duplicates or unexpected:
        raise ValueError(f"Invalid query ledger: duplicates={duplicates}, unexpected={unexpected}")
    return {
        "expected": len(expected),
        "completed": len(completed),
        "missing": missing,
        "complete": not missing,
    }
