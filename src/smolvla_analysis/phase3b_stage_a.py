from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from typing import Any, Iterable

import numpy as np

from .libero_state import LiberoStateSnapshot


DRAWER_APERTURES = ("closed", "open")
POSSESSIONS = ("on_table", "grasped")
TRANSIT_LOCI = ("drawer_side", "cabinet_side")
SUPPORT_STRATA = ("demonstration_near", "transverse_low_support")
LAYOUTS = ("A", "B")
LAYOUT_INIT_STATE_IDS = {"A": 0, "B": 1}
GOALS = ("drawer", "cabinet")
SUPPORT_CATEGORICAL_FIELDS = (
    "layout",
    "drawer_aperture",
    "possession",
    "transit_locus",
    "motion_event",
)
SUPPORT_SCALE_FIELDS = (
    "eef_position_m",
    "eef_orientation_rad",
    "robot_joint_rms_rad",
    "bowl_position_m",
    "drawer_joint_m",
    "eef_motion_m",
    "bowl_motion_m",
    "action_motion_rms",
    "grasp_relative_position_m",
    "goal_distance_m",
)

FACTOR_LEVELS = {
    "drawer_aperture": DRAWER_APERTURES,
    "possession": POSSESSIONS,
    "transit_locus": TRANSIT_LOCI,
    "support_stratum": SUPPORT_STRATA,
    "layout": LAYOUTS,
}


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        _jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class StageACandidateSpec:
    drawer_aperture: str
    possession: str
    transit_locus: str
    support_stratum: str
    layout: str

    def __post_init__(self) -> None:
        for field_name, allowed in FACTOR_LEVELS.items():
            value = getattr(self, field_name)
            if value not in allowed:
                raise ValueError(
                    f"Invalid {field_name}={value!r}; expected one of {allowed}"
                )

    @property
    def candidate_id(self) -> str:
        return "__".join(
            (
                "stagea",
                f"drawer-{self.drawer_aperture}",
                f"possession-{self.possession.replace('_', '-')}",
                f"locus-{self.transit_locus.replace('_', '-')}",
                f"support-{self.support_stratum.replace('_', '-')}",
                f"layout-{self.layout.lower()}",
            )
        )

    @property
    def family_id(self) -> str:
        return "__".join(
            (
                f"drawer-{self.drawer_aperture}",
                f"possession-{self.possession.replace('_', '-')}",
                f"locus-{self.transit_locus.replace('_', '-')}",
            )
        )

    @property
    def support_pair_id(self) -> str:
        return f"{self.family_id}__layout-{self.layout.lower()}"

    @property
    def init_state_id(self) -> int:
        return LAYOUT_INIT_STATE_IDS[self.layout]

    def as_dict(self) -> dict[str, str | int]:
        return {
            "candidate_id": self.candidate_id,
            "drawer_aperture": self.drawer_aperture,
            "possession": self.possession,
            "transit_locus": self.transit_locus,
            "support_stratum": self.support_stratum,
            "layout": self.layout,
            "init_state_id": self.init_state_id,
            "family_id": self.family_id,
            "support_pair_id": self.support_pair_id,
        }


def iter_candidate_specs() -> tuple[StageACandidateSpec, ...]:
    specs = tuple(
        StageACandidateSpec(*values)
        for values in product(
            DRAWER_APERTURES,
            POSSESSIONS,
            TRANSIT_LOCI,
            SUPPORT_STRATA,
            LAYOUTS,
        )
    )
    validate_lattice_specs(specs)
    return specs


def validate_lattice_specs(specs: Iterable[StageACandidateSpec]) -> None:
    specs = tuple(specs)
    expected = set(product(*FACTOR_LEVELS.values()))
    observed = {
        (
            spec.drawer_aperture,
            spec.possession,
            spec.transit_locus,
            spec.support_stratum,
            spec.layout,
        )
        for spec in specs
    }
    ids = [spec.candidate_id for spec in specs]
    if len(specs) != 32 or observed != expected:
        raise ValueError("Stage A must contain the exact 32-cell full factorial")
    if len(set(ids)) != len(ids):
        raise ValueError("Stage A candidate IDs must be unique")


def candidate_spec(candidate_id: str) -> StageACandidateSpec:
    lookup = {spec.candidate_id: spec for spec in iter_candidate_specs()}
    try:
        return lookup[candidate_id]
    except KeyError as exc:
        raise ValueError(f"Unknown Stage A candidate ID: {candidate_id}") from exc


def build_selection_lock(
    *, contract_sha256: str, construction_revision: str
) -> dict[str, Any]:
    specs = iter_candidate_specs()
    payload = {
        "schema_version": 1,
        "stage": "phase3b_stage_a",
        "contract_sha256": contract_sha256,
        "construction_revision": construction_revision,
        "factor_levels": {key: list(values) for key, values in FACTOR_LEVELS.items()},
        "candidate_ids": [spec.candidate_id for spec in specs],
        "candidates": [spec.as_dict() for spec in specs],
        "policy_outcomes_used": False,
    }
    return {**payload, "selection_lock_sha256": canonical_sha256(payload)}


def validate_selection_lock(
    lock: dict[str, Any], *, contract_sha256: str, construction_revision: str
) -> None:
    expected = build_selection_lock(
        contract_sha256=contract_sha256,
        construction_revision=construction_revision,
    )
    if lock != expected:
        raise ValueError("Selection lock differs from the exact pre-policy Stage A lattice")


def rotate_point_about_axis(
    point: np.ndarray,
    axis_start: np.ndarray,
    axis_end: np.ndarray,
    angle_radians: float,
) -> np.ndarray:
    point = np.asarray(point, dtype=np.float64)
    axis_start = np.asarray(axis_start, dtype=np.float64)
    axis_end = np.asarray(axis_end, dtype=np.float64)
    if point.shape != (3,) or axis_start.shape != (3,) or axis_end.shape != (3,):
        raise ValueError("Point and axis endpoints must be three-vectors")
    direction = axis_end - axis_start
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        raise ValueError("Rotation axis endpoints are degenerate")
    unit = direction / norm
    relative = point - axis_start
    cosine = np.cos(angle_radians)
    sine = np.sin(angle_radians)
    rotated = (
        relative * cosine
        + np.cross(unit, relative) * sine
        + unit * np.dot(unit, relative) * (1.0 - cosine)
    )
    return axis_start + rotated


def recovery_balanced_goal_axis_point(
    scripted_waypoint: np.ndarray,
    first_goal: np.ndarray,
    second_goal: np.ndarray,
    recovery_center: np.ndarray,
    *,
    maximum_angle_degrees: float,
    target_recovery_mismatch: float,
) -> dict[str, Any]:
    scripted_waypoint = np.asarray(scripted_waypoint, dtype=np.float64)
    recovery_center = np.asarray(recovery_center, dtype=np.float64)
    if scripted_waypoint.shape != (3,) or recovery_center.shape != (3,):
        raise ValueError("Support and recovery points must be three-vectors")
    if not np.isfinite(maximum_angle_degrees) or not (
        0.0 < maximum_angle_degrees < 180.0
    ):
        raise ValueError("Maximum support rotation must be between 0 and 180 degrees")
    if not np.isfinite(target_recovery_mismatch) or not (
        0.0 < target_recovery_mismatch < 1.0
    ):
        raise ValueError("Target recovery mismatch must be between zero and one")

    near_recovery_distance = float(
        np.linalg.norm(scripted_waypoint - recovery_center)
    )

    def at_angle(angle_degrees: float) -> tuple[np.ndarray, float, float]:
        point = rotate_point_about_axis(
            scripted_waypoint,
            first_goal,
            second_goal,
            np.deg2rad(angle_degrees),
        )
        recovery_distance = float(np.linalg.norm(point - recovery_center))
        mismatch = symmetric_relative_difference(
            near_recovery_distance, recovery_distance
        )
        return point, recovery_distance, mismatch

    low_point, low_recovery_distance, mismatch = at_angle(
        maximum_angle_degrees
    )
    selected_angle = maximum_angle_degrees
    if mismatch > target_recovery_mismatch:
        lower = 0.0
        upper = maximum_angle_degrees
        for _ in range(64):
            midpoint = (lower + upper) / 2.0
            _, _, midpoint_mismatch = at_angle(midpoint)
            if midpoint_mismatch <= target_recovery_mismatch:
                lower = midpoint
            else:
                upper = midpoint
        selected_angle = lower
        low_point, low_recovery_distance, mismatch = at_angle(selected_angle)
    return {
        "point": low_point,
        "selected_angle_degrees": float(selected_angle),
        "near_recovery_distance_m": near_recovery_distance,
        "low_recovery_distance_m": low_recovery_distance,
        "planned_recovery_mismatch": mismatch,
        "pair_separation_m": float(
            np.linalg.norm(low_point - scripted_waypoint)
        ),
    }


def goal_distances(
    point: np.ndarray, *, drawer_goal: np.ndarray, cabinet_goal: np.ndarray
) -> dict[str, float]:
    point = np.asarray(point, dtype=np.float64)
    return {
        "drawer": float(np.linalg.norm(point - np.asarray(drawer_goal))),
        "cabinet": float(np.linalg.norm(point - np.asarray(cabinet_goal))),
    }


def _support_array(
    state: dict[str, Any], key: str, shape: tuple[int, ...]
) -> np.ndarray:
    value = np.asarray(state.get(key), dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"Support field {key} must have finite shape {shape}")
    return value


def _rotation_distance_radians(first: np.ndarray, second: np.ndarray) -> float:
    relative = first @ second.T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def joint_support_distance(
    query: dict[str, Any],
    reference: dict[str, Any],
    *,
    scales: dict[str, float],
    categorical_mismatch_penalty: float,
) -> tuple[float, dict[str, float]]:
    missing_scales = set(SUPPORT_SCALE_FIELDS) - set(scales)
    if missing_scales:
        raise ValueError(f"Missing support scales: {sorted(missing_scales)}")
    if any(
        not np.isfinite(float(scales[key])) or float(scales[key]) <= 0.0
        for key in SUPPORT_SCALE_FIELDS
    ):
        raise ValueError("Support scales must be finite and positive")
    if (
        not np.isfinite(categorical_mismatch_penalty)
        or categorical_mismatch_penalty <= 0.0
    ):
        raise ValueError("Categorical support penalty must be finite and positive")

    query_orientation = _support_array(query, "eef_orientation", (3, 3))
    reference_orientation = _support_array(reference, "eef_orientation", (3, 3))
    components = {
        "eef_position": float(
            np.linalg.norm(
                _support_array(query, "eef_position", (3,))
                - _support_array(reference, "eef_position", (3,))
            )
            / scales["eef_position_m"]
        ),
        "eef_orientation": _rotation_distance_radians(
            query_orientation, reference_orientation
        )
        / scales["eef_orientation_rad"],
        "robot_joints": float(
            np.sqrt(
                np.mean(
                    np.square(
                        _support_array(query, "robot_joint_positions", (7,))
                        - _support_array(
                            reference, "robot_joint_positions", (7,)
                        )
                    )
                )
            )
            / scales["robot_joint_rms_rad"]
        ),
        "bowl_position": float(
            np.linalg.norm(
                _support_array(query, "bowl_position", (3,))
                - _support_array(reference, "bowl_position", (3,))
            )
            / scales["bowl_position_m"]
        ),
        "drawer_joint": abs(
            float(query["drawer_joint"]) - float(reference["drawer_joint"])
        )
        / scales["drawer_joint_m"],
        "eef_motion": float(
            np.linalg.norm(
                _support_array(query, "eef_motion", (3,))
                - _support_array(reference, "eef_motion", (3,))
            )
            / scales["eef_motion_m"]
        ),
        "bowl_motion": float(
            np.linalg.norm(
                _support_array(query, "bowl_motion", (3,))
                - _support_array(reference, "bowl_motion", (3,))
            )
            / scales["bowl_motion_m"]
        ),
        "action_motion": float(
            np.sqrt(
                np.mean(
                    np.square(
                        _support_array(query, "action_motion", (6,))
                        - _support_array(reference, "action_motion", (6,))
                    )
                )
            )
            / scales["action_motion_rms"]
        ),
        "grasp_relative_position": float(
            np.linalg.norm(
                _support_array(query, "grasp_relative_position", (3,))
                - _support_array(reference, "grasp_relative_position", (3,))
            )
            / scales["grasp_relative_position_m"]
        ),
        "drawer_goal_distance": abs(
            float(query["drawer_goal_distance"])
            - float(reference["drawer_goal_distance"])
        )
        / scales["goal_distance_m"],
        "cabinet_goal_distance": abs(
            float(query["cabinet_goal_distance"])
            - float(reference["cabinet_goal_distance"])
        )
        / scales["goal_distance_m"],
    }
    for field in SUPPORT_CATEGORICAL_FIELDS:
        if field not in query or field not in reference:
            raise ValueError(f"Missing categorical support field {field}")
        components[f"{field}_mismatch"] = (
            0.0 if query[field] == reference[field] else categorical_mismatch_penalty
        )
    if not all(np.isfinite(value) for value in components.values()):
        raise ValueError("Support distance contains a non-finite component")
    distance = float(np.sqrt(np.mean(np.square(list(components.values())))))
    return distance, components


def measure_joint_support(
    query: dict[str, Any],
    references: Iterable[dict[str, Any]],
    *,
    scales: dict[str, float],
    categorical_mismatch_penalty: float,
) -> dict[str, Any]:
    references = tuple(references)
    if not references:
        raise ValueError("Joint support measurement requires reference states")

    def score(reference: dict[str, Any]) -> dict[str, Any]:
        distance, components = joint_support_distance(
            query,
            reference,
            scales=scales,
            categorical_mismatch_penalty=categorical_mismatch_penalty,
        )
        provenance = {
            key: reference[key]
            for key in (
                "reference_id",
                "goal",
                "demo_episode_index",
                "demo_task_index",
                "demo_action_sha256",
                "frame_index",
                *SUPPORT_CATEGORICAL_FIELDS,
            )
        }
        return {
            "distance": distance,
            "components": components,
            "reference": provenance,
        }

    scored = [score(reference) for reference in references]
    scored.sort(
        key=lambda item: (
            item["distance"],
            item["reference"]["reference_id"],
        )
    )
    exact = [
        item
        for item in scored
        if all(
            item["reference"][field] == query[field]
            for field in SUPPORT_CATEGORICAL_FIELDS
        )
    ]
    return {
        "pass": True,
        "metric": "joint_robot_object_event_motion_rms_v1",
        "reference_count": len(references),
        "event_matching_reference_count": len(exact),
        "query_categories": {
            field: query[field] for field in SUPPORT_CATEGORICAL_FIELDS
        },
        "scales": {key: float(scales[key]) for key in SUPPORT_SCALE_FIELDS},
        "categorical_mismatch_penalty": float(categorical_mismatch_penalty),
        "nearest": scored[0],
        "event_matched_nearest": exact[0] if exact else None,
    }


def snapshot_sha256(snapshot: LiberoStateSnapshot) -> str:
    payload = {
        "schema_version": 1,
        "mujoco_state": {
            "dtype": str(snapshot.mujoco_state.dtype),
            "shape": list(snapshot.mujoco_state.shape),
            "values": snapshot.mujoco_state,
        },
        "metadata": snapshot.metadata(),
    }
    return canonical_sha256(payload)


def symmetric_relative_difference(first: float, second: float) -> float:
    first = float(first)
    second = float(second)
    if not np.isfinite(first) or not np.isfinite(second):
        raise ValueError("Relative-difference inputs must be finite")
    scale = (abs(first) + abs(second)) / 2.0
    if scale == 0.0:
        return 0.0
    return abs(first - second) / scale


def validate_oracle_proposal_ledger(
    oracle: dict[str, Any], *, candidate_id: str, goal: str
) -> dict[str, Any]:
    proposal_bank = oracle.get("proposal_bank")
    attempts = oracle.get("proposal_attempts")
    selected_index = oracle.get("selected_proposal_index")
    if not isinstance(proposal_bank, list) or not proposal_bank:
        raise ValueError(f"Empty {goal} proposal bank for {candidate_id}")
    if not isinstance(attempts, list) or len(attempts) != len(proposal_bank):
        raise ValueError(
            f"Incomplete {goal} proposal-attempt ledger for {candidate_id}"
        )
    if oracle.get("proposal_bank_sha256") != canonical_sha256(proposal_bank):
        raise ValueError(f"Invalid {goal} proposal-bank hash for {candidate_id}")
    execution_mode = oracle.get("proposal_execution_mode")
    execution_contract = oracle.get("proposal_execution_contract")
    execution_contract_sha256 = oracle.get(
        "proposal_execution_contract_sha256"
    )
    if (
        execution_mode
        not in {
            "full_trajectory_replay",
            "action_intrinsic_pregrasp_phase_continuation_v2",
        }
        or execution_contract_sha256
        != canonical_sha256(execution_contract)
    ):
        raise ValueError(
            f"Invalid {goal} proposal-execution contract for {candidate_id}"
        )
    if any(
        attempt.get("proposal_execution_mode") != execution_mode
        for attempt in attempts
    ):
        raise ValueError(
            f"Mixed {goal} proposal-execution modes for {candidate_id}"
        )
    if execution_mode == "action_intrinsic_pregrasp_phase_continuation_v2":
        if not isinstance(execution_contract, list) or len(
            execution_contract
        ) != len(proposal_bank):
            raise ValueError(
                f"Invalid {goal} action-phase execution bank for {candidate_id}"
            )
        for proposal, attempt, phase_contract in zip(
            proposal_bank, attempts, execution_contract, strict=True
        ):
            phase_attempt = attempt.get("phase_proposal")
            if (
                not isinstance(phase_attempt, dict)
                or phase_attempt != phase_contract
                or any(
                phase_attempt.get(field) != proposal.get(source_field)
                for field, source_field in (
                    ("episode_index", "episode_index"),
                    ("task_index", "task_index"),
                    ("source_action_sha256", "action_sha256"),
                )
                )
            ):
                raise ValueError(
                    f"Mismatched {goal} action-phase contract for {candidate_id}"
                )
            if not isinstance(attempt.get("action_phase_bridge"), dict):
                raise ValueError(
                    f"Missing {goal} action-phase bridge for {candidate_id}"
                )
    elif execution_contract != {
        "execution_mode": "full_trajectory_replay",
        "transformation": "none",
    }:
        raise ValueError(
            f"Invalid full-trajectory contract for {candidate_id}/{goal}"
        )
    expected_indices = list(range(len(proposal_bank)))
    if [item.get("proposal_index") for item in proposal_bank] != expected_indices:
        raise ValueError(f"Unordered {goal} proposal bank for {candidate_id}")
    if [item.get("proposal_index") for item in attempts] != expected_indices:
        raise ValueError(f"Unordered {goal} proposal attempts for {candidate_id}")
    identity_fields = ("episode_index", "task_index", "action_sha256")
    for proposal, attempt in zip(proposal_bank, attempts, strict=True):
        if proposal.get("goal") != goal or any(
            attempt.get(field) != proposal.get(field) for field in identity_fields
        ):
            raise ValueError(
                f"Mismatched {goal} proposal-attempt identity for {candidate_id}"
            )
        if not isinstance(attempt.get("pass"), bool):
            raise ValueError(
                f"Non-boolean {goal} proposal result for {candidate_id}"
            )
    normalized_hashes = {
        attempt.get("normalized_state_sha256") for attempt in attempts
    }
    normalization_action_hashes = {
        attempt.get("normalization_action_sha256") for attempt in attempts
    }
    normalized_hash = (
        next(iter(normalized_hashes)) if len(normalized_hashes) == 1 else None
    )
    normalization_action_hash = (
        next(iter(normalization_action_hashes))
        if len(normalization_action_hashes) == 1
        else None
    )
    if (
        not isinstance(normalized_hash, str)
        or len(normalized_hash) != 64
        or oracle.get("shared_normalized_state_sha256")
        != normalized_hash
        or not isinstance(normalization_action_hash, str)
        or len(normalization_action_hash) != 64
        or oracle.get("shared_normalization_action_sha256")
        != normalization_action_hash
    ):
        raise ValueError(
            f"Inconsistent {goal} normalized-root provenance for {candidate_id}"
        )
    successful_indices = [
        index for index, attempt in enumerate(attempts) if attempt["pass"]
    ]
    if not successful_indices:
        raise ValueError(f"No successful {goal} proposal for {candidate_id}")
    if (
        not isinstance(selected_index, int)
        or selected_index not in successful_indices
        or oracle.get("successful_proposal_indices") != successful_indices
        or int(oracle.get("proposal_attempt_count", -1)) != len(attempts)
        or int(oracle.get("proposal_success_count", -1))
        != len(successful_indices)
        or not np.isclose(
            float(oracle.get("proposal_success_fraction", np.nan)),
            len(successful_indices) / len(attempts),
            rtol=0.0,
            atol=1e-15,
        )
        or oracle.get("proposal_selection_rule")
        != "minimum_executed_steps_then_path_effort_index"
    ):
        raise ValueError(f"Invalid {goal} proposal selection for {candidate_id}")
    expected_selected_index = min(
        successful_indices,
        key=lambda index: (
            int(attempts[index]["cost"]["executed_action_steps"]),
            float(attempts[index]["cost"]["eef_path_length_m"]),
            float(attempts[index]["cost"]["motion_control_effort"]),
            int(index),
        ),
    )
    selected_attempt = attempts[selected_index]
    selected_proposal = proposal_bank[selected_index]
    if (
        selected_index != expected_selected_index
        or oracle.get("demo_episode_index")
        != selected_proposal["episode_index"]
        or oracle.get("demo_task_index") != selected_proposal["task_index"]
        or oracle.get("demo_action_sha256")
        != selected_proposal["action_sha256"]
        or oracle.get("cost") != selected_attempt.get("cost")
    ):
        raise ValueError(
            f"Selected {goal} proposal payload mismatch for {candidate_id}"
        )
    actual_search_steps = int(oracle.get("shared_normalization_action_steps", -1))
    actual_search_steps += sum(
        int(attempt["cost"]["executed_demonstration_action_steps"])
        for attempt in attempts
    )
    counterfactual_steps = sum(
        int(attempt["cost"]["executed_action_steps"])
        for attempt in attempts
    )
    if (
        int(oracle.get("total_attempted_action_steps", -1))
        != actual_search_steps
        or int(oracle.get("counterfactual_full_attempt_action_steps", -1))
        != counterfactual_steps
    ):
        raise ValueError(f"Invalid {goal} proposal search cost for {candidate_id}")
    normalization_preparation = oracle.get("normalization_preparation")
    if execution_mode == "action_intrinsic_pregrasp_phase_continuation_v2":
        if (
            not isinstance(normalization_preparation, dict)
            or normalization_preparation.get("execution_mode")
            != "normalization_only"
            or normalization_preparation.get("source_proposal_replayed")
            is not False
            or int(
                normalization_preparation.get("executed_action_steps", -1)
            )
            != int(oracle.get("shared_normalization_action_steps", -2))
            or normalization_preparation.get("action_sha256")
            != oracle.get("shared_normalization_action_sha256")
        ):
            raise ValueError(
                f"Invalid {goal} normalization preparation for {candidate_id}"
            )
    else:
        if normalization_preparation is not None:
            raise ValueError(
                f"Unexpected normalization preparation for {candidate_id}/{goal}"
            )
    environment_steps = actual_search_steps
    if int(oracle.get("total_environment_action_steps", -1)) != environment_steps:
        raise ValueError(
            f"Invalid {goal} environment-action accounting for {candidate_id}"
        )
    return {
        "successful_indices": successful_indices,
        "selected_index": selected_index,
        "attempts": attempts,
        "execution_mode": execution_mode,
        "execution_contract_sha256": execution_contract_sha256,
    }


def validate_support_pair_geometry_records(
    near: dict[str, Any],
    low: dict[str, Any],
    *,
    max_realized_goal_distance_mismatch: float = 0.10,
    max_planned_recovery_distance_mismatch: float = 0.10,
 ) -> dict[str, Any]:
    """Validate the physical support-pair geometry without oracle comparability."""

    near_spec = candidate_spec(str(near.get("candidate_id")))
    low_spec = candidate_spec(str(low.get("candidate_id")))
    if (
        near_spec.support_stratum != "demonstration_near"
        or low_spec.support_stratum != "transverse_low_support"
        or near_spec.support_pair_id != low_spec.support_pair_id
    ):
        raise ValueError("Stage A support-pair identities or strata do not match")
    pair_id = near_spec.support_pair_id
    support_distance_difference = (
        float(low["support_measurement"]["nearest"]["distance"])
        - float(near["support_measurement"]["nearest"]["distance"])
    )
    if not np.isfinite(support_distance_difference):
        raise ValueError(f"Support pair {pair_id} has non-finite support distance")
    planned_recovery_mismatch = symmetric_relative_difference(
        near["root_geometry"]["planned_recovery_distance_m"],
        low["root_geometry"]["planned_recovery_distance_m"],
    )
    if planned_recovery_mismatch > max_planned_recovery_distance_mismatch:
        raise ValueError(
            f"Support pair {pair_id} exceeds planned recovery-distance limit"
        )
    realized_recovery_mismatch = symmetric_relative_difference(
        near["root_geometry"]["realized_recovery_distance_m"],
        low["root_geometry"]["realized_recovery_distance_m"],
    )
    planned_goal_mismatches = []
    realized_goal_mismatches = []
    for goal in GOALS:
        planned_mismatch = symmetric_relative_difference(
            near["root_geometry"]["planned_goal_distances_m"][goal],
            low["root_geometry"]["planned_goal_distances_m"][goal],
        )
        planned_goal_mismatches.append(planned_mismatch)
        if planned_mismatch > 1e-6:
            raise ValueError(
                f"Support pair {pair_id} does not preserve planned {goal} distance"
            )
        realized_mismatch = symmetric_relative_difference(
            near["root_geometry"]["realized_goal_distances_m"][goal],
            low["root_geometry"]["realized_goal_distances_m"][goal],
        )
        realized_goal_mismatches.append(realized_mismatch)
        if realized_mismatch > max_realized_goal_distance_mismatch:
            raise ValueError(
                f"Support pair {pair_id} exceeds realized {goal} distance limit"
            )
    return {
        "support_pair_id": pair_id,
        "support_distance_difference": support_distance_difference,
        "planned_recovery_mismatch": planned_recovery_mismatch,
        "realized_recovery_mismatch": realized_recovery_mismatch,
        "planned_goal_mismatches": planned_goal_mismatches,
        "realized_goal_mismatches": realized_goal_mismatches,
    }


def validate_support_pair_records(
    near: dict[str, Any],
    low: dict[str, Any],
    *,
    max_oracle_cost_mismatch: float = 0.10,
    max_realized_goal_distance_mismatch: float = 0.10,
    max_planned_recovery_distance_mismatch: float = 0.10,
    max_executed_step_mismatch: float = 0.10,
    max_active_step_mismatch: float = 0.20,
    max_eef_path_mismatch: float = 0.10,
    max_motion_control_effort_mismatch: float = 0.10,
) -> dict[str, Any]:
    geometry = validate_support_pair_geometry_records(
        near,
        low,
        max_realized_goal_distance_mismatch=max_realized_goal_distance_mismatch,
        max_planned_recovery_distance_mismatch=(
            max_planned_recovery_distance_mismatch
        ),
    )
    pair_id = geometry["support_pair_id"]
    near_spec = candidate_spec(str(near.get("candidate_id")))
    low_spec = candidate_spec(str(low.get("candidate_id")))
    budgeted_cost_mismatches: list[float] = []
    executed_step_mismatches: list[float] = []
    active_step_mismatches: list[float] = []
    eef_path_mismatches: list[float] = []
    motion_effort_mismatches: list[float] = []
    selected_proposal_matches: list[bool] = []
    shared_success_counts: list[int] = []
    success_set_jaccards: list[float] = []
    matched_cost_proposal_indices: list[int | None] = []
    for goal in GOALS:
        near_proposal_bank = near["oracles"][goal].get(
            "proposal_bank_sha256", ""
        )
        low_proposal_bank = low["oracles"][goal].get(
            "proposal_bank_sha256", ""
        )
        if (
            len(near_proposal_bank) != 64
            or near_proposal_bank != low_proposal_bank
        ):
            raise ValueError(
                f"Support pair {pair_id} uses inconsistent {goal} proposal banks"
            )
        if (
            near["oracles"][goal].get(
                "proposal_execution_contract_sha256"
            )
            != low["oracles"][goal].get(
                "proposal_execution_contract_sha256"
            )
        ):
            raise ValueError(
                f"Support pair {pair_id} uses inconsistent {goal} "
                "proposal-execution contracts"
            )
        near_ledger = validate_oracle_proposal_ledger(
            near["oracles"][goal], candidate_id=near_spec.candidate_id, goal=goal
        )
        low_ledger = validate_oracle_proposal_ledger(
            low["oracles"][goal], candidate_id=low_spec.candidate_id, goal=goal
        )
        near_success = set(near_ledger["successful_indices"])
        low_success = set(low_ledger["successful_indices"])
        shared_success = sorted(near_success & low_success)
        success_union = near_success | low_success
        shared_success_counts.append(len(shared_success))
        success_set_jaccards.append(len(shared_success) / len(success_union))
        common_index = (
            min(
                shared_success,
                key=lambda index: (
                    max(
                        int(
                            near_ledger["attempts"][index]["cost"][
                                "executed_action_steps"
                            ]
                        ),
                        int(
                            low_ledger["attempts"][index]["cost"][
                                "executed_action_steps"
                            ]
                        ),
                    ),
                    sum(
                        float(
                            ledger["attempts"][index]["cost"][
                                "eef_path_length_m"
                            ]
                        )
                        for ledger in (near_ledger, low_ledger)
                    ),
                    int(index),
                ),
            )
            if shared_success
            else None
        )
        matched_cost_proposal_indices.append(common_index)
        near_cost = near["oracles"][goal]["cost"]
        low_cost = low["oracles"][goal]["cost"]
        selected_proposal_matches.append(
            near_ledger["selected_index"] == low_ledger["selected_index"]
        )
        budgeted_mismatch = symmetric_relative_difference(
            near_cost["budgeted_action_steps"],
            low_cost["budgeted_action_steps"],
        )
        budgeted_cost_mismatches.append(budgeted_mismatch)
        if budgeted_mismatch > max_oracle_cost_mismatch:
            raise ValueError(
                f"Support pair {pair_id} exceeds the {goal} oracle-cost limit"
            )
        for field, limit, ledger in (
            ("executed_action_steps", max_executed_step_mismatch, executed_step_mismatches),
            ("active_servo_steps", max_active_step_mismatch, active_step_mismatches),
            ("eef_path_length_m", max_eef_path_mismatch, eef_path_mismatches),
            (
                "motion_control_effort",
                max_motion_control_effort_mismatch,
                motion_effort_mismatches,
            ),
        ):
            mismatch = symmetric_relative_difference(
                near_cost[field],
                low_cost[field],
            )
            ledger.append(mismatch)
            if mismatch > limit:
                raise ValueError(
                    f"Support pair {pair_id} exceeds the {goal} {field} limit"
                )
    return {
        **geometry,
        "budgeted_cost_mismatches": budgeted_cost_mismatches,
        "executed_step_mismatches": executed_step_mismatches,
        "active_step_mismatches": active_step_mismatches,
        "eef_path_mismatches": eef_path_mismatches,
        "motion_effort_mismatches": motion_effort_mismatches,
        "selected_proposal_matches": selected_proposal_matches,
        "shared_success_counts": shared_success_counts,
        "success_set_jaccards": success_set_jaccards,
        "matched_cost_proposal_indices": matched_cost_proposal_indices,
    }


def validate_stage_a_records(
    records: Iterable[dict[str, Any]],
    *,
    max_oracle_cost_mismatch: float = 0.10,
    max_realized_goal_distance_mismatch: float = 0.10,
    max_planned_recovery_distance_mismatch: float = 0.10,
    max_executed_step_mismatch: float = 0.10,
    max_active_step_mismatch: float = 0.20,
    max_eef_path_mismatch: float = 0.10,
    max_motion_control_effort_mismatch: float = 0.10,
) -> dict[str, Any]:
    records = tuple(records)
    specs = iter_candidate_specs()
    expected = {spec.candidate_id: spec for spec in specs}
    observed = {record.get("candidate_id"): record for record in records}
    if len(records) != 32 or set(observed) != set(expected):
        raise ValueError("Stage A records must cover each locked candidate exactly once")

    oracle_proposal_bank_hashes: dict[str, set[str]] = {
        goal: set() for goal in GOALS
    }
    oracle_nonfirst_selection_counts: dict[str, int] = {
        goal: 0 for goal in GOALS
    }
    oracle_success_fractions: dict[str, list[float]] = {
        goal: [] for goal in GOALS
    }
    oracle_execution_modes: dict[str, set[str]] = {
        goal: set() for goal in GOALS
    }
    oracle_execution_contract_hashes: dict[str, set[str]] = {
        goal: set() for goal in GOALS
    }
    for candidate_id, spec in expected.items():
        record = observed[candidate_id]
        if record.get("factors") != spec.as_dict():
            raise ValueError(f"Factor provenance mismatch for {candidate_id}")
        if record.get("policy_loaded") is not False:
            raise ValueError(f"Stage A policy boundary violated for {candidate_id}")
        if record.get("root_validation", {}).get("pass") is not True:
            raise ValueError(f"Root validation failed for {candidate_id}")
        if record.get("root_validation", {}).get("goals") != {
            "drawer": False,
            "cabinet": False,
        }:
            raise ValueError(f"A goal is already true at root {candidate_id}")
        if record.get("certificate", {}).get("pass") is not True:
            raise ValueError(f"Computational-state certificate failed for {candidate_id}")
        state_hash = record.get("state_sha256", "")
        if len(state_hash) != 64:
            raise ValueError(f"Missing immutable state hash for {candidate_id}")
        for goal in GOALS:
            oracle = record.get("oracles", {}).get(goal, {})
            if oracle.get("pass") is not True:
                raise ValueError(f"{goal} oracle failed for {candidate_id}")
            if oracle.get("goal_ever_achieved") is not True:
                raise ValueError(f"{goal} was never achieved for {candidate_id}")
            proposal_bank_hash = oracle.get("proposal_bank_sha256", "")
            if len(proposal_bank_hash) != 64:
                raise ValueError(
                    f"Missing {goal} proposal-bank hash for {candidate_id}"
                )
            oracle_proposal_bank_hashes[goal].add(proposal_bank_hash)
            ledger = validate_oracle_proposal_ledger(
                oracle, candidate_id=candidate_id, goal=goal
            )
            expected_mode = "action_intrinsic_pregrasp_phase_continuation_v2"
            if ledger["execution_mode"] != expected_mode:
                raise ValueError(
                    f"Unexpected {goal} execution mode for {candidate_id}"
                )
            oracle_execution_modes[goal].add(ledger["execution_mode"])
            oracle_execution_contract_hashes[goal].add(
                ledger["execution_contract_sha256"]
            )
            oracle_nonfirst_selection_counts[goal] += int(
                ledger["selected_index"] > 0
            )
            oracle_success_fractions[goal].append(
                float(oracle["proposal_success_fraction"])
            )
        normalized_hashes = {
            record["oracles"][goal]["shared_normalized_state_sha256"]
            for goal in GOALS
        }
        normalization_action_hashes = {
            record["oracles"][goal]["shared_normalization_action_sha256"]
            for goal in GOALS
        }
        if len(normalized_hashes) != 1 or len(normalization_action_hashes) != 1:
            raise ValueError(
                f"Goal oracles use different normalized roots for {candidate_id}"
            )

    for goal, hashes in oracle_proposal_bank_hashes.items():
        if len(hashes) != 1:
            raise ValueError(f"Stage A records use inconsistent {goal} proposal banks")

    pair_cost_mismatches: list[float] = []
    pair_planned_distance_mismatches: list[float] = []
    pair_realized_distance_mismatches: list[float] = []
    pair_planned_recovery_mismatches: list[float] = []
    pair_realized_recovery_mismatches: list[float] = []
    pair_executed_step_mismatches: list[float] = []
    pair_active_step_mismatches: list[float] = []
    pair_eef_path_mismatches: list[float] = []
    pair_motion_control_effort_mismatches: list[float] = []
    support_distance_differences: list[float] = []
    pair_selected_proposal_matches: dict[str, list[bool]] = {
        goal: [] for goal in GOALS
    }
    pair_shared_success_counts: dict[str, list[int]] = {
        goal: [] for goal in GOALS
    }
    pair_success_set_jaccards: dict[str, list[float]] = {
        goal: [] for goal in GOALS
    }
    support_bank_hashes: set[str] = set()
    for candidate_id in sorted(expected):
        support = observed[candidate_id].get("support_measurement", {})
        if support.get("pass") is not True:
            raise ValueError(f"Joint support measurement failed for {candidate_id}")
        bank_hash = support.get("reference_bank_sha256", "")
        if len(bank_hash) != 64:
            raise ValueError(f"Missing support-bank hash for {candidate_id}")
        support_bank_hashes.add(bank_hash)
        if int(support.get("reference_count", 0)) < 1:
            raise ValueError(f"Empty support reference bank for {candidate_id}")
        category_matches = support.get("factor_category_matches", {})
        if category_matches.get("drawer_aperture") is not True:
            raise ValueError(f"Physical drawer category mismatch for {candidate_id}")
        if category_matches.get("possession") is not True:
            raise ValueError(f"Physical possession category mismatch for {candidate_id}")
        nearest_distance = support.get("nearest", {}).get("distance")
        if nearest_distance is None or not np.isfinite(float(nearest_distance)):
            raise ValueError(f"Invalid joint support distance for {candidate_id}")
    if len(support_bank_hashes) != 1:
        raise ValueError("Stage A records use inconsistent support reference banks")

    for pair_id in sorted({spec.support_pair_id for spec in specs}):
        pair_specs = [spec for spec in specs if spec.support_pair_id == pair_id]
        if len(pair_specs) != 2:
            raise ValueError(f"Support pair {pair_id} is incomplete")
        near_spec = next(
            spec for spec in pair_specs if spec.support_stratum == "demonstration_near"
        )
        low_spec = next(
            spec
            for spec in pair_specs
            if spec.support_stratum == "transverse_low_support"
        )
        near = observed[near_spec.candidate_id]
        low = observed[low_spec.candidate_id]
        metrics = validate_support_pair_records(
            near,
            low,
            max_oracle_cost_mismatch=max_oracle_cost_mismatch,
            max_realized_goal_distance_mismatch=(
                max_realized_goal_distance_mismatch
            ),
            max_planned_recovery_distance_mismatch=(
                max_planned_recovery_distance_mismatch
            ),
            max_executed_step_mismatch=max_executed_step_mismatch,
            max_active_step_mismatch=max_active_step_mismatch,
            max_eef_path_mismatch=max_eef_path_mismatch,
            max_motion_control_effort_mismatch=(
                max_motion_control_effort_mismatch
            ),
        )
        support_distance_differences.append(metrics["support_distance_difference"])
        pair_planned_recovery_mismatches.append(
            metrics["planned_recovery_mismatch"]
        )
        pair_realized_recovery_mismatches.append(
            metrics["realized_recovery_mismatch"]
        )
        pair_planned_distance_mismatches.extend(metrics["planned_goal_mismatches"])
        pair_realized_distance_mismatches.extend(metrics["realized_goal_mismatches"])
        pair_cost_mismatches.extend(metrics["budgeted_cost_mismatches"])
        pair_executed_step_mismatches.extend(metrics["executed_step_mismatches"])
        pair_active_step_mismatches.extend(metrics["active_step_mismatches"])
        pair_eef_path_mismatches.extend(metrics["eef_path_mismatches"])
        pair_motion_control_effort_mismatches.extend(
            metrics["motion_effort_mismatches"]
        )
        for goal, matches in zip(
            GOALS, metrics["selected_proposal_matches"], strict=True
        ):
            pair_selected_proposal_matches[goal].append(matches)
        for goal, count, jaccard in zip(
            GOALS,
            metrics["shared_success_counts"],
            metrics["success_set_jaccards"],
            strict=True,
        ):
            pair_shared_success_counts[goal].append(int(count))
            pair_success_set_jaccards[goal].append(float(jaccard))

    return {
        "pass": True,
        "candidate_count": len(records),
        "support_pair_count": 16,
        "max_planned_goal_distance_mismatch": max(
            pair_planned_distance_mismatches, default=0.0
        ),
        "max_realized_goal_distance_mismatch": max(
            pair_realized_distance_mismatches, default=0.0
        ),
        "max_planned_recovery_distance_mismatch": max(
            pair_planned_recovery_mismatches, default=0.0
        ),
        "max_realized_recovery_distance_mismatch": max(
            pair_realized_recovery_mismatches, default=0.0
        ),
        "max_budgeted_oracle_cost_mismatch": max(pair_cost_mismatches, default=0.0),
        "max_executed_oracle_step_mismatch": max(
            pair_executed_step_mismatches, default=0.0
        ),
        "max_active_oracle_step_mismatch": max(
            pair_active_step_mismatches, default=0.0
        ),
        "max_oracle_eef_path_mismatch": max(pair_eef_path_mismatches, default=0.0),
        "max_oracle_motion_control_effort_mismatch": max(
            pair_motion_control_effort_mismatches, default=0.0
        ),
        "support_reference_bank_sha256": next(iter(support_bank_hashes)),
        "oracle_proposal_bank_sha256_by_goal": {
            goal: next(iter(hashes))
            for goal, hashes in oracle_proposal_bank_hashes.items()
        },
        "oracle_proposal_execution_mode_by_goal": {
            goal: next(iter(modes))
            for goal, modes in oracle_execution_modes.items()
        },
        "oracle_proposal_execution_contract_sha256_by_goal": {
            goal: sorted(hashes)
            for goal, hashes in oracle_execution_contract_hashes.items()
        },
        "proposal_nonfirst_selection_fraction_by_goal": {
            goal: float(
                oracle_nonfirst_selection_counts[goal] / len(records)
            )
            for goal in GOALS
        },
        "mean_proposal_success_fraction_by_goal": {
            goal: float(np.mean(values))
            for goal, values in oracle_success_fractions.items()
        },
        "minimum_proposal_success_fraction_by_goal": {
            goal: float(np.min(values))
            for goal, values in oracle_success_fractions.items()
        },
        "support_pair_same_selected_proposal_fraction_by_goal": {
            goal: float(np.mean(matches))
            for goal, matches in pair_selected_proposal_matches.items()
        },
        "support_pair_minimum_shared_success_count_by_goal": {
            goal: int(min(values))
            for goal, values in pair_shared_success_counts.items()
        },
        "support_pair_mean_success_set_jaccard_by_goal": {
            goal: float(np.mean(values))
            for goal, values in pair_success_set_jaccards.items()
        },
        "support_pair_positive_direction_fraction": float(
            np.mean(np.asarray(support_distance_differences) > 0.0)
        ),
        "median_joint_support_distance_difference": float(
            np.median(support_distance_differences)
        ),
        "policy_loaded_count": 0,
    }
