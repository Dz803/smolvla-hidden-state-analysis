"""Compatibility validation for frozen v35 bowl-registered oracle ledgers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from .phase3b_stage_a import (
    GOALS,
    canonical_sha256,
    candidate_spec,
    validate_oracle_proposal_ledger,
    validate_support_pair_records,
)


REGISTERED_EXECUTION_MODE = "action_intrinsic_pregrasp_bowl_registered_v1"
LEGACY_ACTION_PHASE_MODE = "action_intrinsic_pregrasp_phase_continuation_v2"
REGISTERED_ANCHOR_RULE = (
    "canonical_layout_a_pregrasp_anchor_translated_by_bowl_landmark"
)


def _vector(value: Any, *, field: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"Invalid registered vector: {field}")
    return result


def _orientation(value: Any, *, field: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3, 3) or not np.isfinite(result).all():
        raise ValueError(f"Invalid registered orientation: {field}")
    return result


def validate_registered_oracle_execution(
    oracle: dict[str, Any], *, candidate_id: str, goal: str
) -> dict[str, Any]:
    """Validate v35 registration metadata before adapting to the v34 ledger gate."""

    if oracle.get("proposal_execution_mode") != REGISTERED_EXECUTION_MODE:
        raise ValueError(
            f"Invalid registered {goal} execution mode for {candidate_id}"
        )
    contract = oracle.get("proposal_execution_contract")
    attempts = oracle.get("proposal_attempts")
    proposals = oracle.get("proposal_bank")
    if (
        not isinstance(contract, list)
        or not isinstance(attempts, list)
        or not isinstance(proposals, list)
        or len(contract) != len(attempts)
        or len(contract) != len(proposals)
        or not contract
    ):
        raise ValueError(
            f"Incomplete registered {goal} execution ledger for {candidate_id}"
        )
    if oracle.get("proposal_execution_contract_sha256") != canonical_sha256(
        contract
    ):
        raise ValueError(
            f"Invalid registered {goal} execution hash for {candidate_id}"
        )

    target_layout = candidate_spec(candidate_id).layout
    translation_norms = []
    root_binding_modes = []
    for index, (phase, attempt, proposal) in enumerate(
        zip(contract, attempts, proposals, strict=True)
    ):
        if (
            phase.get("proposal_index") != index
            or attempt.get("proposal_index") != index
            or attempt.get("phase_proposal") != phase
            or attempt.get("proposal_execution_mode")
            != REGISTERED_EXECUTION_MODE
            or phase.get("execution_mode") != REGISTERED_EXECUTION_MODE
            or phase.get("anchor_rule") != REGISTERED_ANCHOR_RULE
        ):
            raise ValueError(
                f"Mismatched registered {goal} phase {index} for {candidate_id}"
            )
        for phase_field, proposal_field in (
            ("episode_index", "episode_index"),
            ("task_index", "task_index"),
            ("source_action_sha256", "action_sha256"),
        ):
            if phase.get(phase_field) != proposal.get(proposal_field):
                raise ValueError(
                    f"Changed registered {goal} proposal identity for {candidate_id}"
                )
        registration = phase.get("landmark_registration")
        canonical = phase.get("canonical_reference")
        if (
            not isinstance(registration, dict)
            or not isinstance(canonical, dict)
            or registration.get("type") != "translation_only"
            or registration.get("orientation_transform") != "none"
            or registration.get("reference_layout") != "A"
            or registration.get("target_layout") != target_layout
            or canonical.get("layout") != "A"
            or canonical.get("execution_mode") != REGISTERED_EXECUTION_MODE
            or canonical.get("anchor_rule") != REGISTERED_ANCHOR_RULE
        ):
            raise ValueError(
                f"Invalid registered {goal} provenance for {candidate_id}"
            )
        reference_landmark = _vector(
            registration.get("reference_landmark_position"),
            field="reference_landmark_position",
        )
        target_landmark = _vector(
            registration.get("target_landmark_position"),
            field="target_landmark_position",
        )
        translation = _vector(
            registration.get("translation_m"), field="translation_m"
        )
        canonical_anchor = _vector(
            canonical.get("anchor_eef_position"),
            field="canonical_anchor_eef_position",
        )
        target_anchor = _vector(
            phase.get("anchor_eef_position"), field="anchor_eef_position"
        )
        canonical_orientation = _orientation(
            canonical.get("anchor_eef_orientation"),
            field="canonical_anchor_eef_orientation",
        )
        target_orientation = _orientation(
            phase.get("anchor_eef_orientation"),
            field="anchor_eef_orientation",
        )
        if (
            not np.allclose(
                translation,
                target_landmark - reference_landmark,
                rtol=0.0,
                atol=1e-12,
            )
            or not np.allclose(
                target_anchor,
                canonical_anchor + translation,
                rtol=0.0,
                atol=1e-12,
            )
            or not np.allclose(
                target_orientation,
                canonical_orientation,
                rtol=0.0,
                atol=1e-12,
            )
            or not np.isclose(
                float(registration.get("translation_norm_m", np.nan)),
                float(np.linalg.norm(translation)),
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError(
                f"Changed registered {goal} transform for {candidate_id}"
            )
        translation_norms.append(float(np.linalg.norm(translation)))
        binding = phase.get(
            "root_landmark_binding", "nominal_target_landmark_v1"
        )
        root_binding_modes.append(binding)
        action_phase_bridge = attempt.get("action_phase_bridge")
        root_registration = (
            action_phase_bridge.get("root_landmark_registration")
            if isinstance(action_phase_bridge, dict)
            else None
        )
        if binding == "normalized_bowl_translation_v1":
            if not isinstance(root_registration, dict):
                raise ValueError(
                    f"Missing normalized-root registration for "
                    f"{candidate_id}/{goal}/{index}"
                )
            expected_root_landmark = _vector(
                root_registration.get("expected_target_landmark_position"),
                field="expected_target_landmark_position",
            )
            observed_root_landmark = _vector(
                root_registration.get("observed_normalized_bowl_position"),
                field="observed_normalized_bowl_position",
            )
            root_residual = _vector(
                root_registration.get("normalized_bowl_residual_m"),
                field="normalized_bowl_residual_m",
            )
            nominal_root_anchor = _vector(
                root_registration.get("nominal_anchor_position"),
                field="nominal_anchor_position",
            )
            executed_root_anchor = _vector(
                root_registration.get("executed_anchor_position"),
                field="executed_anchor_position",
            )
            root_tolerance = float(
                root_registration.get("tolerance_m", np.nan)
            )
            declared_root_tolerance = float(
                phase.get("root_landmark_tolerance_m", np.nan)
            )
            if (
                root_registration.get("mode") != binding
                or not np.allclose(
                    expected_root_landmark,
                    target_landmark,
                    rtol=0.0,
                    atol=1e-12,
                )
                or not np.allclose(
                    root_residual,
                    observed_root_landmark - expected_root_landmark,
                    rtol=0.0,
                    atol=1e-12,
                )
                or not np.allclose(
                    nominal_root_anchor,
                    target_anchor,
                    rtol=0.0,
                    atol=1e-12,
                )
                or not np.allclose(
                    executed_root_anchor,
                    nominal_root_anchor + root_residual,
                    rtol=0.0,
                    atol=1e-12,
                )
                or not np.isclose(
                    float(
                        root_registration.get(
                            "normalized_bowl_residual_norm_m", np.nan
                        )
                    ),
                    float(np.linalg.norm(root_residual)),
                    rtol=0.0,
                    atol=1e-12,
                )
                or not np.isfinite(root_tolerance)
                or root_tolerance <= 0.0
                or not np.isclose(
                    root_tolerance,
                    declared_root_tolerance,
                    rtol=0.0,
                    atol=0.0,
                )
                or float(np.linalg.norm(root_residual)) > root_tolerance
            ):
                raise ValueError(
                    f"Changed normalized-root registration for "
                    f"{candidate_id}/{goal}/{index}"
                )
        elif binding != "nominal_target_landmark_v1":
            raise ValueError(
                f"Unknown registered root binding for {candidate_id}/{goal}"
            )
    return {
        "pass": True,
        "candidate_id": candidate_id,
        "goal": goal,
        "execution_mode": REGISTERED_EXECUTION_MODE,
        "proposal_count": len(contract),
        "target_layout": target_layout,
        "translation_norm_m": translation_norms[0],
        "all_proposals_share_translation": bool(
            np.allclose(
                translation_norms,
                translation_norms[0],
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "root_landmark_binding_modes": sorted(set(root_binding_modes)),
    }


def validate_support_pair_records_compatible(
    near: dict[str, Any], low: dict[str, Any], **limits: Any
) -> dict[str, Any]:
    """Run the existing strict pair gate after validating the added v35 mode."""

    adapted = [deepcopy(near), deepcopy(low)]
    registered = []
    for record in adapted:
        candidate_id = str(record.get("candidate_id"))
        for goal in GOALS:
            oracle = record["oracles"][goal]
            if oracle.get("proposal_execution_mode") != REGISTERED_EXECUTION_MODE:
                continue
            registered.append(
                validate_registered_oracle_execution(
                    oracle, candidate_id=candidate_id, goal=goal
                )
            )
            oracle["proposal_execution_mode"] = LEGACY_ACTION_PHASE_MODE
            for attempt in oracle["proposal_attempts"]:
                attempt["proposal_execution_mode"] = LEGACY_ACTION_PHASE_MODE
    result = validate_support_pair_records(adapted[0], adapted[1], **limits)
    result["registered_execution_validation"] = registered
    return result


def validate_oracle_proposal_ledger_compatible(
    oracle: dict[str, Any], *, candidate_id: str, goal: str
) -> dict[str, Any]:
    """Validate legacy or registered ledgers through one fail-closed entry point."""

    if oracle.get("proposal_execution_mode") != REGISTERED_EXECUTION_MODE:
        return validate_oracle_proposal_ledger(
            oracle, candidate_id=candidate_id, goal=goal
        )
    registered = validate_registered_oracle_execution(
        oracle, candidate_id=candidate_id, goal=goal
    )
    adapted = deepcopy(oracle)
    adapted["proposal_execution_mode"] = LEGACY_ACTION_PHASE_MODE
    for attempt in adapted["proposal_attempts"]:
        attempt["proposal_execution_mode"] = LEGACY_ACTION_PHASE_MODE
    result = validate_oracle_proposal_ledger(
        adapted, candidate_id=candidate_id, goal=goal
    )
    result["execution_mode"] = REGISTERED_EXECUTION_MODE
    result["registered_execution_validation"] = registered
    return result
