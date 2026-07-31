from __future__ import annotations

from typing import Any

import numpy as np


EXPECTED_CANDIDATES = (
    "stagea__drawer-open__possession-on-table__locus-drawer-side__"
    "support-transverse-low-support__layout-a",
    "stagea__drawer-open__possession-on-table__locus-drawer-side__"
    "support-transverse-low-support__layout-b",
)


def build_registered_smoke_summary(
    contract: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    if contract.get("diagnostic_revision") != (
        "phase3b-v35-registered-held-roots-v1"
    ):
        raise ValueError("Unexpected registered-smoke revision")
    if tuple(contract.get("candidate_ids", ())) != EXPECTED_CANDIDATES:
        raise ValueError("Registered-smoke candidate lock changed")
    if (
        contract.get("proposal_episode_index") != 474
        or contract.get("proposal_index") != 31
    ):
        raise ValueError("Registered-smoke proposal lock changed")
    if result.get("status") != "complete" or result.get("all_pass") is not True:
        raise ValueError("Registered-smoke result did not pass")
    conditions = result.get("conditions")
    if not isinstance(conditions, list) or tuple(
        condition.get("candidate_id") for condition in conditions
    ) != EXPECTED_CANDIDATES:
        raise ValueError("Registered-smoke result candidates changed")
    rows = []
    for layout, condition in zip(("A", "B"), conditions, strict=True):
        attempt = condition.get("attempt", {})
        bridge = attempt.get("phases", {}).get("action_phase_bridge", {})
        phase = attempt.get("phase_proposal", {})
        registration = phase.get("landmark_registration", {})
        if (
            condition.get("pass") is not True
            or condition.get("root_validation", {}).get("pass") is not True
            or condition.get("certificate", {}).get("pass") is not True
            or attempt.get("pass") is not True
            or attempt.get("goal_ever_achieved") is not True
            or attempt.get("wrong_goal_ever_achieved") is not False
            or attempt.get("unexpected_done_before_goal") is not False
            or bridge.get("pass") is not True
            or bridge.get("drawer_aperture_preserved") is not True
            or phase.get("proposal_index") != 31
            or phase.get("layout") != layout
            or registration.get("reference_layout") != "A"
            or registration.get("target_layout") != layout
            or registration.get("type") != "translation_only"
        ):
            raise ValueError(f"Registered-smoke condition {layout} failed validation")
        translation = np.asarray(registration.get("translation_m"), dtype=np.float64)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError(f"Registered-smoke condition {layout} has bad translation")
        declared_norm = float(registration.get("translation_norm_m", np.nan))
        if not np.isclose(np.linalg.norm(translation), declared_norm, atol=1e-12):
            raise ValueError(f"Registered-smoke condition {layout} translation changed")
        rows.append(
            {
                "candidate_id": condition["candidate_id"],
                "layout": layout,
                "root_state_sha256": condition["root_state_sha256"],
                "normalized_state_sha256": attempt["normalized_state_sha256"],
                "certificate_pass": True,
                "support_distance": float(
                    condition["support_measurement"]["nearest"]["distance"]
                ),
                "registration_translation_norm_m": declared_norm,
                "registration_translation_xyz_m": translation.tolist(),
                "bridge_pass": True,
                "bridge_active_action_steps": int(bridge["active_action_steps"]),
                "bridge_bowl_drift_m": float(bridge["bowl_drift_m"]),
                "goal_ever_achieved": True,
                "first_goal_source_frame": int(attempt["first_goal_demo_frame"]),
                "executed_action_steps": int(
                    attempt["cost"]["executed_action_steps"]
                ),
                "executed_source_action_steps": int(
                    attempt["cost"]["executed_source_action_steps"]
                ),
                "eef_path_length_m": float(
                    attempt["cost"]["eef_path_length_m"]
                ),
                "motion_control_effort": float(
                    attempt["cost"]["motion_control_effort"]
                ),
            }
        )
    if len({row["root_state_sha256"] for row in rows}) != 2:
        raise ValueError("Registered-smoke physical roots are not distinct")
    return {
        "schema_version": 1,
        "diagnostic_revision": contract["diagnostic_revision"],
        "selection_status": contract["selection_status"],
        "contract_sha256": result["contract_sha256"],
        "proposal": {
            "proposal_index": contract["proposal_index"],
            "episode_index": contract["proposal_episode_index"],
            "action_sha256": contract["proposal_action_sha256"],
        },
        "support_reference_bank_sha256": result[
            "support_reference_bank_sha256"
        ],
        "phase_proposal_contract_sha256_by_layout": result[
            "phase_proposal_contract_sha256_by_layout"
        ],
        "condition_count": len(rows),
        "pass_count": len(rows),
        "conditions": rows,
        "prospective_generalization": {
            "frozen_on": "prior demonstration-near layout-A/layout-B pair",
            "held_factor": "transverse_low_support",
            "layout_a_pass": True,
            "layout_b_pass": True,
            "interpretation": (
                "the registered proposal transfers across the locked support "
                "displacement and both scene-layout replicates"
            ),
        },
        "scientific_boundary": {
            "establishes": (
                "held-root generalization of one policy-free registered proposal "
                "within one LIBERO task and physical family"
            ),
            "does_not_establish": (
                "universal proposal coverage, cross-task generalization, language "
                "grounding, or a hidden-state mechanism"
            ),
        },
        "policy_loaded": False,
        "canonical_rollout_reused": False,
    }
