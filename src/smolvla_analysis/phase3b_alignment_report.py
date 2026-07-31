from __future__ import annotations

from typing import Any

import numpy as np


EXPECTED_CONDITIONS = (
    "layout_a_world_anchor_full_suffix",
    "layout_b_world_anchor_full_suffix",
    "layout_b_bowl_registered_full_suffix",
    "layout_b_registered_acquisition_then_goal_registered_placement",
)


def _clean_condition(condition: dict[str, Any]) -> dict[str, Any]:
    return {
        key: condition.get(key)
        for key in (
            "condition",
            "status",
            "bridge_pass",
            "bridge_drawer_aperture_preserved",
            "stable_grasp_achieved",
            "first_stable_grasp_source_frame",
            "goal_ever_achieved",
            "wrong_goal_ever_achieved",
            "unexpected_done_before_goal",
            "source_actions_executed",
            "source_action_count",
            "minimum_eef_bowl_distance_m",
            "grasp_preserved_through_transport",
            "goals_before_release",
            "final_goals",
            "pass",
            "action_count",
        )
        if key in condition
    }


def build_alignment_summary(
    contract: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    """Validate and compact the bounded layout-alignment diagnostic."""

    if contract.get("diagnostic_revision") != "phase3b-layout-alignment-v1":
        raise ValueError("Unexpected layout-alignment diagnostic revision")
    if result.get("status") != "complete":
        raise ValueError("Layout-alignment result is not complete")
    if contract.get("conditions") != list(EXPECTED_CONDITIONS):
        raise ValueError("Layout-alignment contract condition order changed")
    conditions = result.get("conditions")
    if not isinstance(conditions, list) or [
        item.get("condition") for item in conditions
    ] != list(EXPECTED_CONDITIONS):
        raise ValueError("Layout-alignment result condition order changed")
    if result.get("proposal", {}).get("episode_index") != contract.get(
        "proposal_episode_index"
    ):
        raise ValueError("Layout-alignment proposal identity changed")

    roots = result.get("roots", {})
    if set(roots) != {"A", "B"}:
        raise ValueError("Layout-alignment result must contain roots A and B")
    for layout, root in roots.items():
        expected = contract["source_evidence"][layout.lower()]
        if root.get("root_state_sha256") != expected["root_state_sha256"]:
            raise ValueError(f"Layout-{layout} root hash changed")
        if root.get("normalized_state_sha256") != expected[
            "normalized_state_sha256"
        ]:
            raise ValueError(f"Layout-{layout} normalized root hash changed")
        if root.get("certificate", {}).get("pass") is not True:
            raise ValueError(f"Layout-{layout} state certificate failed")

    layout_a, layout_b_world, layout_b_registered, factorized = conditions
    full_suffix_pass_fields = (
        "bridge_pass",
        "bridge_drawer_aperture_preserved",
        "stable_grasp_achieved",
    )
    if any(
        condition.get(field) is not True
        for condition in conditions[:3]
        for field in full_suffix_pass_fields
    ):
        raise ValueError("A full-suffix bridge or acquisition certificate failed")
    if layout_a.get("goal_ever_achieved") is not True:
        raise ValueError("Layout-A reference did not achieve the cabinet goal")
    if layout_b_world.get("goal_ever_achieved") is not False:
        raise ValueError("Layout-B world-anchor contrast is not a failure")
    if layout_b_registered.get("goal_ever_achieved") is not True:
        raise ValueError("Layout-B registered-anchor contrast did not succeed")
    if int(layout_b_world.get("source_actions_executed", -1)) != int(
        layout_b_world.get("source_action_count", -2)
    ):
        raise ValueError("Layout-B world-anchor suffix was not exhausted")
    if any(
        condition.get("wrong_goal_ever_achieved") is not False
        or condition.get("unexpected_done_before_goal") is not False
        for condition in conditions
    ):
        raise ValueError("A diagnostic condition hit a wrong goal or early terminal")
    if (
        factorized.get("status") != "complete"
        or factorized.get("pass") is not True
        or factorized.get("grasp_preserved_through_transport") is not True
        or factorized.get("goal_ever_achieved") is not True
    ):
        raise ValueError("Factorized registered placement did not pass")

    anchor = result.get("anchor_comparison", {})
    registration_delta = np.asarray(
        anchor.get("registration_delta_from_layout_b_world_anchor"),
        dtype=np.float64,
    )
    if registration_delta.shape != (3,) or not np.isfinite(
        registration_delta
    ).all():
        raise ValueError("Invalid registered-anchor displacement")
    bowl_a = np.asarray(roots["A"]["normalized_bowl_position"], dtype=np.float64)
    bowl_b = np.asarray(roots["B"]["normalized_bowl_position"], dtype=np.float64)

    return {
        "schema_version": 1,
        "diagnostic_revision": contract["diagnostic_revision"],
        "contract_sha256": result["contract_sha256"],
        "proposal": result["proposal"],
        "policy_loaded": contract["policy_loaded"],
        "canonical_rollout_reused": contract["canonical_rollout_reused"],
        "root_hashes": {
            layout: {
                "root_state_sha256": root["root_state_sha256"],
                "normalized_state_sha256": root["normalized_state_sha256"],
                "certificate_pass": root["certificate"]["pass"],
            }
            for layout, root in roots.items()
        },
        "registration": {
            "anchor_displacement_m": float(np.linalg.norm(registration_delta)),
            "anchor_displacement_xyz_m": registration_delta.tolist(),
            "normalized_bowl_displacement_m": float(np.linalg.norm(bowl_b - bowl_a)),
            "normalized_bowl_displacement_xyz_m": (bowl_b - bowl_a).tolist(),
        },
        "conditions": [_clean_condition(item) for item in conditions],
        "causal_contrast": {
            "root": "identical certified layout-B normalized snapshot",
            "source_suffix": "identical episode-474 action suffix",
            "intervention": "translate only the pre-grasp EEF anchor to preserve the layout-A bowl-relative offset",
            "world_anchor_goal": False,
            "bowl_registered_anchor_goal": True,
            "world_anchor_stable_grasp": True,
            "bowl_registered_anchor_stable_grasp": True,
        },
        "competence_chain": {
            "physical_root_certified": True,
            "world_anchor_bridge_reachable": True,
            "world_anchor_stable_grasp": True,
            "world_anchor_goal_placement": False,
            "registered_anchor_stable_grasp": True,
            "registered_suffix_goal_placement": True,
            "registered_factorized_transport_preserved_grasp": True,
            "registered_factorized_release_goal": True,
        },
        "scientific_boundary": {
            "status": "exploratory_post_failure_diagnostic",
            "scope": "one proposal, two layouts, one physical factor cell",
            "establishes": (
                "a causal controller-input alignment effect for this exact root and "
                "proposal, plus stage-wise physical competence"
            ),
            "does_not_establish": (
                "a general oracle, a hidden-state mechanism, language grounding, "
                "or cross-layout population-level generalization"
            ),
        },
    }
