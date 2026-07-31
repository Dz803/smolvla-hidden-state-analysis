from __future__ import annotations

from copy import deepcopy

import pytest

from smolvla_analysis.phase3b_alignment_report import (
    EXPECTED_CONDITIONS,
    build_alignment_summary,
)


def _fixture() -> tuple[dict, dict]:
    roots = {
        layout: {
            "root_state_sha256": f"root-{layout}",
            "normalized_state_sha256": f"normalized-{layout}",
            "normalized_bowl_position": [float(layout == "B"), 0.0, 0.0],
            "certificate": {"pass": True},
        }
        for layout in ("A", "B")
    }
    contract = {
        "diagnostic_revision": "phase3b-layout-alignment-v1",
        "conditions": list(EXPECTED_CONDITIONS),
        "proposal_episode_index": 474,
        "policy_loaded": False,
        "canonical_rollout_reused": False,
        "source_evidence": {
            layout.lower(): {
                "root_state_sha256": roots[layout]["root_state_sha256"],
                "normalized_state_sha256": roots[layout][
                    "normalized_state_sha256"
                ],
            }
            for layout in ("A", "B")
        },
    }
    base = {
        "bridge_pass": True,
        "bridge_drawer_aperture_preserved": True,
        "stable_grasp_achieved": True,
        "wrong_goal_ever_achieved": False,
        "unexpected_done_before_goal": False,
        "source_actions_executed": 8,
        "source_action_count": 10,
    }
    conditions = [
        {**base, "condition": EXPECTED_CONDITIONS[0], "goal_ever_achieved": True},
        {
            **base,
            "condition": EXPECTED_CONDITIONS[1],
            "goal_ever_achieved": False,
            "source_actions_executed": 10,
        },
        {**base, "condition": EXPECTED_CONDITIONS[2], "goal_ever_achieved": True},
        {
            "condition": EXPECTED_CONDITIONS[3],
            "status": "complete",
            "pass": True,
            "grasp_preserved_through_transport": True,
            "goal_ever_achieved": True,
            "wrong_goal_ever_achieved": False,
            "unexpected_done_before_goal": False,
        },
    ]
    result = {
        "status": "complete",
        "contract_sha256": "contract",
        "proposal": {"episode_index": 474},
        "roots": roots,
        "anchor_comparison": {
            "registration_delta_from_layout_b_world_anchor": [1.0, 0.0, 0.0]
        },
        "conditions": conditions,
    }
    return contract, result


def test_build_alignment_summary_preserves_the_causal_contrast() -> None:
    contract, result = _fixture()
    summary = build_alignment_summary(contract, result)
    assert summary["registration"]["anchor_displacement_m"] == 1.0
    assert summary["causal_contrast"]["world_anchor_goal"] is False
    assert summary["causal_contrast"]["bowl_registered_anchor_goal"] is True
    assert summary["scientific_boundary"]["status"] == (
        "exploratory_post_failure_diagnostic"
    )


@pytest.mark.parametrize(
    ("condition_index", "field", "value", "message"),
    [
        (0, "bridge_pass", False, "bridge or acquisition"),
        (1, "goal_ever_achieved", True, "not a failure"),
        (2, "goal_ever_achieved", False, "did not succeed"),
        (3, "pass", False, "did not pass"),
    ],
)
def test_build_alignment_summary_fails_closed(
    condition_index: int, field: str, value: bool, message: str
) -> None:
    contract, result = _fixture()
    changed = deepcopy(result)
    changed["conditions"][condition_index][field] = value
    with pytest.raises(ValueError, match=message):
        build_alignment_summary(contract, changed)
