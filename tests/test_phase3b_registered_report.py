from __future__ import annotations

from copy import deepcopy

import pytest

from smolvla_analysis.phase3b_registered_report import (
    EXPECTED_CANDIDATES,
    build_registered_smoke_summary,
)


def _fixture():
    contract = {
        "diagnostic_revision": "phase3b-v35-registered-held-roots-v1",
        "selection_status": "prospective_after_freezing_on_prior_layout_pair",
        "candidate_ids": list(EXPECTED_CANDIDATES),
        "proposal_index": 31,
        "proposal_episode_index": 474,
        "proposal_action_sha256": "a" * 64,
    }
    conditions = []
    for layout, candidate_id in zip(("A", "B"), EXPECTED_CANDIDATES, strict=True):
        conditions.append(
            {
                "candidate_id": candidate_id,
                "root_state_sha256": layout.lower() * 64,
                "pass": True,
                "root_validation": {"pass": True},
                "certificate": {"pass": True},
                "support_measurement": {"nearest": {"distance": 0.5}},
                "attempt": {
                    "pass": True,
                    "goal_ever_achieved": True,
                    "wrong_goal_ever_achieved": False,
                    "unexpected_done_before_goal": False,
                    "first_goal_demo_frame": 80,
                    "normalized_state_sha256": (layout.lower() + "1") * 32,
                    "phase_proposal": {
                        "proposal_index": 31,
                        "layout": layout,
                        "landmark_registration": {
                            "type": "translation_only",
                            "reference_layout": "A",
                            "target_layout": layout,
                            "translation_m": [0.0, 0.0, 0.0],
                            "translation_norm_m": 0.0,
                        },
                    },
                    "phases": {
                        "action_phase_bridge": {
                            "pass": True,
                            "drawer_aperture_preserved": True,
                            "active_action_steps": 1,
                            "bowl_drift_m": 0.0,
                        }
                    },
                    "cost": {
                        "executed_action_steps": 10,
                        "executed_source_action_steps": 8,
                        "eef_path_length_m": 1.0,
                        "motion_control_effort": 2.0,
                    },
                },
            }
        )
    result = {
        "status": "complete",
        "all_pass": True,
        "contract_sha256": "contract",
        "support_reference_bank_sha256": "s" * 64,
        "phase_proposal_contract_sha256_by_layout": {"A": "x", "B": "y"},
        "conditions": conditions,
    }
    return contract, result


def test_registered_smoke_summary_preserves_prospective_status() -> None:
    contract, result = _fixture()
    summary = build_registered_smoke_summary(contract, result)
    assert summary["condition_count"] == 2
    assert summary["pass_count"] == 2
    assert summary["prospective_generalization"]["layout_b_pass"] is True


def test_registered_smoke_summary_fails_closed() -> None:
    contract, result = _fixture()
    changed = deepcopy(result)
    changed["conditions"][1]["attempt"]["phases"][
        "action_phase_bridge"
    ]["pass"] = False
    with pytest.raises(ValueError, match="condition B"):
        build_registered_smoke_summary(contract, changed)
