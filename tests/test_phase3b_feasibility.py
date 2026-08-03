from __future__ import annotations

from copy import deepcopy

import pytest

from smolvla_analysis.phase3b_feasibility import (
    FACTORIZED_EXECUTION_MODE,
    FACTORIZED_FEASIBILITY_KIND,
    validate_factorized_feasibility_evidence,
)


def _factorized_evidence() -> dict:
    return {
        "schema_version": 1,
        "kind": FACTORIZED_FEASIBILITY_KIND,
        "execution_mode": FACTORIZED_EXECUTION_MODE,
        "goal": "cabinet",
        "pass": True,
        "policy_loaded": False,
        "root_state_sha256": "a" * 64,
        "normalized_state_sha256": "b" * 64,
        "normalization_action_sha256": "c" * 64,
        "normalization_action_steps": 970,
        "source_proposal": {
            "proposal_index": 31,
            "episode_index": 474,
            "action_sha256": "d" * 64,
            "source_actions_executed": 45,
            "full_tail_executed": False,
        },
        "acquisition": {
            "acquisition_state_sha256": "e" * 64,
            "bridge_action_count": 340,
            "bridge_action_sha256": "f" * 64,
            "first_stable_grasp_source_frame": 45,
            "stable_grasp_streak": 3,
            "trace_sha256": "1" * 64,
        },
        "placement": {
            "action_count": 204,
            "action_sha256": "2" * 64,
            "transport_phases": [
                {
                    "phase": "clearance_lift",
                    "budget_ceiling_action_steps": 150,
                    "executed_action_steps": 95,
                    "active_action_steps": 95,
                    "final_position_error_m": 0.018,
                    "stopped_on_goal": False,
                    "stopped_on_terminal": False,
                    "bowl_grasped_after_phase": True,
                },
                {
                    "phase": "clearance_transit",
                    "budget_ceiling_action_steps": 120,
                    "executed_action_steps": 95,
                    "active_action_steps": 95,
                    "final_position_error_m": 0.018,
                    "stopped_on_goal": False,
                    "stopped_on_terminal": False,
                    "bowl_grasped_after_phase": True,
                },
                {
                    "phase": "target_descent",
                    "budget_ceiling_action_steps": 70,
                    "executed_action_steps": 14,
                    "active_action_steps": 14,
                    "final_position_error_m": 0.010,
                    "stopped_on_goal": True,
                    "stopped_on_terminal": False,
                    "bowl_grasped_after_phase": False,
                },
            ],
            "final_goals": {"drawer": False, "cabinet": True},
            "bowl_released": True,
            "wrong_goal_ever_achieved": False,
            "unexpected_done_before_goal": False,
        },
        "execution_scope": {
            "completed_full_suffix_baselines_reexecuted": 0,
            "source_proposal_full_tail_executed": False,
            "policy_forwards": 0,
        },
        "source_artifact": {
            "diagnostic_revision": "v4",
            "condition": "factorized",
            "contract_sha256": "3" * 64,
            "contract_file_sha256": "4" * 64,
            "manifest_file_sha256": "5" * 64,
            "result_file_sha256": "6" * 64,
            "acquisition_file_sha256": "7" * 64,
            "source_checkpoint_file_sha256": "8" * 64,
        },
        "scientific_boundary": "One policy-free path, not a VLA mechanism.",
    }


def test_factorized_feasibility_keeps_horizon_and_root_bindings() -> None:
    evidence = _factorized_evidence()
    result = validate_factorized_feasibility_evidence(
        evidence,
        candidate_id="candidate",
        root_state_sha256="a" * 64,
        normalized_state_sha256="b" * 64,
        normalization_action_sha256="c" * 64,
    )
    assert result["pass"] is True
    assert result["placement_action_count"] == 204

    padded = deepcopy(evidence)
    padded["placement"]["transport_phases"][0][
        "executed_action_steps"
    ] = 151
    with pytest.raises(ValueError, match="action ceiling"):
        validate_factorized_feasibility_evidence(padded)

    wrong_root = deepcopy(evidence)
    with pytest.raises(ValueError, match="root_state_sha256 mismatch"):
        validate_factorized_feasibility_evidence(
            wrong_root, candidate_id="candidate", root_state_sha256="9" * 64
        )
