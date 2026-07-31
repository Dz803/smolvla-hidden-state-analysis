from __future__ import annotations

from copy import deepcopy

import pytest

import smolvla_analysis.phase3b_registered_validation as registered_validation
from smolvla_analysis.phase3b_registered_validation import (
    REGISTERED_ANCHOR_RULE,
    REGISTERED_EXECUTION_MODE,
    validate_registered_oracle_execution,
    validate_support_pair_records_compatible,
)
from smolvla_analysis.phase3b_stage_a import canonical_sha256


NEAR_A = (
    "stagea__drawer-open__possession-on-table__locus-cabinet-side__"
    "support-demonstration-near__layout-a"
)
LOW_A = NEAR_A.replace("demonstration-near", "transverse-low-support")


def _oracle(*, goal: str = "drawer") -> dict:
    proposal = {
        "proposal_index": 0,
        "goal": goal,
        "episode_index": 10,
        "task_index": 12 if goal == "drawer" else 18,
        "action_sha256": "a" * 64,
    }
    canonical = {
        "layout": "A",
        "execution_mode": REGISTERED_EXECUTION_MODE,
        "anchor_rule": REGISTERED_ANCHOR_RULE,
        "anchor_eef_position": [0.1, 0.2, 0.3],
        "anchor_eef_orientation": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    }
    phase = {
        "proposal_index": 0,
        "episode_index": 10,
        "task_index": proposal["task_index"],
        "source_action_sha256": "a" * 64,
        "execution_mode": REGISTERED_EXECUTION_MODE,
        "anchor_rule": REGISTERED_ANCHOR_RULE,
        "anchor_eef_position": [0.1, 0.2, 0.3],
        "anchor_eef_orientation": canonical["anchor_eef_orientation"],
        "canonical_reference": canonical,
        "landmark_registration": {
            "type": "translation_only",
            "orientation_transform": "none",
            "reference_layout": "A",
            "target_layout": "A",
            "reference_landmark_position": [0.0, 0.0, 0.0],
            "target_landmark_position": [0.0, 0.0, 0.0],
            "translation_m": [0.0, 0.0, 0.0],
            "translation_norm_m": 0.0,
        },
    }
    attempt = {
        "proposal_index": 0,
        "proposal_execution_mode": REGISTERED_EXECUTION_MODE,
        "phase_proposal": phase,
    }
    return {
        "proposal_execution_mode": REGISTERED_EXECUTION_MODE,
        "proposal_execution_contract": [phase],
        "proposal_execution_contract_sha256": canonical_sha256([phase]),
        "proposal_bank": [proposal],
        "proposal_attempts": [attempt],
    }


def test_registered_execution_validation_accepts_translation_only_contract() -> None:
    result = validate_registered_oracle_execution(
        _oracle(), candidate_id=NEAR_A, goal="drawer"
    )
    assert result["pass"] is True
    assert result["proposal_count"] == 1
    assert result["all_proposals_share_translation"] is True


def test_registered_execution_validation_rejects_changed_transform() -> None:
    oracle = _oracle()
    oracle["proposal_execution_contract"][0]["anchor_eef_position"][0] += 0.01
    oracle["proposal_execution_contract_sha256"] = canonical_sha256(
        oracle["proposal_execution_contract"]
    )
    oracle["proposal_attempts"][0]["phase_proposal"] = deepcopy(
        oracle["proposal_execution_contract"][0]
    )
    with pytest.raises(ValueError, match="Changed registered drawer transform"):
        validate_registered_oracle_execution(
            oracle, candidate_id=NEAR_A, goal="drawer"
        )


def test_compatibility_gate_adapts_only_outer_mode(monkeypatch) -> None:
    records = []
    for candidate_id in (NEAR_A, LOW_A):
        records.append(
            {
                "candidate_id": candidate_id,
                "oracles": {
                    goal: _oracle(goal=goal) for goal in ("drawer", "cabinet")
                },
            }
        )

    def downstream(near, low, **limits):
        assert limits == {"limit": 1}
        for record in (near, low):
            for oracle in record["oracles"].values():
                assert (
                    oracle["proposal_execution_mode"]
                    == registered_validation.LEGACY_ACTION_PHASE_MODE
                )
                assert (
                    oracle["proposal_execution_contract"][0]["execution_mode"]
                    == REGISTERED_EXECUTION_MODE
                )
        return {"pass": True}

    monkeypatch.setattr(
        registered_validation, "validate_support_pair_records", downstream
    )
    result = validate_support_pair_records_compatible(
        records[0], records[1], limit=1
    )
    assert result["pass"] is True
    assert len(result["registered_execution_validation"]) == 4
    assert (
        records[0]["oracles"]["drawer"]["proposal_execution_mode"]
        == REGISTERED_EXECUTION_MODE
    )
