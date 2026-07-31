from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from smolvla_analysis.phase3b_completion import (
    oracle_pair_comparability,
    proposal_inventory,
    summarize_exhaustive_negative_checkpoint,
    validate_completion_candidate_ids,
    validate_imported_checkpoint,
)
from smolvla_analysis.phase3b_stage_a import canonical_sha256


OPEN_ID = (
    "stagea__drawer-open__possession-on-table__locus-drawer-side__"
    "support-demonstration-near__layout-b"
)
CLOSED_ID = OPEN_ID.replace("drawer-open", "drawer-closed")


def _proposals():
    return tuple(
        SimpleNamespace(
            episode_index=index + 10,
            task_index=12,
            action_sha256=str(index) * 64,
        )
        for index in range(2)
    )


def test_completion_selection_is_unique_and_open_only() -> None:
    assert validate_completion_candidate_ids([OPEN_ID], expected_count=1) == (
        OPEN_ID,
    )
    with pytest.raises(ValueError, match="count or uniqueness"):
        validate_completion_candidate_ids([OPEN_ID, OPEN_ID], expected_count=2)
    with pytest.raises(ValueError, match="only missing open"):
        validate_completion_candidate_ids([CLOSED_ID], expected_count=1)


def test_imported_checkpoint_is_identity_bound_and_exhaustive() -> None:
    proposals = _proposals()
    phases = tuple(
        SimpleNamespace(metadata={"proposal_index": index})
        for index in range(2)
    )
    checkpoint = {
        "candidate_id": OPEN_ID,
        "goal": "drawer",
        "root_state_sha256": "a" * 64,
        "contract_sha256": "b" * 64,
        "selection_lock_sha256": "c" * 64,
        "proposal_inventory_sha256": canonical_sha256(
            proposal_inventory(proposals)
        ),
        "proposal_execution_contract_sha256": canonical_sha256(
            [phase.metadata for phase in phases]
        ),
        "status": "complete",
        "oracle_sha256": "d" * 64,
        "result_count": 2,
        "results": [
            {"proposal_index": 0, "result": {"pass": False}},
            {"proposal_index": 1, "result": {"pass": True}},
        ],
    }
    imported = validate_imported_checkpoint(
        checkpoint,
        candidate_id=OPEN_ID,
        goal="drawer",
        root_state_sha256="a" * 64,
        source_contract_sha256="b" * 64,
        source_selection_lock_sha256="c" * 64,
        proposals=proposals,
        phase_proposals=phases,
    )
    assert sorted(imported) == [0, 1]
    changed = deepcopy(checkpoint)
    changed["root_state_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="root_state_sha256"):
        validate_imported_checkpoint(
            changed,
            candidate_id=OPEN_ID,
            goal="drawer",
            root_state_sha256="a" * 64,
            source_contract_sha256="b" * 64,
            source_selection_lock_sha256="c" * 64,
            proposals=proposals,
            phase_proposals=phases,
        )


def test_negative_checkpoint_is_preserved_without_calling_it_complete() -> None:
    checkpoint = {
        "candidate_id": OPEN_ID,
        "goal": "cabinet",
        "root_state_sha256": "a" * 64,
        "status": "in_progress",
        "result_count": 2,
        "results": [
            {
                "proposal_index": index,
                "result": {
                    "pass": False,
                    "wrong_goal_ever_achieved": False,
                    "unexpected_done_before_goal": False,
                    "normalized_state_sha256": "n" * 64,
                    "normalization_action_sha256": "m" * 64,
                },
            }
            for index in range(2)
        ],
    }
    summary = summarize_exhaustive_negative_checkpoint(
        checkpoint,
        candidate_id=OPEN_ID,
        goal="cabinet",
        root_state_sha256="a" * 64,
        proposal_count=2,
    )
    assert summary["status"] == "exhaustive_negative_evidence"
    assert summary["checkpoint_status"] == "in_progress"


def test_oracle_pair_comparability_is_goal_specific() -> None:
    near = {
        "oracles": {
            "drawer": {
                "proposal_bank_sha256": "a",
                "proposal_execution_contract_sha256": "x",
                "proposal_execution_mode": "world",
            },
            "cabinet": {
                "proposal_bank_sha256": "b",
                "proposal_execution_contract_sha256": "y",
                "proposal_execution_mode": "registered",
            },
        }
    }
    low = deepcopy(near)
    low["oracles"]["drawer"]["proposal_execution_contract_sha256"] = "z"
    result = oracle_pair_comparability(near, low)
    assert result["by_goal"]["drawer"]["estimable"] is False
    assert result["by_goal"]["cabinet"]["estimable"] is True
    assert result["all_goals_estimable"] is False
