from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.run_phase3b_stage_a_completion import (
    _checkpoint,
    _load_construction_gate_binding,
    _smoke_episode_indices,
    _validate_completion_config,
)
from scripts.run_phase3b_stage_a import PROJECT, _load_config
from scripts.run_phase3b_stage_a_construction_gate import (
    _construction_contract_checks,
)
from scripts.status_phase3b_stage_a_completion import brief_summary
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


def test_completion_config_accepts_a_fresh_shard_without_imports() -> None:
    result = _validate_completion_config(
        {
            "completion": {
                "expected_candidate_count": 1,
                "candidate_ids": [OPEN_ID],
                "imports": [],
            }
        }
    )
    assert result["candidate_ids"] == (OPEN_ID,)
    assert result["imports"] == []


def test_v37_completion_binds_gate_and_prospective_smoke() -> None:
    config = _load_config(PROJECT / "configs/phase3b_stage_a_v37.yaml")
    completion = _validate_completion_config(config)
    gate = _load_construction_gate_binding(completion, config=config)
    assert gate is not None
    assert set(gate["expected_candidates"]) == set(
        completion["candidate_ids"]
    )
    assert completion["causal_smoke"]["proposal_episode_by_goal"] == {
        "drawer": 694,
        "cabinet": 474,
    }


def test_smoke_episode_indices_require_unique_identity() -> None:
    banks = {
        "drawer": _proposals(),
        "cabinet": _proposals(),
    }
    assert _smoke_episode_indices(
        banks, drawer_episode=10, cabinet_episode=11
    ) == {"drawer": 0, "cabinet": 1}
    with pytest.raises(ValueError, match="not unique"):
        _smoke_episode_indices(
            banks, drawer_episode=99, cabinet_episode=11
        )


def test_construction_gate_allows_the_environment_initial_timestep_offset() -> None:
    construction = {
        "grasp_acquisition": {
            "mode": "registered_cabinet_phase_until_stable_grasp_v1",
            "final_grasped": True,
            "final_goals": {"drawer": False, "cabinet": False},
        },
        "safe_lift": {
            "padded_to_budget": False,
            "executed_action_steps": 17,
            "budgeted_action_steps": 40,
        },
        "final_timestep": 560,
        "action_count": 550,
    }
    checks = _construction_contract_checks(
        construction, {"construction": {"root_final_timestep": 560}}
    )
    assert all(checks.values())

    construction["final_timestep"] = 559
    checks = _construction_contract_checks(
        construction, {"construction": {"root_final_timestep": 560}}
    )
    assert checks["normalized_final_timestep"] is False


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


def test_completion_checkpoint_accepts_sparse_import_without_rerun(tmp_path) -> None:
    (tmp_path / "checkpoints").mkdir()
    proposals = _proposals()
    phases = tuple(
        SimpleNamespace(metadata={"proposal_index": index})
        for index in range(2)
    )
    imported = {1: {"pass": True}}
    provenance = {1: {"kind": "held_root_smoke"}}
    completed, record, finish = _checkpoint(
        tmp_path,
        candidate_id=OPEN_ID,
        goal="drawer",
        root_state_sha256="a" * 64,
        contract_sha256="b" * 64,
        selection_lock_sha256="c" * 64,
        proposals=proposals,
        phase_proposals=phases,
        imported_results=imported,
        imported_provenance=provenance,
    )
    assert completed == imported
    record(0, {"pass": False})
    with pytest.raises(ValueError, match="Refusing to rerun"):
        record(1, {"pass": True})
    finish({"pass": True})

    resumed, _, _ = _checkpoint(
        tmp_path,
        candidate_id=OPEN_ID,
        goal="drawer",
        root_state_sha256="a" * 64,
        contract_sha256="b" * 64,
        selection_lock_sha256="c" * 64,
        proposals=proposals,
        phase_proposals=phases,
        imported_results=imported,
        imported_provenance=provenance,
    )
    assert resumed == {0: {"pass": False}, 1: {"pass": True}}


def test_completion_status_brief_selects_only_first_incomplete_candidate() -> None:
    complete = {
        "candidate_id": "complete",
        "candidate_record_complete": True,
    }
    active = {
        "candidate_id": "active",
        "candidate_record_complete": False,
    }
    report = {
        "run_id": "run",
        "status": "in_progress",
        "manifest_candidate_count": 1,
        "observed_candidate_count": 1,
        "expected_candidate_count": 2,
        "first_incomplete_candidate": "active",
        "error_count": 0,
        "errors": [],
        "candidates": [complete, active],
    }
    brief = brief_summary(report)
    assert brief["active_candidate"] == active
    assert "candidates" not in brief
