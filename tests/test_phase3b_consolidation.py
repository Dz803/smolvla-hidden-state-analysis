from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.consolidate_phase3b_stage_a import _oracle_counterfactual_rows
from smolvla_analysis.phase3b_consolidation import (
    LEGACY_EXECUTION_CONTRACT,
    LEGACY_EXECUTION_MODE,
    factor_source_overlap,
    migrate_legacy_full_replay_record,
    validate_construction_contracts,
    validate_source_assignment,
    validate_source_root_timesteps,
)
from smolvla_analysis.phase3b_stage_a import canonical_sha256, iter_candidate_specs


def _construction_contract(horizon: int) -> dict:
    return {
        "candidate_ids": [spec.candidate_id for spec in iter_candidate_specs()],
        "certificate": {"actions": [[0.1] * 6]},
        "dataset_root": "local/training_data",
        "demonstrations": {"drawer": {"episode_index": 1}},
        "environment": {
            "task_id": 3,
            "episode_length": horizon,
            "control_mode": "relative",
        },
        "construction_revision": f"revision-{horizon}",
        "config_sha256": f"config-{horizon}",
        "goals": {"drawer": ["In"], "cabinet": ["On"]},
        "support_metric": {"scale": 1.0},
        "source_sha256": {"runner": f"runner-{horizon}"},
        "validation": {"limit": 0.1},
    }


def test_shared_root_context_excludes_later_horizon_and_exposes_gap() -> None:
    result = validate_construction_contracts(
        {
            "legacy": _construction_contract(1850),
            "phase": _construction_contract(2200),
        }
    )
    assert result["pass"] is True
    assert result["source_episode_horizons"] == {
        "legacy": 1850,
        "phase": 2200,
    }
    assert result["shared_root_context_contract_match"] is True
    assert result["construction_parameters_cryptographically_bound"] is False
    assert result["construction_revision_identical"] is False
    assert result["source_implementation_identical"] is False

    changed = _construction_contract(2200)
    changed["demonstrations"]["drawer"]["episode_index"] = 2
    with pytest.raises(ValueError, match="different shared root-context"):
        validate_construction_contracts(
            {"legacy": _construction_contract(1850), "changed": changed}
        )


def test_source_assignment_is_exact_and_disjoint() -> None:
    candidate_ids = [spec.candidate_id for spec in iter_candidate_specs()]
    assignments = {
        "first": candidate_ids[:16],
        "second": candidate_ids[16:],
    }
    selected = validate_source_assignment(assignments)
    assert len(selected) == 32
    assert selected[candidate_ids[0]] == "first"
    assert selected[candidate_ids[-1]] == "second"

    with pytest.raises(ValueError, match="assigned to both"):
        validate_source_assignment(
            {"first": candidate_ids, "duplicate": [candidate_ids[0]]}
        )
    with pytest.raises(ValueError, match="omits candidates"):
        validate_source_assignment({"partial": candidate_ids[:-1]})


def test_source_root_timesteps_are_bound_per_generation_batch() -> None:
    contracts = {
        "legacy": _construction_contract(1850),
        "registered": _construction_contract(2200),
    }
    result = validate_source_root_timesteps(
        {"legacy": {540}, "registered": {560}},
        contracts=contracts,
        expected={"legacy": 540, "registered": 560},
    )
    assert result["observed_root_final_timesteps"] == [540, 560]
    assert result["root_final_timestep_identical"] is False
    assert result["observed_roots_precede_oracle_horizons"] is True

    with pytest.raises(ValueError, match="declared root timestep"):
        validate_source_root_timesteps(
            {"legacy": {540}, "registered": {559}},
            contracts=contracts,
            expected={"legacy": 540, "registered": 560},
        )


def test_factor_source_overlap_exposes_aperture_batch_alias() -> None:
    specs = iter_candidate_specs()
    closed = [
        spec.candidate_id for spec in specs if spec.drawer_aperture == "closed"
    ]
    open_states = [
        spec.candidate_id for spec in specs if spec.drawer_aperture == "open"
    ]
    result = factor_source_overlap(
        {"closed_revision": closed, "open_revision": open_states}
    )
    aperture = result["drawer_aperture"]
    assert aperture["source_blocked_contrast_available"] is False
    assert aperture["common_sources_across_levels"] == []
    assert aperture["sources_by_level"] == {
        "closed": ["closed_revision"],
        "open": ["open_revision"],
    }
    assert result["support_stratum"]["source_blocked_contrast_available"] is True


def test_legacy_migration_is_additive_and_rejects_partial_provenance() -> None:
    original = {
        "candidate_id": "candidate",
        "oracles": {
            goal: {
                "total_attempted_action_steps": 123,
                "proposal_attempts": [{"proposal_index": 0}],
            }
            for goal in ("drawer", "cabinet")
        },
    }
    untouched = deepcopy(original)
    migrated, goals = migrate_legacy_full_replay_record(original)
    assert original == untouched
    assert goals == ["drawer", "cabinet"]
    for oracle in migrated["oracles"].values():
        assert oracle["proposal_execution_mode"] == LEGACY_EXECUTION_MODE
        assert oracle["proposal_execution_contract"] == LEGACY_EXECUTION_CONTRACT
        assert oracle["proposal_execution_contract_sha256"] == canonical_sha256(
            LEGACY_EXECUTION_CONTRACT
        )
        assert oracle["total_environment_action_steps"] == 123
        assert (
            oracle["proposal_attempts"][0]["proposal_execution_mode"]
            == LEGACY_EXECUTION_MODE
        )

    partial = deepcopy(original)
    partial["oracles"]["drawer"]["proposal_execution_mode"] = (
        LEGACY_EXECUTION_MODE
    )
    with pytest.raises(ValueError, match="Partially migrated"):
        migrate_legacy_full_replay_record(partial)


def test_oracle_counterfactual_preserves_both_execution_ledgers() -> None:
    entry = {
        "source_id": "v35",
        "record": {
            "candidate_id": "candidate",
            "prior_negative_oracle_evidence": {
                "cabinet": {
                    "normalized_state_sha256": "a" * 64,
                    "proposal_execution_mode": "world_anchor",
                    "proposal_count": 46,
                    "successful_proposal_count": 0,
                    "source_checkpoint_file_sha256": "b" * 64,
                }
            },
            "oracles": {
                "cabinet": {
                    "shared_normalized_state_sha256": "a" * 64,
                    "proposal_execution_mode": "bowl_registered",
                    "proposal_attempt_count": 46,
                    "proposal_success_count": 5,
                }
            },
        },
    }
    assert _oracle_counterfactual_rows(entry) == [
        {
            "candidate_id": "candidate",
            "source_id": "v35",
            "goal": "cabinet",
            "identical_normalized_state": True,
            "normalized_state_sha256": "a" * 64,
            "prior_execution_mode": "world_anchor",
            "prior_proposal_count": 46,
            "prior_success_count": 0,
            "current_execution_mode": "bowl_registered",
            "current_proposal_count": 46,
            "current_success_count": 5,
            "source_checkpoint_file_sha256": "b" * 64,
        }
    ]
