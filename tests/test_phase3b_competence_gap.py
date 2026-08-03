from __future__ import annotations

import pandas as pd
import pytest

from smolvla_analysis.phase3b_competence_gap import (
    build_gap_pair_table,
    build_goal_cell_table,
    summarize_competence_gap,
)
from smolvla_analysis.phase3b_stage_a import iter_candidate_specs


def _pair_specs():
    specs = iter_candidate_specs()
    pair_id = specs[0].support_pair_id
    return [spec for spec in specs if spec.support_pair_id == pair_id]


def _candidate_frame(specs) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": spec.candidate_id,
                "source_id": f"source-{index}",
                "drawer_aperture": spec.drawer_aperture,
                "possession": spec.possession,
                "transit_locus": spec.transit_locus,
                "support_stratum": spec.support_stratum,
                "layout": spec.layout,
                "state_sha256": str(index) * 64,
            }
            for index, spec in enumerate(specs, start=1)
        ]
    )


def test_competence_gap_is_distinct_from_proposal_failure() -> None:
    specs = _pair_specs()
    near = next(
        spec for spec in specs if spec.support_stratum == "demonstration_near"
    )
    low = next(
        spec
        for spec in specs
        if spec.support_stratum == "transverse_low_support"
    )
    candidates = _candidate_frame(specs)
    proposals = pd.DataFrame(
        [
            {
                "candidate_id": candidate_id,
                "source_id": (
                    "source-1"
                    if candidate_id == specs[0].candidate_id
                    else "source-2"
                ),
                "goal": "cabinet",
                "execution_mode": "registered",
                "pass": candidate_id == near.candidate_id and index == 0,
                "selected": candidate_id == near.candidate_id and index == 0,
                "wrong_goal_ever_achieved": False,
                "unexpected_done_before_goal": False,
                "action_phase_bridge_pass": True,
            }
            for candidate_id in (near.candidate_id, low.candidate_id)
            for index in range(2)
        ]
    )
    feasibility = pd.DataFrame(
        [
            {
                "candidate_id": spec.candidate_id,
                "source_id": f"source-{index}",
                "goal": "cabinet",
                "feasibility_kind": (
                    "successful_proposal_ledger_v1"
                    if spec.candidate_id == near.candidate_id
                    else "factorized_policy_free_path_v1"
                ),
                "pass": True,
                "policy_loaded": False,
                "proposal_execution_mode": "registered",
                "proposal_attempt_count": 2,
                "proposal_success_count": (
                    1 if spec.candidate_id == near.candidate_id else 0
                ),
                "factorized_placement_action_count": (
                    None if spec.candidate_id == near.candidate_id else 204
                ),
            }
            for index, spec in enumerate(specs, start=1)
        ]
    )
    cells = build_goal_cell_table(candidates, proposals, feasibility)
    gap = cells[cells["competence_compatibility_gap"]]
    assert gap[["candidate_id", "goal"]].to_dict("records") == [
        {"candidate_id": low.candidate_id, "goal": "cabinet"}
    ]
    assert cells["physical_feasible"].sum() == 2
    assert cells["proposal_compatible"].sum() == 1

    pair_metrics = pd.DataFrame(
        [
            {
                "support_pair_id": low.support_pair_id,
                "cabinet_same_proposal_bank": True,
                "cabinet_same_execution_contract": True,
                "cabinet_proposal_outcome_estimable": True,
                "cabinet_oracle_balance_estimable": False,
                "cabinet_shared_success_count": 0,
                "cabinet_success_set_jaccard": 0.0,
            }
        ]
    )
    gap_pairs = build_gap_pair_table(cells, pair_metrics, specs)
    row = gap_pairs.iloc[0]
    assert row["comparison_candidate_id"] == near.candidate_id
    assert row["gap_bridge_pass_count"] == 2
    assert bool(row["proposal_outcome_estimable"]) is True
    assert bool(row["selected_cost_estimable"]) is False

    summary = summarize_competence_gap(cells, gap_pairs)
    assert summary["physically_feasible_goal_cell_count"] == 2
    assert summary["proposal_compatible_goal_cell_count"] == 1
    assert summary["competence_compatibility_gap_cell_count"] == 1


def test_competence_gap_rejects_disagreeing_evidence_counts() -> None:
    specs = _pair_specs()
    candidates = _candidate_frame(specs)
    proposals = pd.DataFrame(
        [
            {
                "candidate_id": spec.candidate_id,
                "source_id": f"source-{index}",
                "goal": "cabinet",
                "execution_mode": "registered",
                "pass": False,
                "selected": False,
                "wrong_goal_ever_achieved": False,
                "unexpected_done_before_goal": False,
                "action_phase_bridge_pass": True,
            }
            for index, spec in enumerate(specs, start=1)
        ]
    )
    feasibility = pd.DataFrame(
        [
            {
                "candidate_id": spec.candidate_id,
                "source_id": f"source-{index}",
                "goal": "cabinet",
                "feasibility_kind": "factorized_policy_free_path_v1",
                "pass": True,
                "policy_loaded": False,
                "proposal_execution_mode": "registered",
                "proposal_attempt_count": 2,
                "proposal_success_count": 0,
                "factorized_placement_action_count": 204,
            }
            for index, spec in enumerate(specs, start=1)
        ]
    )
    with pytest.raises(ValueError, match="attempt counts disagree"):
        build_goal_cell_table(candidates, proposals, feasibility)
