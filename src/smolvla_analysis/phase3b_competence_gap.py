"""Derive physical-recoverability versus proposal-compatibility evidence."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from smolvla_analysis.phase3b_stage_a import StageACandidateSpec


CELL_KEYS = ["candidate_id", "goal"]


def _require_columns(
    frame: pd.DataFrame,
    columns: set[str],
    *,
    name: str,
) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def build_goal_cell_table(
    candidates: pd.DataFrame,
    proposals: pd.DataFrame,
    feasibility: pd.DataFrame,
) -> pd.DataFrame:
    """Construct one row per state-goal with separate evidence classes."""

    _require_columns(
        candidates,
        {
            "candidate_id",
            "source_id",
            "drawer_aperture",
            "possession",
            "transit_locus",
            "support_stratum",
            "layout",
            "state_sha256",
        },
        name="candidate inventory",
    )
    _require_columns(
        proposals,
        {
            "candidate_id",
            "source_id",
            "goal",
            "execution_mode",
            "pass",
            "selected",
            "wrong_goal_ever_achieved",
            "unexpected_done_before_goal",
            "action_phase_bridge_pass",
        },
        name="proposal coverage",
    )
    _require_columns(
        feasibility,
        {
            "candidate_id",
            "source_id",
            "goal",
            "feasibility_kind",
            "pass",
            "policy_loaded",
            "proposal_execution_mode",
            "proposal_attempt_count",
            "proposal_success_count",
            "factorized_placement_action_count",
        },
        name="goal feasibility evidence",
    )
    if candidates["candidate_id"].duplicated().any():
        raise ValueError("candidate inventory contains duplicate candidate IDs")
    if feasibility.duplicated(CELL_KEYS).any():
        raise ValueError("goal feasibility evidence contains duplicate cells")

    proposal_groups = proposals.groupby(CELL_KEYS, sort=True, dropna=False)
    proposal_summary = proposal_groups.agg(
        proposal_attempt_count=("pass", "size"),
        proposal_success_count=("pass", "sum"),
        selected_proposal_count=("selected", "sum"),
        wrong_goal_count=("wrong_goal_ever_achieved", "sum"),
        unexpected_terminal_count=("unexpected_done_before_goal", "sum"),
        bridge_evidence_count=("action_phase_bridge_pass", "count"),
        bridge_pass_count=(
            "action_phase_bridge_pass",
            lambda values: int(values.dropna().astype(bool).sum()),
        ),
        proposal_execution_mode=("execution_mode", "first"),
        execution_mode_count=("execution_mode", "nunique"),
        proposal_source_id=("source_id", "first"),
        proposal_source_count=("source_id", "nunique"),
    ).reset_index()
    if (
        proposal_summary["execution_mode_count"].ne(1).any()
        or proposal_summary["proposal_source_count"].ne(1).any()
    ):
        raise ValueError("a proposal cell mixes execution modes or sources")
    proposal_summary = proposal_summary.drop(
        columns=["execution_mode_count", "proposal_source_count"]
    )

    evidence = feasibility.rename(
        columns={
            "pass": "physical_feasible",
            "policy_loaded": "feasibility_policy_loaded",
            "proposal_execution_mode": "evidence_proposal_execution_mode",
            "proposal_attempt_count": "evidence_proposal_attempt_count",
            "proposal_success_count": "evidence_proposal_success_count",
        }
    )
    merged = evidence.merge(
        proposal_summary,
        on=CELL_KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("proposal and feasibility cell inventories differ")
    merged = merged.drop(columns="_merge")
    for prefix in ("attempt", "success"):
        evidence_column = f"evidence_proposal_{prefix}_count"
        proposal_column = f"proposal_{prefix}_count"
        if not merged[evidence_column].eq(merged[proposal_column]).all():
            raise ValueError(
                f"feasibility and proposal {prefix} counts disagree"
            )
        merged = merged.drop(columns=evidence_column)
    if not merged["evidence_proposal_execution_mode"].eq(
        merged["proposal_execution_mode"]
    ).all():
        raise ValueError("feasibility and proposal execution modes disagree")
    merged = merged.drop(columns="evidence_proposal_execution_mode")

    factors = candidates[
        [
            "candidate_id",
            "source_id",
            "drawer_aperture",
            "possession",
            "transit_locus",
            "support_stratum",
            "layout",
            "state_sha256",
        ]
    ]
    merged = merged.merge(
        factors,
        on="candidate_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_candidate"),
    )
    if merged["source_id_candidate"].isna().any():
        raise ValueError("a goal cell lacks candidate metadata")
    if "source_id" in merged and not merged["source_id"].eq(
        merged["source_id_candidate"]
    ).all():
        raise ValueError("feasibility and candidate source IDs disagree")
    if not merged["proposal_source_id"].eq(
        merged["source_id_candidate"]
    ).all():
        raise ValueError("proposal and candidate source IDs disagree")
    merged = merged.drop(columns="proposal_source_id")
    merged["source_id"] = merged.pop("source_id_candidate")

    merged["physical_feasible"] = merged["physical_feasible"].astype(bool)
    merged["proposal_compatible"] = merged["proposal_success_count"].gt(0)
    merged["competence_compatibility_gap"] = (
        merged["physical_feasible"] & ~merged["proposal_compatible"]
    )
    if merged["feasibility_policy_loaded"].astype(bool).any():
        raise ValueError("policy-independent feasibility loaded a VLA policy")
    return merged.sort_values(CELL_KEYS).reset_index(drop=True)


def build_gap_pair_table(
    goal_cells: pd.DataFrame,
    support_pairs: pd.DataFrame,
    specs: Iterable[StageACandidateSpec],
) -> pd.DataFrame:
    """Attach each gap cell to its matched support-state comparison."""

    specs_by_id = {spec.candidate_id: spec for spec in specs}
    specs_by_pair: dict[str, list[StageACandidateSpec]] = {}
    for spec in specs_by_id.values():
        specs_by_pair.setdefault(spec.support_pair_id, []).append(spec)
    pairs_by_id = support_pairs.set_index("support_pair_id", drop=False)
    rows: list[dict[str, Any]] = []
    for gap in goal_cells[goal_cells["competence_compatibility_gap"]].to_dict(
        "records"
    ):
        candidate_id = str(gap["candidate_id"])
        goal = str(gap["goal"])
        spec = specs_by_id.get(candidate_id)
        if spec is None:
            raise ValueError(f"gap candidate is outside the lattice: {candidate_id}")
        pair_specs = specs_by_pair[spec.support_pair_id]
        if len(pair_specs) != 2:
            raise ValueError(f"support pair is not binary: {spec.support_pair_id}")
        comparison = next(
            candidate
            for candidate in pair_specs
            if candidate.candidate_id != candidate_id
        )
        comparison_cell = goal_cells[
            (goal_cells["candidate_id"] == comparison.candidate_id)
            & (goal_cells["goal"] == goal)
        ]
        if len(comparison_cell) != 1:
            raise ValueError("matched comparison goal cell is missing")
        if spec.support_pair_id not in pairs_by_id.index:
            raise ValueError("matched support-pair metric row is missing")
        pair = pairs_by_id.loc[spec.support_pair_id]
        if isinstance(pair, pd.DataFrame):
            raise ValueError("support-pair metrics contain duplicate pair IDs")
        comparison_record = comparison_cell.iloc[0]
        rows.append(
            {
                "support_pair_id": spec.support_pair_id,
                "goal": goal,
                "gap_candidate_id": candidate_id,
                "gap_support_stratum": spec.support_stratum,
                "gap_proposal_attempt_count": int(
                    gap["proposal_attempt_count"]
                ),
                "gap_proposal_success_count": int(
                    gap["proposal_success_count"]
                ),
                "gap_bridge_evidence_count": int(gap["bridge_evidence_count"]),
                "gap_bridge_pass_count": int(gap["bridge_pass_count"]),
                "gap_wrong_goal_count": int(gap["wrong_goal_count"]),
                "gap_unexpected_terminal_count": int(
                    gap["unexpected_terminal_count"]
                ),
                "gap_feasibility_kind": gap["feasibility_kind"],
                "gap_factorized_placement_action_count": int(
                    gap["factorized_placement_action_count"]
                ),
                "comparison_candidate_id": comparison.candidate_id,
                "comparison_support_stratum": comparison.support_stratum,
                "comparison_proposal_attempt_count": int(
                    comparison_record["proposal_attempt_count"]
                ),
                "comparison_proposal_success_count": int(
                    comparison_record["proposal_success_count"]
                ),
                "same_proposal_bank": bool(pair[f"{goal}_same_proposal_bank"]),
                "same_execution_contract": bool(
                    pair[f"{goal}_same_execution_contract"]
                ),
                "proposal_outcome_estimable": bool(
                    pair[f"{goal}_proposal_outcome_estimable"]
                ),
                "selected_cost_estimable": bool(
                    pair[f"{goal}_oracle_balance_estimable"]
                ),
                "shared_success_count": int(
                    pair[f"{goal}_shared_success_count"]
                ),
                "success_set_jaccard": float(
                    pair[f"{goal}_success_set_jaccard"]
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_competence_gap(
    goal_cells: pd.DataFrame,
    gap_pairs: pd.DataFrame,
) -> dict[str, Any]:
    """Return a compact scientific summary with explicit claim boundaries."""

    physical_count = int(goal_cells["physical_feasible"].sum())
    compatible_count = int(goal_cells["proposal_compatible"].sum())
    gap_count = int(goal_cells["competence_compatibility_gap"].sum())
    cell_count = int(len(goal_cells))
    if gap_count != len(gap_pairs):
        raise ValueError("gap cell and matched-pair counts disagree")
    return {
        "schema_version": 1,
        "status": "complete",
        "policy_loaded": False,
        "estimands": {
            "physical_recoverability": (
                "F_C(s,g): existence of a passing path in the declared "
                "policy-independent controller/certificate class C"
            ),
            "finite_bank_proposal_compatibility": (
                "P_K(s,g)=max_{k in K} Y(N(s),g,k) for the complete registered "
                "proposal bank K and its bound execution contract"
            ),
            "competence_compatibility_gap": "F_C(s,g)=1 and P_K(s,g)=0",
        },
        "goal_cell_count": cell_count,
        "physically_feasible_goal_cell_count": physical_count,
        "proposal_compatible_goal_cell_count": compatible_count,
        "competence_compatibility_gap_cell_count": gap_count,
        "competence_compatibility_gap_fraction": (
            gap_count / cell_count if cell_count else None
        ),
        "falsified_inference": (
            "Failure of every member of a finite registered replay bank implies "
            "physical infeasibility of that state-goal cell."
        ),
        "supported_claim": (
            "At least one certified Stage A state-goal cell is physically "
            "recoverable despite zero success in its complete registered replay bank."
        ),
        "not_supported": [
            "The SmolVLA policy can solve the gap cell.",
            "A hidden-state mechanism caused the proposal incompatibility.",
            "The adaptively developed factorized route estimates a population rate.",
            "The observed gap transfers to another state, task, or model.",
        ],
        "confirmatory_requirement": (
            "Freeze the certificate/controller family and tolerances before testing "
            "held-out roots; then evaluate model-specific V, Q, and self-knowledge "
            "targets separately on identical restored states."
        ),
    }
