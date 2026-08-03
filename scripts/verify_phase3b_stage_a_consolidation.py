#!/usr/bin/env python
"""Verify a self-contained compact Stage A consolidation without raw runs."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import pandas as pd

from smolvla_analysis.phase3b_stage_a import GOALS, iter_candidate_specs


PROJECT = Path(__file__).resolve().parents[1]
FINAL_FACTORIZED_CANDIDATE = (
    "stagea__drawer-open__possession-grasped__locus-cabinet-side__"
    "support-transverse-low-support__layout-a"
)
SUCCESSFUL_LEDGER_KIND = "successful_proposal_ledger_v1"
FACTORIZED_PATH_KIND = "factorized_policy_free_path_v1"
EXPECTED_SOURCE_COUNTS = {
    "v31_open_hard_pair": 2,
    "v32_closed": 16,
    "v34_open_completion": 1,
    "v35_open_completion": 7,
    "v37_open_grasped_completion": 5,
    "v38_factorized_promotion": 1,
}
EXPECTED_SOURCE_TIMESTEPS = {
    "v31_open_hard_pair": 540,
    "v32_closed": 540,
    "v34_open_completion": 540,
    "v35_open_completion": 540,
    "v37_open_grasped_completion": 560,
    "v38_factorized_promotion": 560,
}
IMPLEMENTATION_PATHS = {
    "consolidator": PROJECT / "scripts/consolidate_phase3b_stage_a.py",
    "consolidation": PROJECT / "src/smolvla_analysis/phase3b_consolidation.py",
    "feasibility": PROJECT / "src/smolvla_analysis/phase3b_feasibility.py",
    "registered_validation": (
        PROJECT / "src/smolvla_analysis/phase3b_registered_validation.py"
    ),
    "completion": PROJECT / "src/smolvla_analysis/phase3b_completion.py",
    "stage_a_lattice": PROJECT / "src/smolvla_analysis/phase3b_stage_a.py",
    "snapshot_storage": PROJECT / "src/smolvla_analysis/phase2_storage.py",
    "snapshot_schema": PROJECT / "src/smolvla_analysis/libero_state.py",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify compact hashes and complete Stage A lattice coverage."
    )
    parser.add_argument("--report-dir", type=Path, required=True)
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_report(report_dir: Path) -> dict:
    report_dir = report_dir.resolve()
    manifest_path = report_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing compact manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 2 or manifest.get("status") != "complete":
        raise ValueError("Stage A consolidation manifest is not complete")
    expected_artifacts = manifest.get("artifact_sha256")
    observed_files = {
        path.name for path in report_dir.iterdir() if path.is_file()
    } - {"manifest.json"}
    if not isinstance(expected_artifacts, dict) or observed_files != set(
        expected_artifacts
    ):
        raise ValueError("Compact artifact inventory changed")
    for name, expected_sha in expected_artifacts.items():
        if _file_sha256(report_dir / name) != expected_sha:
            raise ValueError(f"Compact artifact hash mismatch: {name}")
    expected_implementation = {
        name: _file_sha256(path) for name, path in IMPLEMENTATION_PATHS.items()
    }
    if manifest.get("implementation_sha256") != expected_implementation:
        raise ValueError("Compact consolidation implementation provenance changed")

    summary = json.loads((report_dir / "summary.json").read_text())
    if (
        summary.get("schema_version") != 2
        or summary.get("status") != "complete"
        or summary.get("consolidation_revision")
        != manifest.get("consolidation_revision")
        or summary.get("candidate_count") != 32
        or summary.get("support_pair_count") != 16
        or summary.get("policy_loaded") is not False
        or summary.get("validation", {}).get("pass") is not True
    ):
        raise ValueError("Compact Stage A summary fails its primary gate")
    validation = summary["validation"]
    if (
        validation.get("physical_geometry_pair_count") != 16
        or validation.get("oracle_balance_all_goals_estimable_pair_count") != 13
        or validation.get("oracle_balance_estimable_pair_count_by_goal")
        != {"drawer": 14, "cabinet": 14}
        or validation.get("proposal_outcome_all_goals_estimable_pair_count") != 14
        or validation.get("proposal_outcome_estimable_pair_count_by_goal")
        != {"drawer": 14, "cabinet": 15}
        or validation.get("feasibility_kinds_by_goal")
        != {
            "drawer": [SUCCESSFUL_LEDGER_KIND],
            "cabinet": [FACTORIZED_PATH_KIND, SUCCESSFUL_LEDGER_KIND],
        }
    ):
        raise ValueError("Compact goal-specific oracle estimability changed")
    construction = summary["validation"].get("construction_compatibility", {})
    if (
        construction.get("shared_root_context_contract_match") is not True
        or construction.get("construction_parameters_cryptographically_bound")
        is not False
        or construction.get("construction_revision_identical") is not False
        or construction.get("source_implementation_identical") is not False
        or construction.get("observed_root_final_timesteps") != [540, 560]
        or construction.get("root_final_timestep_identical") is not False
        or construction.get("source_expected_root_final_timesteps")
        != EXPECTED_SOURCE_TIMESTEPS
        or construction.get("observed_roots_precede_oracle_horizons") is not True
    ):
        raise ValueError("Compact source-revision boundary is missing or incorrect")
    aperture_overlap = summary["validation"].get(
        "factor_source_overlap", {}
    ).get("drawer_aperture", {})
    if aperture_overlap.get("source_blocked_contrast_available") is not False:
        raise ValueError("Drawer-aperture/source-revision alias is not preserved")
    boundary = summary.get("scientific_boundary", {})
    if (
        boundary.get("cross_execution_mode_factor_effects_estimable") is not False
        or boundary.get("cross_source_generation_revision_effects_estimable")
        is not False
        or boundary.get("drawer_aperture_source_revision_alias") is not True
        or boundary.get(
            "historical_construction_parameters_cryptographically_bound"
        )
        is not False
        or boundary.get("require_support_pair_within_one_source") is not False
        or boundary.get("require_all_physical_pair_geometry_gates") is not True
        or boundary.get("require_goal_specific_oracle_comparability_labels")
        is not True
        or boundary.get("proposal_compatibility_is_not_physical_feasibility")
        is not True
        or boundary.get(
            "adaptive_factorized_certificate_precludes_population_inference"
        )
        is not True
        or boundary.get("factorized_controller_certificate_count") != 1
    ):
        raise ValueError("Compact scientific boundary is missing or incorrect")

    candidates = pd.read_csv(report_dir / "candidate_inventory.csv")
    expected_ids = {spec.candidate_id for spec in iter_candidate_specs()}
    if len(candidates) != 32 or set(candidates["candidate_id"]) != expected_ids:
        raise ValueError("Compact candidate inventory is incomplete")
    if not candidates["root_validation_pass"].all() or not candidates[
        "certificate_pass"
    ].all():
        raise ValueError("Compact candidate inventory contains a failed root")
    if candidates["state_sha256"].str.len().ne(64).any():
        raise ValueError("Compact candidate inventory has an invalid state hash")
    if candidates["state_sha256"].nunique() != 32:
        raise ValueError("Compact candidate inventory contains duplicate states")
    if set(candidates["root_final_timestep"]) != {540, 560}:
        raise ValueError("Compact candidate inventory has an invalid root timestep")
    observed_source_counts = candidates.groupby("source_id").size().to_dict()
    if observed_source_counts != EXPECTED_SOURCE_COUNTS:
        raise ValueError("Compact candidate/source counts changed")
    for source_id, timestep in EXPECTED_SOURCE_TIMESTEPS.items():
        observed = set(
            candidates.loc[
                candidates["source_id"] == source_id, "root_final_timestep"
            ]
        )
        if observed != {timestep}:
            raise ValueError(f"Compact source root timestep changed: {source_id}")

    source_inventory = json.loads(
        (report_dir / "source_inventory.json").read_text()
    )
    assigned = [
        candidate_id
        for source in source_inventory.values()
        for candidate_id in source["candidate_ids"]
    ]
    if len(assigned) != 32 or set(assigned) != expected_ids:
        raise ValueError("Compact source assignment is incomplete or duplicated")
    expected_source_by_candidate = {
        candidate_id: source_id
        for source_id, source in source_inventory.items()
        for candidate_id in source["candidate_ids"]
    }
    observed_source_by_candidate = candidates.set_index("candidate_id")[
        "source_id"
    ].to_dict()
    if observed_source_by_candidate != expected_source_by_candidate:
        raise ValueError("Compact candidate/source assignment changed")

    proposals = pd.read_csv(report_dir / "proposal_coverage.csv")
    expected_proposal_rows = 32 * (36 + 46)
    if len(proposals) != expected_proposal_rows:
        raise ValueError("Compact proposal coverage has an unexpected row count")
    if set(proposals["candidate_id"]) != expected_ids or set(
        proposals["goal"]
    ) != set(GOALS):
        raise ValueError("Compact proposal coverage has an invalid identity set")
    counts = proposals.groupby(["candidate_id", "goal"]).size()
    for candidate_id in expected_ids:
        if counts[(candidate_id, "drawer")] != 36:
            raise ValueError(f"Drawer proposal coverage changed for {candidate_id}")
        if counts[(candidate_id, "cabinet")] != 46:
            raise ValueError(f"Cabinet proposal coverage changed for {candidate_id}")
    proposal_indices = proposals.groupby(["candidate_id", "goal"])[
        "proposal_index"
    ].apply(list)
    for candidate_id in expected_ids:
        if proposal_indices[(candidate_id, "drawer")] != list(range(36)):
            raise ValueError(f"Drawer proposal identities changed for {candidate_id}")
        if proposal_indices[(candidate_id, "cabinet")] != list(range(46)):
            raise ValueError(f"Cabinet proposal identities changed for {candidate_id}")
    cell_index = ["candidate_id", "goal"]
    selected_counts = proposals.groupby(cell_index)["selected"].sum()
    success_counts = proposals.groupby(cell_index)["pass"].sum()
    zero_success_cells = {
        (str(candidate_id), str(goal))
        for candidate_id, goal in success_counts[success_counts == 0].index
    }
    expected_zero_cell = {(FINAL_FACTORIZED_CANDIDATE, "cabinet")}
    if zero_success_cells != expected_zero_cell:
        raise ValueError("Compact zero-success proposal cell changed")
    if int(selected_counts.loc[(FINAL_FACTORIZED_CANDIDATE, "cabinet")]) != 0:
        raise ValueError("Exhaustive negative cabinet ledger has a selection")
    positive_cells = success_counts[success_counts > 0].index
    if not (selected_counts.loc[positive_cells] == 1).all():
        raise ValueError("A positive compact proposal ledger lacks one selection")
    if not proposals.loc[proposals["selected"], "pass"].all():
        raise ValueError("A compact proposal ledger selects an infeasible proposal")
    if proposals["action_sha256"].str.len().ne(64).any():
        raise ValueError("Compact proposal coverage has an invalid action hash")
    proposal_source = proposals.set_index("candidate_id")["source_id"].to_dict()
    if any(
        proposal_source[candidate_id] != source_id
        for candidate_id, source_id in expected_source_by_candidate.items()
    ):
        raise ValueError("Compact proposal/source assignment changed")
    candidate_rows = candidates.set_index("candidate_id")
    for goal in GOALS:
        for candidate_id in expected_ids:
            goal_rows = proposals[
                (proposals["candidate_id"] == candidate_id)
                & (proposals["goal"] == goal)
            ]
            modes = goal_rows["execution_mode"].unique().tolist()
            if modes != [candidate_rows.loc[candidate_id, f"{goal}_execution_mode"]]:
                raise ValueError(
                    f"Compact {goal} execution mode changed for {candidate_id}"
                )

    final_cabinet = proposals[
        (proposals["candidate_id"] == FINAL_FACTORIZED_CANDIDATE)
        & (proposals["goal"] == "cabinet")
    ]
    if (
        len(final_cabinet) != 46
        or int(final_cabinet["pass"].sum()) != 0
        or int(final_cabinet["selected"].sum()) != 0
        or int(final_cabinet["wrong_goal_ever_achieved"].sum()) != 0
        or int(final_cabinet["unexpected_done_before_goal"].sum()) != 0
        or not final_cabinet["action_phase_bridge_pass"].all()
    ):
        raise ValueError("Final exhaustive negative cabinet ledger changed")

    feasibility = pd.read_csv(report_dir / "goal_feasibility_evidence.csv")
    if (
        len(feasibility) != 64
        or set(feasibility["candidate_id"]) != expected_ids
        or set(feasibility["goal"]) != set(GOALS)
        or not feasibility["pass"].all()
        or feasibility["policy_loaded"].any()
        or feasibility.groupby(cell_index).size().ne(1).any()
    ):
        raise ValueError("Compact physical-feasibility inventory changed")
    factorized = feasibility[
        feasibility["feasibility_kind"] == FACTORIZED_PATH_KIND
    ]
    successful = feasibility[
        feasibility["feasibility_kind"] == SUCCESSFUL_LEDGER_KIND
    ]
    if (
        len(factorized) != 1
        or len(successful) != 63
        or factorized.iloc[0]["candidate_id"] != FINAL_FACTORIZED_CANDIDATE
        or factorized.iloc[0]["goal"] != "cabinet"
        or int(factorized.iloc[0]["proposal_success_count"]) != 0
        or int(factorized.iloc[0]["factorized_placement_action_count"]) != 204
        or factorized.iloc[0]["factorized_result_file_sha256"]
        != "f8283663b953e97ac62afef0c6685c6c4e1a1b373b83a45845dc6ef29614c26c"
        or successful["proposal_success_count"].le(0).any()
    ):
        raise ValueError("Compact feasibility evidence classes changed")
    expected_feasibility_summary = {
        ("cabinet", FACTORIZED_PATH_KIND, 1),
        ("cabinet", SUCCESSFUL_LEDGER_KIND, 31),
        ("drawer", SUCCESSFUL_LEDGER_KIND, 32),
    }
    observed_feasibility_summary = {
        (row["goal"], row["feasibility_kind"], int(row["candidate_count"]))
        for row in summary.get("physical_feasibility_summary_by_kind", [])
    }
    if observed_feasibility_summary != expected_feasibility_summary:
        raise ValueError("Compact feasibility summary changed")

    counterfactuals = pd.read_csv(
        report_dir / "oracle_execution_counterfactuals.csv"
    )
    expected_counterfactual_candidate = (
        "stagea__drawer-open__possession-on-table__locus-drawer-side__"
        "support-demonstration-near__layout-b"
    )
    if (
        len(counterfactuals) != 1
        or summary.get("oracle_execution_counterfactual_count") != 1
        or counterfactuals.iloc[0]["candidate_id"]
        != expected_counterfactual_candidate
        or counterfactuals.iloc[0]["goal"] != "cabinet"
        or bool(counterfactuals.iloc[0]["identical_normalized_state"]) is not True
        or int(counterfactuals.iloc[0]["prior_proposal_count"]) != 46
        or int(counterfactuals.iloc[0]["prior_success_count"]) != 0
        or int(counterfactuals.iloc[0]["current_proposal_count"]) != 46
        or int(counterfactuals.iloc[0]["current_success_count"]) != 5
    ):
        raise ValueError("Compact oracle-execution counterfactual changed")

    pairs = pd.read_csv(report_dir / "support_pairs.csv")
    expected_pair_ids = {spec.support_pair_id for spec in iter_candidate_specs()}
    if (
        len(pairs) != 16
        or pairs["support_pair_id"].nunique() != 16
        or set(pairs["support_pair_id"]) != expected_pair_ids
    ):
        raise ValueError("Compact support-pair inventory is incomplete")
    for _, row in pairs.iterrows():
        pair_id = row["support_pair_id"]
        pair_specs = [
            spec
            for spec in iter_candidate_specs()
            if spec.support_pair_id == pair_id
        ]
        near_spec = next(
            spec
            for spec in pair_specs
            if spec.support_stratum == "demonstration_near"
        )
        low_spec = next(
            spec
            for spec in pair_specs
            if spec.support_stratum == "transverse_low_support"
        )
        expected_near = expected_source_by_candidate[near_spec.candidate_id]
        expected_low = expected_source_by_candidate[low_spec.candidate_id]
        if (
            row["near_source_id"] != expected_near
            or row["low_source_id"] != expected_low
            or bool(row["same_source"]) != (expected_near == expected_low)
        ):
            raise ValueError(f"Compact support-pair provenance changed: {pair_id}")
        for goal in GOALS:
            proposal_estimable = bool(
                row[f"{goal}_proposal_outcome_estimable"]
            )
            selected_cost_estimable = bool(
                row[f"{goal}_oracle_balance_estimable"]
            )
            same_bank = bool(row[f"{goal}_same_proposal_bank"])
            same_execution = bool(row[f"{goal}_same_execution_contract"])
            if proposal_estimable != (same_bank and same_execution):
                raise ValueError(
                    f"Compact {goal} proposal estimability changed: {pair_id}"
                )
            has_both_selections = (
                int(selected_counts.loc[(near_spec.candidate_id, goal)]) == 1
                and int(selected_counts.loc[(low_spec.candidate_id, goal)]) == 1
            )
            if selected_cost_estimable != (
                proposal_estimable and has_both_selections
            ):
                raise ValueError(
                    f"Compact {goal} selected-cost estimability changed: {pair_id}"
                )
    if int(pairs["same_source"].sum()) != 14:
        raise ValueError("Compact cross-source support-pair count changed")
    if int(pairs["proposal_outcome_all_goals_estimable"].sum()) != 14:
        raise ValueError("Compact all-goal proposal estimability count changed")
    if int(pairs["oracle_balance_all_goals_estimable"].sum()) != 13:
        raise ValueError("Compact all-goal selected-cost estimability changed")

    final_spec = next(
        spec
        for spec in iter_candidate_specs()
        if spec.candidate_id == FINAL_FACTORIZED_CANDIDATE
    )
    final_pair = pairs[pairs["support_pair_id"] == final_spec.support_pair_id]
    if len(final_pair) != 1:
        raise ValueError("Final matched support pair is missing")
    final_pair_row = final_pair.iloc[0]
    unavailable_cost_fields = [
        "cabinet_matched_cost_proposal_index",
        "cabinet_budgeted_cost_mismatch",
        "cabinet_executed_step_mismatch",
        "cabinet_active_step_mismatch",
        "cabinet_eef_path_mismatch",
        "cabinet_motion_effort_mismatch",
    ]
    if (
        bool(final_pair_row["cabinet_proposal_outcome_estimable"]) is not True
        or bool(final_pair_row["cabinet_oracle_balance_estimable"]) is not False
        or int(final_pair_row["cabinet_shared_success_count"]) != 0
        or float(final_pair_row["cabinet_success_set_jaccard"]) != 0.0
        or not all(pd.isna(final_pair_row[field]) for field in unavailable_cost_fields)
    ):
        raise ValueError("Final cabinet competence-compatibility pair changed")
    return {
        "pass": True,
        "consolidation_revision": manifest["consolidation_revision"],
        "artifact_count": len(expected_artifacts),
        "candidate_count": len(candidates),
        "support_pair_count": len(pairs),
        "proposal_row_count": len(proposals),
        "physical_goal_cell_count": len(feasibility),
        "factorized_certificate_count": len(factorized),
        "source_count": len(source_inventory),
        "execution_modes_by_goal": summary["validation"][
            "execution_modes_by_goal"
        ],
        "cross_mode_factor_effects_estimable": summary["validation"][
            "cross_mode_factor_effects_estimable"
        ],
    }


def main() -> None:
    args = _parse_args()
    result = verify_report(args.report_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
