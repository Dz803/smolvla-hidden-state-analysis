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
    if manifest.get("status") != "complete":
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

    summary = json.loads((report_dir / "summary.json").read_text())
    if (
        summary.get("status") != "complete"
        or summary.get("candidate_count") != 32
        or summary.get("support_pair_count") != 16
        or summary.get("policy_loaded") is not False
        or summary.get("validation", {}).get("pass") is not True
    ):
        raise ValueError("Compact Stage A summary fails its primary gate")
    validation = summary["validation"]
    if (
        validation.get("physical_geometry_pair_count") != 16
        or validation.get("oracle_balance_all_goals_estimable_pair_count") != 14
        or validation.get("oracle_balance_estimable_pair_count_by_goal")
        != {"drawer": 14, "cabinet": 15}
    ):
        raise ValueError("Compact goal-specific oracle estimability changed")
    construction = summary["validation"].get("construction_compatibility", {})
    if (
        construction.get("shared_root_context_contract_match") is not True
        or construction.get("construction_parameters_cryptographically_bound")
        is not False
        or construction.get("construction_revision_identical") is not False
        or construction.get("source_implementation_identical") is not False
        or construction.get("observed_root_final_timestep") != 540
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
    if set(candidates["root_final_timestep"]) != {540}:
        raise ValueError("Compact candidate inventory has an invalid root timestep")

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
    selected_counts = proposals.groupby(["candidate_id", "goal"])[
        "selected"
    ].sum()
    if not (selected_counts == 1).all():
        raise ValueError("A compact proposal ledger has no unique selection")
    if not proposals.groupby(["candidate_id", "goal"])["pass"].any().all():
        raise ValueError("A compact candidate-goal cell has no feasible proposal")
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
            estimable = bool(row[f"{goal}_oracle_balance_estimable"])
            same_bank = bool(row[f"{goal}_same_proposal_bank"])
            same_execution = bool(row[f"{goal}_same_execution_contract"])
            if estimable != (same_bank and same_execution):
                raise ValueError(
                    f"Compact {goal} pair estimability changed: {pair_id}"
                )
    if int(pairs["same_source"].sum()) != 15:
        raise ValueError("Compact cross-source support-pair count changed")
    if int(pairs["oracle_balance_all_goals_estimable"].sum()) != 14:
        raise ValueError("Compact all-goal oracle estimability count changed")
    return {
        "pass": True,
        "consolidation_revision": manifest["consolidation_revision"],
        "artifact_count": len(expected_artifacts),
        "candidate_count": len(candidates),
        "support_pair_count": len(pairs),
        "proposal_row_count": len(proposals),
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
