#!/usr/bin/env python
"""Independently verify the compact Stage A competence-gap analysis."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import pandas as pd

from verify_phase3b_stage_a_consolidation import verify_report


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = (
    PROJECT / "reports/phase3b_stage_a/competence_compatibility_gap_v1"
)
FINAL_FACTORIZED_CANDIDATE = (
    "stagea__drawer-open__possession-grasped__locus-cabinet-side__"
    "support-transverse-low-support__layout-a"
)
MATCHED_NEAR_CANDIDATE = (
    "stagea__drawer-open__possession-grasped__locus-cabinet-side__"
    "support-demonstration-near__layout-a"
)
IMPLEMENTATION_PATHS = {
    "analyzer": PROJECT / "scripts/analyze_phase3b_competence_gap.py",
    "competence_gap_module": (
        PROJECT / "src/smolvla_analysis/phase3b_competence_gap.py"
    ),
    "consolidation_verifier": (
        PROJECT / "scripts/verify_phase3b_stage_a_consolidation.py"
    ),
    "stage_a_lattice": PROJECT / "src/smolvla_analysis/phase3b_stage_a.py",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify compact Stage A competence-gap evidence."
    )
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_competence_gap(report_dir: Path) -> dict:
    report_dir = report_dir.resolve()
    manifest = json.loads((report_dir / "manifest.json").read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("analysis_revision")
        != "competence_compatibility_gap_v1"
        or manifest.get("status") != "complete"
    ):
        raise ValueError("Competence-gap manifest contract changed")
    expected_artifacts = manifest.get("artifact_sha256")
    observed_artifacts = {
        path.name for path in report_dir.iterdir() if path.is_file()
    } - {"manifest.json"}
    if (
        not isinstance(expected_artifacts, dict)
        or observed_artifacts != set(expected_artifacts)
    ):
        raise ValueError("Competence-gap artifact inventory changed")
    for name, expected_sha in expected_artifacts.items():
        if _file_sha256(report_dir / name) != expected_sha:
            raise ValueError(f"Competence-gap artifact hash mismatch: {name}")
    expected_implementation = {
        name: _file_sha256(path) for name, path in IMPLEMENTATION_PATHS.items()
    }
    if manifest.get("implementation_sha256") != expected_implementation:
        raise ValueError("Competence-gap implementation provenance changed")

    source_report = PROJECT / str(manifest["source_report_dir"])
    verify_report(source_report)
    source_manifest_sha = _file_sha256(source_report / "manifest.json")
    if manifest.get("source_manifest_sha256") != source_manifest_sha:
        raise ValueError("Competence-gap source consolidation changed")

    summary = json.loads((report_dir / "summary.json").read_text())
    if (
        summary.get("schema_version") != 1
        or summary.get("status") != "complete"
        or summary.get("policy_loaded") is not False
        or summary.get("goal_cell_count") != 64
        or summary.get("physically_feasible_goal_cell_count") != 64
        or summary.get("proposal_compatible_goal_cell_count") != 63
        or summary.get("competence_compatibility_gap_cell_count") != 1
        or summary.get("competence_compatibility_gap_fraction") != 1 / 64
        or summary.get("source_manifest_sha256") != source_manifest_sha
    ):
        raise ValueError("Competence-gap summary changed")

    cells = pd.read_csv(report_dir / "goal_cells.csv")
    gaps = pd.read_csv(report_dir / "gap_cells.csv")
    pairs = pd.read_csv(report_dir / "matched_gap_pairs.csv")
    if (
        len(cells) != 64
        or cells["candidate_id"].nunique() != 32
        or set(cells["goal"]) != {"drawer", "cabinet"}
        or not cells["physical_feasible"].all()
        or int(cells["proposal_compatible"].sum()) != 63
        or int(cells["competence_compatibility_gap"].sum()) != 1
        or cells["feasibility_policy_loaded"].any()
    ):
        raise ValueError("Competence-gap goal-cell table changed")
    if len(gaps) != 1:
        raise ValueError("Competence-gap cell identity changed")
    gap = gaps.iloc[0]
    if (
        gap["candidate_id"] != FINAL_FACTORIZED_CANDIDATE
        or gap["goal"] != "cabinet"
        or gap["source_id"] != "v38_factorized_promotion"
        or gap["feasibility_kind"] != "factorized_policy_free_path_v1"
        or bool(gap["physical_feasible"]) is not True
        or bool(gap["proposal_compatible"]) is not False
        or int(gap["proposal_attempt_count"]) != 46
        or int(gap["proposal_success_count"]) != 0
        or int(gap["selected_proposal_count"]) != 0
        or int(gap["bridge_evidence_count"]) != 46
        or int(gap["bridge_pass_count"]) != 46
        or int(gap["wrong_goal_count"]) != 0
        or int(gap["unexpected_terminal_count"]) != 0
        or int(gap["factorized_placement_action_count"]) != 204
        or gap["state_sha256"]
        != "e800c3ac08e6c4f66486f2c6e637b22683474b6696e00d837661abab24ad4255"
    ):
        raise ValueError("Competence-gap factorized cell changed")
    if len(pairs) != 1:
        raise ValueError("Competence-gap matched pair changed")
    pair = pairs.iloc[0]
    if (
        pair["goal"] != "cabinet"
        or pair["gap_candidate_id"] != FINAL_FACTORIZED_CANDIDATE
        or pair["comparison_candidate_id"] != MATCHED_NEAR_CANDIDATE
        or int(pair["gap_proposal_success_count"]) != 0
        or int(pair["comparison_proposal_success_count"]) != 1
        or bool(pair["same_proposal_bank"]) is not True
        or bool(pair["same_execution_contract"]) is not True
        or bool(pair["proposal_outcome_estimable"]) is not True
        or bool(pair["selected_cost_estimable"]) is not False
        or int(pair["shared_success_count"]) != 0
        or float(pair["success_set_jaccard"]) != 0.0
    ):
        raise ValueError("Competence-gap matched comparison changed")
    return {
        "pass": True,
        "analysis_revision": manifest["analysis_revision"],
        "artifact_count": len(expected_artifacts),
        "goal_cell_count": len(cells),
        "physical_feasible_goal_cell_count": int(
            cells["physical_feasible"].sum()
        ),
        "proposal_compatible_goal_cell_count": int(
            cells["proposal_compatible"].sum()
        ),
        "competence_compatibility_gap_cell_count": len(gaps),
        "source_manifest_sha256": source_manifest_sha,
    }


def main() -> None:
    args = _parse_args()
    result = verify_competence_gap(args.report_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
