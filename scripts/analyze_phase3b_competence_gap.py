#!/usr/bin/env python
"""Publish the Stage A physical-recoverability/proposal-compatibility gap."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pandas as pd

from smolvla_analysis.phase3_crd import atomic_write_json
from smolvla_analysis.phase3b_competence_gap import (
    build_gap_pair_table,
    build_goal_cell_table,
    summarize_competence_gap,
)
from smolvla_analysis.phase3b_stage_a import iter_candidate_specs
from verify_phase3b_stage_a_consolidation import verify_report


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    PROJECT
    / "reports/phase3b_stage_a/phase3b-stage-a-consolidated-v4"
)
DEFAULT_OUTPUT = (
    PROJECT / "reports/phase3b_stage_a/competence_compatibility_gap_v1"
)
FINAL_FACTORIZED_CANDIDATE = (
    "stagea__drawer-open__possession-grasped__locus-cabinet-side__"
    "support-transverse-low-support__layout-a"
)
IMPLEMENTATION_PATHS = {
    "analyzer": Path(__file__).resolve(),
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
        description=(
            "Derive a compact competence-compatibility analysis from verified "
            "Stage A consolidation tables."
        )
    )
    parser.add_argument("--source-report-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _readme(summary: dict) -> str:
    return f"""# Stage A competence–compatibility gap

This compact report separates two quantities that an ordinary oracle-success flag
conflates:

- `F_C(s,g)`: whether a declared policy-independent controller/certificate class
  contains a physically successful path;
- `P_K(s,g) = max_k Y(N(s),g,k)`: whether any member of the complete, registered
  replay bank succeeds under its bound execution contract.

The observed gap is `F_C=1, P_K=0`. All
`{summary['goal_cell_count']}` state-goal cells are physically certified, while
`{summary['proposal_compatible_goal_cell_count']}` have a successful replay-bank
member. The sole gap cell preserves all 46 cabinet failures. Every bridge passes,
no wrong goal is reached, and no attempt terminates early. A separate factorized
path reaches the cabinet using stable acquisition followed by 204 feedback
transport/release actions.

This falsifies the inference that failure of this finite replay bank proves physical
infeasibility. It does **not** show that SmolVLA solves the cell, that a hidden state
causes the failure, or that the adaptively developed factorized certificate
generalises. Stage A loaded no VLA. The certificate is an existence witness for one
cell and must be frozen before a held-out confirmatory study.

Files:

- `goal_cells.csv`: one row for each physical state and goal, with the evidence
  classes kept separate;
- `gap_cells.csv`: only cells where physical feasibility passes but proposal
  compatibility fails;
- `matched_gap_pairs.csv`: the gap cell and its demonstration-near matched support
  state under the same proposal bank/execution contract;
- `summary.json`: estimands, counts, and claim boundary;
- `manifest.json`: source and artifact hashes.
"""


def main() -> None:
    args = _parse_args()
    source = args.source_report_dir.resolve()
    output = args.output_dir.resolve()
    verify_report(source)

    candidates = pd.read_csv(source / "candidate_inventory.csv")
    proposals = pd.read_csv(source / "proposal_coverage.csv")
    feasibility = pd.read_csv(source / "goal_feasibility_evidence.csv")
    support_pairs = pd.read_csv(source / "support_pairs.csv")
    goal_cells = build_goal_cell_table(candidates, proposals, feasibility)
    gap_cells = goal_cells[goal_cells["competence_compatibility_gap"]].copy()
    gap_pairs = build_gap_pair_table(
        goal_cells,
        support_pairs,
        iter_candidate_specs(),
    )
    summary = summarize_competence_gap(goal_cells, gap_pairs)

    if (
        len(gap_cells) != 1
        or gap_cells.iloc[0]["candidate_id"] != FINAL_FACTORIZED_CANDIDATE
        or gap_cells.iloc[0]["goal"] != "cabinet"
        or int(gap_cells.iloc[0]["proposal_attempt_count"]) != 46
        or int(gap_cells.iloc[0]["proposal_success_count"]) != 0
        or int(gap_cells.iloc[0]["bridge_pass_count"]) != 46
        or int(gap_cells.iloc[0]["wrong_goal_count"]) != 0
        or int(gap_cells.iloc[0]["unexpected_terminal_count"]) != 0
    ):
        raise ValueError("Stage A competence-compatibility gap identity changed")

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite analysis: {output}")
    staging = output.with_name(f".{output.name}__tmp__pid{os.getpid()}")
    if staging.exists():
        quarantine_root = PROJECT / "local/phase3b_stage_a/failed_analyses"
        quarantine_root.mkdir(parents=True, exist_ok=True)
        quarantine = quarantine_root / (
            f"{staging.name}__{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
        )
        os.replace(staging, quarantine)
    staging.mkdir(parents=True)

    goal_cells.to_csv(staging / "goal_cells.csv", index=False)
    gap_cells.to_csv(staging / "gap_cells.csv", index=False)
    gap_pairs.to_csv(staging / "matched_gap_pairs.csv", index=False)
    source_manifest = source / "manifest.json"
    source_payload = json.loads(source_manifest.read_text())
    summary.update(
        {
            "analysis_revision": "competence_compatibility_gap_v1",
            "source_consolidation_revision": source_payload[
                "consolidation_revision"
            ],
            "source_manifest_sha256": _file_sha256(source_manifest),
        }
    )
    atomic_write_json(staging / "summary.json", summary)
    (staging / "README.md").write_text(_readme(summary))

    artifact_sha256 = {
        path.name: _file_sha256(path)
        for path in sorted(staging.iterdir())
        if path.is_file()
    }
    atomic_write_json(
        staging / "manifest.json",
        {
            "schema_version": 1,
            "analysis_revision": "competence_compatibility_gap_v1",
            "status": "complete",
            "created_at": datetime.now(UTC).isoformat(),
            "source_report_dir": str(source.relative_to(PROJECT)),
            "source_manifest_sha256": _file_sha256(source_manifest),
            "implementation_sha256": {
                name: _file_sha256(path)
                for name, path in IMPLEMENTATION_PATHS.items()
            },
            "artifact_sha256": artifact_sha256,
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Competence-compatibility analysis complete: {output}")


if __name__ == "__main__":
    main()
