#!/usr/bin/env python
"""Report resumable v35 completion progress without reading large result payloads."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

try:
    from smolvla_analysis.phase3b_stage_a import canonical_sha256
except ModuleNotFoundError:  # Support direct execution from a source checkout.
    sys.path.insert(0, str(PROJECT / "src"))
    from smolvla_analysis.phase3b_stage_a import canonical_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a v35 completion shard.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--brief",
        action="store_true",
        help="Print aggregate progress and only the first incomplete candidate.",
    )
    return parser.parse_args()


def summarize(run_dir: Path) -> dict:
    run_dir = run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    contract = json.loads((run_dir / "contract.json").read_text())
    if manifest.get("contract_sha256") != canonical_sha256(contract):
        raise ValueError("Completion status found a changed contract")
    assignment = contract.get("completion_assignment", {})
    candidate_ids = assignment.get("candidate_ids")
    if not isinstance(candidate_ids, list) or len(candidate_ids) != int(
        assignment.get("expected_candidate_count", -1)
    ):
        raise ValueError("Completion status found a changed candidate assignment")
    expected_proposals = {
        goal: len(contract["oracle_proposal_bank"][goal])
        for goal in ("drawer", "cabinet")
    }

    rows = []
    for candidate_id in candidate_ids:
        record_path = run_dir / "candidates" / f"{candidate_id}.json"
        state_path = run_dir / "states.zarr" / candidate_id
        record = json.loads(record_path.read_text()) if record_path.is_file() else None
        checkpoints = {}
        for goal in ("drawer", "cabinet"):
            path = run_dir / "checkpoints" / f"{candidate_id}__{goal}.json"
            if not path.is_file():
                oracle = record.get("oracles", {}).get(goal) if record else None
                imported = record.get("oracle_imports", {}).get(goal) if record else None
                checkpoints[goal] = {
                    "status": (
                        "record_complete_imported"
                        if imported is not None
                        else "not_started"
                    ),
                    "result_count": int(
                        oracle.get("proposal_attempt_count", 0)
                        if oracle is not None
                        else 0
                    ),
                    "expected_result_count": expected_proposals[goal],
                    "success_count": int(
                        oracle.get("proposal_success_count", 0)
                        if oracle is not None
                        else 0
                    ),
                    "imported_result_count": int(
                        oracle.get("proposal_attempt_count", 0)
                        if imported is not None and oracle is not None
                        else 0
                    ),
                }
                continue
            checkpoint = json.loads(path.read_text())
            result_rows = checkpoint.get("results")
            if not isinstance(result_rows, list):
                raise ValueError(f"Invalid checkpoint rows: {candidate_id}/{goal}")
            indices = [int(item["proposal_index"]) for item in result_rows]
            if (
                len(indices) != len(set(indices))
                or any(index not in range(expected_proposals[goal]) for index in indices)
                or int(checkpoint.get("result_count", -1)) != len(indices)
            ):
                raise ValueError(
                    f"Invalid checkpoint inventory: {candidate_id}/{goal}"
                )
            checkpoints[goal] = {
                "status": checkpoint.get("status"),
                "result_count": len(indices),
                "expected_result_count": expected_proposals[goal],
                "success_count": sum(
                    item["result"].get("pass") is True for item in result_rows
                ),
                "imported_result_count": sum(
                    item.get("provenance", {}).get("kind")
                    != "simulated_in_completion_shard"
                    for item in result_rows
                ),
            }
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_record_complete": record_path.is_file(),
                "state_payload_complete": state_path.is_dir(),
                "checkpoints": checkpoints,
            }
        )
    completed = [row for row in rows if row["candidate_record_complete"]]
    errors = sorted(
        path.name for path in (run_dir / "errors").glob("*.json") if path.is_file()
    )
    return {
        "run_id": manifest["run_id"],
        "status": manifest["status"],
        "manifest_candidate_count": int(manifest["candidate_count"]),
        "observed_candidate_count": len(completed),
        "expected_candidate_count": len(candidate_ids),
        "first_incomplete_candidate": next(
            (
                row["candidate_id"]
                for row in rows
                if not row["candidate_record_complete"]
            ),
            None,
        ),
        "error_count": len(errors),
        "errors": errors,
        "candidates": rows,
    }


def brief_summary(report: dict) -> dict:
    first_incomplete = report["first_incomplete_candidate"]
    active = next(
        (
            row
            for row in report["candidates"]
            if row["candidate_id"] == first_incomplete
        ),
        None,
    )
    return {
        key: report[key]
        for key in (
            "run_id",
            "status",
            "manifest_candidate_count",
            "observed_candidate_count",
            "expected_candidate_count",
            "first_incomplete_candidate",
            "error_count",
            "errors",
        )
    } | {"active_candidate": active}


def main() -> None:
    args = _parse_args()
    report = summarize(args.run_dir)
    if args.brief:
        report = brief_summary(report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
