#!/usr/bin/env python
"""Resume frozen v35 with a provenance-bound registered-ledger validator."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT / "src"))

try:
    import run_phase3b_stage_a_completion as runner
except ModuleNotFoundError:
    from scripts import run_phase3b_stage_a_completion as runner

from smolvla_analysis.phase3_crd import atomic_write_json
from smolvla_analysis.phase3b_registered_validation import (
    validate_support_pair_records_compatible,
)


EXPECTED_FAILURE = (
    "Invalid drawer proposal-execution contract for "
    "stagea__drawer-open__possession-on-table__locus-cabinet-side__"
    "support-demonstration-near__layout-a"
)


def _recovery_record(run_dir: Path) -> tuple[Path, dict]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    contract = json.loads((run_dir / "contract.json").read_text())
    frozen_runner = PROJECT / "scripts/run_phase3b_stage_a_completion.py"
    frozen_validation = PROJECT / "src/smolvla_analysis/phase3b_stage_a.py"
    if (
        manifest.get("contract_sha256") != runner.canonical_sha256(contract)
        or manifest.get("failure_message") != EXPECTED_FAILURE
        or contract.get("completion_revision") != runner.COMPLETION_REVISION
        or contract.get("completion_source_sha256", {}).get("runner")
        != runner._file_sha256(frozen_runner)
        or contract.get("source_sha256", {}).get("lattice")
        != runner._file_sha256(frozen_validation)
    ):
        raise ValueError("Frozen v35 recovery preconditions changed")
    known_errors = sorted((run_dir / "errors").glob("*.json"))
    matching_errors = [
        path
        for path in known_errors
        if json.loads(path.read_text()).get("message") == EXPECTED_FAILURE
    ]
    if len(matching_errors) != 1:
        raise ValueError("Frozen v35 recovery requires exactly one known error")
    path = run_dir / "audits" / "registered_validation_compatibility.json"
    record = {
        "schema_version": 1,
        "recovery_revision": "phase3b-v35-registered-validation-compat-v1",
        "status": "in_progress",
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": manifest["run_id"],
        "contract_sha256": manifest["contract_sha256"],
        "known_error_file": matching_errors[0].name,
        "known_error_file_sha256": runner._file_sha256(matching_errors[0]),
        "frozen_runner_sha256": runner._file_sha256(frozen_runner),
        "frozen_validation_sha256": runner._file_sha256(frozen_validation),
        "compatibility_runner_sha256": runner._file_sha256(Path(__file__)),
        "compatibility_validator_sha256": runner._file_sha256(
            PROJECT
            / "src/smolvla_analysis/phase3b_registered_validation.py"
        ),
        "scope": "post-oracle support-pair ledger validation only",
        "proposal_execution_changed": False,
        "state_construction_changed": False,
        "completed_proposals_reexecuted": 0,
    }
    if path.is_file():
        existing = json.loads(path.read_text())
        for key, value in record.items():
            if key not in {"status", "created_at"} and existing.get(key) != value:
                raise ValueError(f"Changed v35 recovery provenance: {key}")
        record["created_at"] = existing["created_at"]
    atomic_write_json(path, record)
    return path, record


def main() -> None:
    args = runner._parse_args()
    if args.run_dir is None:
        raise ValueError("Compatibility resume requires an explicit --run-dir")
    run_dir = runner._run_dir(args.run_dir)
    audit_path, audit = _recovery_record(run_dir)
    runner.validate_support_pair_records = validate_support_pair_records_compatible
    runner.main()
    final_manifest = json.loads((run_dir / "manifest.json").read_text())
    audit.update(
        {
            "status": "complete",
            "completed_at": datetime.now(UTC).isoformat(),
            "post_resume_manifest_status": final_manifest["status"],
            "post_resume_candidate_count": final_manifest["candidate_count"],
        }
    )
    atomic_write_json(audit_path, audit)


if __name__ == "__main__":
    main()
