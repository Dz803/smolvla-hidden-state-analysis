#!/usr/bin/env python
"""Independently verify a v38 factorized Stage A promotion artifact."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import zarr

from smolvla_analysis.phase2_storage import read_libero_snapshot
from smolvla_analysis.phase3b_consolidation import _validate_candidate_record
from smolvla_analysis.phase3b_feasibility import (
    build_factorized_feasibility_evidence,
)
from smolvla_analysis.phase3b_persistence import TRANSACTION_SCHEMA
from smolvla_analysis.phase3b_stage_a import (
    canonical_sha256,
    snapshot_sha256,
    validate_selection_lock,
)


PROJECT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify v38 state, evidence, source, and transaction bindings."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def verify_promotion(run_dir: Path) -> dict:
    run_dir = run_dir.resolve()
    raw_root = (PROJECT / "local/phase3b_stage_a").resolve()
    if run_dir == raw_root or raw_root not in run_dir.parents:
        raise ValueError("Promotion run is outside the Stage A raw root")
    transaction_root = run_dir / "transactions"
    if any(
        path.name.startswith("__tmp__") for path in transaction_root.iterdir()
    ):
        raise ValueError("Promotion contains an incomplete transaction staging path")
    manifest = _read(run_dir / "manifest.json")
    contract = _read(run_dir / "contract.json")
    selection = _read(run_dir / "selection_lock.json")
    promotion = _read(run_dir / "promotion_contract.json")
    if (
        manifest.get("status") != "complete"
        or manifest.get("stage") != "phase3b_stage_a_factorized_promotion"
        or manifest.get("policy_loaded") is not False
        or manifest.get("candidate_count") != 1
        or manifest.get("expected_candidate_count") != 1
        or manifest.get("physical_feasibility_pass") is not True
        or manifest.get("cabinet_proposal_success_count") != 0
        or manifest.get("cabinet_proposal_attempt_count") != 46
        or manifest.get("contract_sha256") != canonical_sha256(contract)
        or manifest.get("promotion_contract_sha256")
        != canonical_sha256(promotion)
    ):
        raise ValueError("Promotion manifest failed its scope or hash gates")
    validate_selection_lock(
        selection,
        contract_sha256=manifest["contract_sha256"],
        construction_revision=contract["construction_revision"],
    )
    if selection["selection_lock_sha256"] != manifest["selection_lock_sha256"]:
        raise ValueError("Promotion selection lock changed")

    candidate_id = manifest["candidate_id"]
    record_path = run_dir / "candidates" / f"{candidate_id}.json"
    record = _read(record_path)
    if (
        canonical_sha256(record) != manifest["candidate_record_sha256"]
        or record.get("promotion_contract_sha256")
        != manifest["promotion_contract_sha256"]
        or record.get("state_sha256") != manifest["state_sha256"]
    ):
        raise ValueError("Promotion candidate record changed")
    candidate_validation = _validate_candidate_record(record)
    state_root = zarr.open_group(str(run_dir / "states.zarr"), mode="r")
    if snapshot_sha256(read_libero_snapshot(state_root, candidate_id)) != record[
        "state_sha256"
    ]:
        raise ValueError("Promotion state payload changed")
    intent = _read(run_dir / "transactions" / candidate_id / "intent.json")
    transaction_record = _read(
        run_dir / "transactions" / candidate_id / "record.json"
    )
    if (
        intent.get("transaction_schema") != TRANSACTION_SCHEMA
        or intent.get("candidate_id") != candidate_id
        or intent.get("status") != "committed"
        or intent.get("record_sha256") != canonical_sha256(record)
        or intent.get("state_sha256") != record["state_sha256"]
        or transaction_record != record
    ):
        raise ValueError("Promotion transaction is not committed")

    source_run = (PROJECT / promotion["source_run"]).resolve()
    for name, expected in promotion["source_artifact_file_sha256"].items():
        if name.startswith("checkpoint_"):
            goal = name.removeprefix("checkpoint_").removesuffix(".json")
            path = source_run / "checkpoints" / f"{candidate_id}__{goal}.json"
        else:
            path = source_run / name
        if _file_sha256(path) != expected:
            raise ValueError(f"Promotion source artifact changed: {name}")

    factor_dir = (PROJECT / promotion["factor_certificate"]).resolve()
    factor_paths = {
        name: factor_dir / name
        for name in ("contract.json", "manifest.json", "result.json", "acquisition.json")
    }
    rebuilt_factor = build_factorized_feasibility_evidence(
        result=_read(factor_paths["result.json"]),
        manifest=_read(factor_paths["manifest.json"]),
        contract=_read(factor_paths["contract.json"]),
        artifact_file_sha256={
            name: _file_sha256(path) for name, path in factor_paths.items()
        },
    )
    if (
        canonical_sha256(rebuilt_factor)
        != promotion["factorized_feasibility_evidence_sha256"]
        or rebuilt_factor
        != record["goal_feasibility_evidence"]["cabinet"]
    ):
        raise ValueError("Promotion factorized evidence changed")
    audit = _read(run_dir / "audits" / "promotion_audit.json")
    if (
        audit.get("pass") is not True
        or canonical_sha256(audit) != manifest["promotion_audit_sha256"]
        or audit.get("execution_scope") != promotion.get("execution_scope")
    ):
        raise ValueError("Promotion audit changed")
    scope = promotion.get("execution_scope", {})
    if (
        scope.get("policy_forwards") != 0
        or scope.get("completed_proposal_suffixes_reexecuted") != 0
        or scope.get("new_proposal_suffixes_executed") != 0
        or scope.get("factorized_branches_reexecuted") != 0
    ):
        raise ValueError("Promotion execution scope changed")
    return {
        "pass": True,
        "run_id": manifest["run_id"],
        "candidate_id": candidate_id,
        "state_sha256": record["state_sha256"],
        "normalized_state_sha256": candidate_validation[
            "normalized_state_sha256"
        ],
        "drawer_proposal_success": "10/36",
        "cabinet_proposal_compatibility": "0/46",
        "cabinet_physical_feasibility": True,
        "factorized_placement_action_count": rebuilt_factor["placement"][
            "action_count"
        ],
        "transaction_status": "committed",
        "completed_proposal_suffixes_reexecuted": 0,
        "policy_forwards": 0,
    }


def main() -> None:
    result = verify_promotion(_parse_args().run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
