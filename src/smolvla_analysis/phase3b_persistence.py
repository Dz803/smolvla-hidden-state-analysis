"""Crash-recoverable persistence for coupled Stage A state/record artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import zarr

from .phase2_storage import read_libero_snapshot, write_libero_snapshot
from .phase3_crd import atomic_write_json
from .phase3b_stage_a import canonical_sha256, snapshot_sha256


TRANSACTION_SCHEMA = "stage_a_candidate_artifact_transaction_v1"


def _paths(run_dir: Path, candidate_id: str) -> dict[str, Path]:
    transaction = run_dir / "transactions" / candidate_id
    return {
        "transaction": transaction,
        "intent": transaction / "intent.json",
        "transaction_record": transaction / "record.json",
        "transaction_state_root": transaction / "state.zarr",
        "transaction_state": transaction / "state.zarr" / candidate_id,
        "final_record": run_dir / "candidates" / f"{candidate_id}.json",
        "final_state_root": run_dir / "states.zarr",
        "final_state": run_dir / "states.zarr" / candidate_id,
    }


def _validate_record(record: dict[str, Any], *, candidate_id: str) -> str:
    if record.get("candidate_id") != candidate_id:
        raise ValueError("Candidate transaction record identity changed")
    state_sha = record.get("state_sha256")
    if not isinstance(state_sha, str) or len(state_sha) != 64:
        raise ValueError("Candidate transaction has no state hash")
    return state_sha


def _validate_state(
    state_root_path: Path, *, candidate_id: str, expected_sha256: str
) -> None:
    root = zarr.open_group(str(state_root_path), mode="r")
    if candidate_id not in root:
        raise ValueError("Candidate transaction state group is missing")
    group = root[candidate_id]
    if (
        group.attrs.get("complete") is not True
        or group.attrs.get("state_sha256") != expected_sha256
    ):
        raise ValueError("Candidate transaction state attributes changed")
    if snapshot_sha256(read_libero_snapshot(root, candidate_id)) != expected_sha256:
        raise ValueError("Candidate transaction state payload changed")


def stage_candidate_transaction(
    run_dir: Path,
    *,
    candidate_id: str,
    snapshot: Any,
    record: dict[str, Any],
) -> Path:
    """Durably stage both artifacts before either appears in final inventory."""

    run_dir = run_dir.resolve()
    paths = _paths(run_dir, candidate_id)
    state_sha = _validate_record(record, candidate_id=candidate_id)
    if snapshot_sha256(snapshot) != state_sha:
        raise ValueError("Candidate transaction snapshot hash changed")
    if (
        paths["transaction"].exists()
        or paths["final_record"].exists()
        or paths["final_state"].exists()
    ):
        raise ValueError(f"Refusing to overwrite candidate artifacts: {candidate_id}")
    transaction_parent = paths["transaction"].parent
    transaction_parent.mkdir(parents=True, exist_ok=True)
    staging = transaction_parent / f"__tmp__{candidate_id}__pid{os.getpid()}"
    if staging.exists():
        raise ValueError(f"Stale candidate transaction staging path: {staging}")
    staging.mkdir()
    staged_state_root = zarr.open_group(str(staging / "state.zarr"), mode="w")
    write_libero_snapshot(staged_state_root, candidate_id, snapshot)
    staged_state_root[candidate_id].attrs.update(
        {"state_sha256": state_sha, "complete": True}
    )
    atomic_write_json(staging / "record.json", record)
    intent = {
        "schema_version": 1,
        "transaction_schema": TRANSACTION_SCHEMA,
        "candidate_id": candidate_id,
        "state_sha256": state_sha,
        "record_sha256": canonical_sha256(record),
        "status": "prepared",
    }
    atomic_write_json(staging / "intent.json", intent)
    _validate_state(
        staging / "state.zarr",
        candidate_id=candidate_id,
        expected_sha256=state_sha,
    )
    os.replace(staging, paths["transaction"])
    return paths["transaction"]


def commit_candidate_transaction(run_dir: Path, *, candidate_id: str) -> None:
    """Complete or recover the idempotent two-artifact commit."""

    run_dir = run_dir.resolve()
    paths = _paths(run_dir, candidate_id)
    if not paths["intent"].is_file() or not paths["transaction_record"].is_file():
        raise ValueError(f"Candidate transaction is incomplete: {candidate_id}")
    intent = json.loads(paths["intent"].read_text())
    record = json.loads(paths["transaction_record"].read_text())
    state_sha = _validate_record(record, candidate_id=candidate_id)
    if (
        intent.get("transaction_schema") != TRANSACTION_SCHEMA
        or intent.get("candidate_id") != candidate_id
        or intent.get("state_sha256") != state_sha
        or intent.get("record_sha256") != canonical_sha256(record)
        or intent.get("status") not in {"prepared", "committed"}
    ):
        raise ValueError(f"Candidate transaction intent changed: {candidate_id}")

    if paths["final_state"].exists():
        _validate_state(
            paths["final_state_root"],
            candidate_id=candidate_id,
            expected_sha256=state_sha,
        )
    else:
        if not paths["transaction_state"].is_dir():
            raise ValueError(f"Candidate transaction lost its state: {candidate_id}")
        _validate_state(
            paths["transaction_state_root"],
            candidate_id=candidate_id,
            expected_sha256=state_sha,
        )
        zarr.open_group(str(paths["final_state_root"]), mode="a")
        os.replace(paths["transaction_state"], paths["final_state"])

    if paths["final_record"].is_file():
        if json.loads(paths["final_record"].read_text()) != record:
            raise ValueError(f"Candidate final record changed: {candidate_id}")
    else:
        paths["final_record"].parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(paths["final_record"], record)

    _validate_state(
        paths["final_state_root"],
        candidate_id=candidate_id,
        expected_sha256=state_sha,
    )
    if json.loads(paths["final_record"].read_text()) != record:
        raise ValueError(f"Candidate final record verification failed: {candidate_id}")
    intent["status"] = "committed"
    atomic_write_json(paths["intent"], intent)


def persist_candidate_artifacts_transactional(
    run_dir: Path,
    *,
    candidate_id: str,
    snapshot: Any,
    record: dict[str, Any],
) -> None:
    stage_candidate_transaction(
        run_dir,
        candidate_id=candidate_id,
        snapshot=snapshot,
        record=record,
    )
    commit_candidate_transaction(run_dir, candidate_id=candidate_id)


def recover_candidate_transactions(run_dir: Path) -> list[str]:
    """Finish every prepared transaction; committed transactions are reverified."""

    transaction_root = run_dir.resolve() / "transactions"
    if not transaction_root.exists():
        return []
    temporary = sorted(
        path.name
        for path in transaction_root.iterdir()
        if path.name.startswith("__tmp__")
    )
    if temporary:
        raise ValueError(
            "Pre-commit candidate transaction staging requires inspection: "
            f"{temporary}"
        )
    recovered = []
    for transaction in sorted(transaction_root.iterdir()):
        if not transaction.is_dir():
            raise ValueError(f"Unknown candidate transaction artifact: {transaction}")
        candidate_id = transaction.name
        before = json.loads((transaction / "intent.json").read_text()).get(
            "status"
        )
        commit_candidate_transaction(run_dir, candidate_id=candidate_id)
        if before == "prepared":
            recovered.append(candidate_id)
    return recovered
