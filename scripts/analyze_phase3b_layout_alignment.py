#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from smolvla_analysis.phase3_crd import atomic_write_json
from smolvla_analysis.phase3b_alignment_report import build_alignment_summary
from smolvla_analysis.phase3b_stage_a import canonical_sha256


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = (
    PROJECT
    / "local/phase3b_stage_a/layout_alignment_diagnostics/"
    "layout_alignment_20260731T070614Z"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and compact a raw Stage A layout-alignment diagnostic."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--report-dir", type=Path)
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_and_verify(raw_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_dir = raw_dir.resolve()
    contract_path = raw_dir / "contract.json"
    result_path = raw_dir / "result.json"
    manifest_path = raw_dir / "manifest.json"
    contract = json.loads(contract_path.read_text())
    result = json.loads(result_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Raw layout-alignment manifest is not complete")
    contract_sha = canonical_sha256(contract)
    if {
        manifest.get("contract_sha256"),
        result.get("contract_sha256"),
    } != {contract_sha}:
        raise ValueError("Raw layout-alignment contract hash changed")
    observed_hashes = {
        path.name: _file_sha256(path)
        for path in (contract_path, result_path)
    }
    if manifest.get("artifact_sha256") != observed_hashes:
        raise ValueError("Raw layout-alignment artifact hashes changed")
    for evidence in contract["source_evidence"].values():
        path = PROJECT / evidence["evidence_file"]
        if _file_sha256(path) != evidence["evidence_file_sha256"]:
            raise ValueError(f"Source evidence changed: {path}")
    return contract, build_alignment_summary(contract, result)


def _write_report(
    raw_dir: Path,
    report_dir: Path,
    contract: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    report_dir = report_dir.resolve()
    report_root = (PROJECT / "reports/phase3b_stage_a").resolve()
    if report_dir == report_root or report_root not in report_dir.parents:
        raise ValueError(f"Compact report must remain under {report_root}")
    if report_dir.exists():
        raise FileExistsError(f"Refusing to overwrite compact report: {report_dir}")
    staging = report_dir.with_name(f".{report_dir.name}__tmp__pid{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"Stale compact-report staging path: {staging}")
    staging.mkdir(parents=True)
    atomic_write_json(staging / "contract.json", contract)
    atomic_write_json(staging / "summary.json", summary)
    pd.DataFrame(summary["conditions"]).to_csv(
        staging / "condition_summary.csv", index=False
    )
    (staging / "README.md").write_text(
        "# Stage A layout-alignment diagnostic\n\n"
        "This compact report verifies a bounded, policy-free diagnostic on the "
        "first v34 layout pair. On the identical certified layout-B normalized "
        "root and the identical episode-474 suffix, the original world-frame "
        "anchor acquires the bowl but misses both goals. A 25 mm bowl-relative "
        "anchor translation reaches the cabinet goal. Reusing that registered "
        "acquisition snapshot, a separate lift/transit/descent/release controller "
        "also preserves the grasp and reaches the cabinet goal.\n\n"
        "The result is an exploratory, post-failure controller-input intervention "
        "on one proposal and one factor cell. It establishes neither a universal "
        "feasibility oracle nor a model or hidden-state mechanism. Raw traces and "
        "simulator snapshots remain workstation-only.\n"
    )
    artifacts = {
        path.name: _file_sha256(path)
        for path in sorted(staging.iterdir())
        if path.is_file()
    }
    atomic_write_json(
        staging / "manifest.json",
        {
            "schema_version": 1,
            "status": "complete",
            "contract_sha256": summary["contract_sha256"],
            "summary_sha256": canonical_sha256(summary),
            "artifact_sha256": artifacts,
            "raw_evidence_location": raw_dir.relative_to(PROJECT).as_posix(),
            "policy_loaded": False,
            "canonical_rollout_reused": False,
        },
    )
    os.replace(staging, report_dir)


def main() -> None:
    args = _parse_args()
    raw_dir = args.raw_dir.resolve()
    report_dir = (
        args.report_dir.resolve()
        if args.report_dir is not None
        else PROJECT / "reports/phase3b_stage_a" / raw_dir.name
    )
    contract, summary = _load_and_verify(raw_dir)
    _write_report(raw_dir, report_dir, contract, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Compact layout-alignment report: {report_dir}")


if __name__ == "__main__":
    main()
