#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from hashlib import sha256
from pathlib import Path

import pandas as pd

from smolvla_analysis.phase3_crd import atomic_write_json
from smolvla_analysis.phase3b_registered_report import (
    build_registered_smoke_summary,
)
from smolvla_analysis.phase3b_stage_a import canonical_sha256


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = (
    PROJECT
    / "local/phase3b_stage_a/registered_generalization_smokes/"
    "registered_smoke_20260731T073347Z"
)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and compact the prospective registered held-root smoke."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--report-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    raw_dir = args.raw_dir.resolve()
    paths = {name: raw_dir / name for name in ("contract.json", "result.json")}
    manifest = json.loads((raw_dir / "manifest.json").read_text())
    contract = json.loads(paths["contract.json"].read_text())
    result = json.loads(paths["result.json"].read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Raw registered-smoke manifest is not complete")
    contract_sha = canonical_sha256(contract)
    if {
        manifest.get("contract_sha256"),
        result.get("contract_sha256"),
    } != {contract_sha}:
        raise ValueError("Raw registered-smoke contract hash changed")
    observed = {name: _file_sha256(path) for name, path in paths.items()}
    if manifest.get("artifact_sha256") != observed:
        raise ValueError("Raw registered-smoke artifact hashes changed")
    source_paths = {
        "script": PROJECT / "scripts/run_phase3b_registered_generalization_smoke.py",
        "runtime": PROJECT / "src/smolvla_analysis/phase3b_libero.py",
        "lattice": PROJECT / "src/smolvla_analysis/phase3b_stage_a.py",
        "completion": PROJECT / "src/smolvla_analysis/phase3b_completion.py",
        "alignment": PROJECT / "src/smolvla_analysis/phase3b_alignment.py",
    }
    if contract["source_sha256"] != {
        key: _file_sha256(path) for key, path in source_paths.items()
    }:
        raise ValueError("Registered-smoke source files changed")
    summary = build_registered_smoke_summary(contract, result)
    report_dir = (
        args.report_dir.resolve()
        if args.report_dir is not None
        else PROJECT / "reports/phase3b_stage_a" / raw_dir.name
    )
    report_root = (PROJECT / "reports/phase3b_stage_a").resolve()
    if report_dir == report_root or report_root not in report_dir.resolve().parents:
        raise ValueError(f"Compact report must remain under {report_root}")
    if report_dir.exists():
        raise FileExistsError(f"Refusing to overwrite compact report: {report_dir}")
    staging = report_dir.with_name(f".{report_dir.name}__tmp__pid{os.getpid()}")
    staging.mkdir(parents=True)
    atomic_write_json(staging / "contract.json", contract)
    atomic_write_json(staging / "summary.json", summary)
    pd.DataFrame(summary["conditions"]).to_csv(
        staging / "condition_summary.csv", index=False
    )
    (staging / "README.md").write_text(
        "# v35 registered held-root smoke\n\n"
        "The bowl-relative proposal rule was frozen on the earlier near-support "
        "layout pair, then evaluated once on the two untouched transverse roots. "
        "Episode 474 reaches the cabinet in both layouts (source frames 83 and "
        "79), with certified roots, valid bridges, zero bowl drift, preserved "
        "drawer aperture, and no wrong-goal or early-terminal event.\n\n"
        "This is prospective held-root transfer for one proposal inside one task, "
        "not universal proposal coverage or model evidence. Each raw attempt is "
        "eligible for hash-bound import into the v35 completion shard so it is not "
        "simulated twice.\n"
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
            "contract_sha256": contract_sha,
            "summary_sha256": canonical_sha256(summary),
            "artifact_sha256": artifacts,
            "raw_evidence_location": raw_dir.relative_to(PROJECT).as_posix(),
            "policy_loaded": False,
        },
    )
    os.replace(staging, report_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Compact registered-smoke report: {report_dir}")


if __name__ == "__main__":
    main()
