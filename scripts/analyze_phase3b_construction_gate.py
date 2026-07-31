#!/usr/bin/env python
"""Verify and compact the policy-free v36 construction smoke."""

from __future__ import annotations

import argparse
import json
import os
from hashlib import sha256
from pathlib import Path

import pandas as pd

from smolvla_analysis.phase3_crd import atomic_write_json
from smolvla_analysis.phase3b_stage_a import canonical_sha256


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = (
    PROJECT
    / "local/phase3b_stage_a/construction_gates/"
    "phase3b_stage_a_construction_gate_20260731T094517Z"
)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and compact the v36 six-root construction gate."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--report-dir", type=Path)
    return parser.parse_args()


def _candidate_row(record: dict, result_row: dict) -> dict:
    candidate_id = record.get("candidate_id")
    construction = record.get("construction", {})
    acquisition = construction.get("grasp_acquisition", {})
    bridge = acquisition.get("bridge", {})
    safe_lift = construction.get("safe_lift", {})
    padding = construction.get("padding", {})
    support = record.get("support_measurement", {})
    certificate = record.get("certificate", {})
    checks = record.get("construction_contract_checks", {})
    if (
        record.get("pass") is not True
        or record.get("policy_loaded") is not False
        or int(record.get("proposal_oracles_executed", -1)) != 0
        or record.get("construction_contract_pass") is not True
        or not checks
        or not all(checks.values())
        or record.get("root_validation", {}).get("pass") is not True
        or support.get("pass") is not True
        or certificate.get("pass") is not True
        or acquisition.get("final_grasped") is not True
        or acquisition.get("final_goals")
        != {"drawer": False, "cabinet": False}
        or result_row.get("pass") is not True
        or result_row.get("state_sha256") != record.get("state_sha256")
        or result_row.get("construction_action_sha256")
        != construction.get("action_sha256")
        or int(result_row.get("final_timestep", -1))
        != int(construction.get("final_timestep", -2))
    ):
        raise ValueError(f"Invalid construction-gate record: {candidate_id}")
    return {
        "candidate_id": candidate_id,
        "layout": record["factors"]["layout"],
        "transit_locus": record["factors"]["transit_locus"],
        "support_stratum": record["factors"]["support_stratum"],
        "state_sha256": record["state_sha256"],
        "construction_action_sha256": construction["action_sha256"],
        "final_timestep": int(construction["final_timestep"]),
        "construction_action_count": int(construction["action_count"]),
        "initial_timestep_offset": int(record["initial_timestep_offset"]),
        "drawer_prefix_steps": int(
            construction["prefix"]["executed_action_count"]
        ),
        "acquisition_source_episode": int(
            acquisition["source_episode_index"]
        ),
        "acquisition_source_frame": int(
            acquisition["acquired_at_source_frame"]
        ),
        "acquisition_bridge_steps": int(bridge["executed_action_steps"]),
        "acquisition_source_steps": int(
            acquisition["executed_source_action_steps"]
        ),
        "acquisition_bridge_bowl_drift_m": float(bridge["bowl_drift_m"]),
        "safe_lift_steps": int(safe_lift["executed_action_steps"]),
        "safe_lift_budget": int(safe_lift["budgeted_action_steps"]),
        "safe_lift_max_relative_pose_deviation_m": float(
            safe_lift["max_grasp_relative_pose_deviation_m"]
        ),
        "padding_steps": int(padding["executed_action_steps"]),
        "support_distance": float(support["nearest"]["distance"]),
        "root_pass": True,
        "support_pass": True,
        "certificate_pass": True,
    }


def main() -> None:
    args = _parse_args()
    raw_dir = args.raw_dir.resolve()
    manifest = json.loads((raw_dir / "manifest.json").read_text())
    contract = json.loads((raw_dir / "contract.json").read_text())
    result = json.loads((raw_dir / "result.json").read_text())
    contract_sha = canonical_sha256(contract)
    if (
        manifest.get("status") != "complete"
        or manifest.get("all_pass") is not True
        or result.get("all_pass") is not True
        or manifest.get("contract_sha256") != contract_sha
        or result.get("contract_sha256") != contract_sha
        or manifest.get("policy_loaded") is not False
        or result.get("policy_loaded") is not False
        or int(result.get("proposal_oracles_executed", -1)) != 0
    ):
        raise ValueError("Raw construction gate is not complete and passing")
    expected_artifacts = manifest.get("artifact_sha256", {})
    observed_artifacts = {
        relative: _file_sha256(raw_dir / relative)
        for relative in expected_artifacts
    }
    if observed_artifacts != expected_artifacts:
        raise ValueError("Raw construction-gate artifact hashes changed")
    source_paths = {
        "runner": PROJECT / "scripts/run_phase3b_stage_a_construction_gate.py",
        "libero": PROJECT / "src/smolvla_analysis/phase3b_libero.py",
        "stage_a_runner": PROJECT / "scripts/run_phase3b_stage_a.py",
    }
    if contract.get("source_sha256") != {
        key: _file_sha256(path) for key, path in source_paths.items()
    }:
        raise ValueError("Construction-gate source files changed")

    candidate_ids = tuple(contract.get("candidate_ids", ()))
    result_rows = {
        row["candidate_id"]: row for row in result.get("conditions", ())
    }
    if (
        len(candidate_ids) != 6
        or len(set(candidate_ids)) != 6
        or set(result_rows) != set(candidate_ids)
        or int(manifest.get("candidate_count", -1)) != 6
        or int(manifest.get("expected_candidate_count", -1)) != 6
    ):
        raise ValueError("Construction-gate candidate inventory changed")
    rows = []
    for candidate_id in candidate_ids:
        record = json.loads(
            (raw_dir / "candidates" / f"{candidate_id}.json").read_text()
        )
        rows.append(_candidate_row(record, result_rows[candidate_id]))
    if len({row["state_sha256"] for row in rows}) != 6:
        raise ValueError("Construction gate contains duplicate root states")
    pair_audits = result.get("support_pair_geometry_audits", [])
    if len(pair_audits) != 3:
        raise ValueError("Construction gate has an incomplete pair audit")
    summary = {
        "schema_version": 1,
        "report_revision": "phase3b-stage-a-v36-construction-gate-report-v1",
        "run_id": result["run_id"],
        "contract_sha256": contract_sha,
        "all_pass": True,
        "candidate_count": len(rows),
        "unique_state_count": len({row["state_sha256"] for row in rows}),
        "root_final_timesteps": sorted(
            {row["final_timestep"] for row in rows}
        ),
        "initial_timestep_offsets": sorted(
            {row["initial_timestep_offset"] for row in rows}
        ),
        "support_reference_bank_sha256": result[
            "support_reference_bank_sha256"
        ],
        "proposal_oracles_executed": 0,
        "policy_loaded": False,
        "conditions": rows,
        "support_pair_geometry_audits": pair_audits,
        "scientific_boundary": (
            "This is a policy-free construction and state-certificate smoke. "
            "It establishes six feasible roots, not goal-proposal coverage, "
            "VLA behavior, or a causal hidden-state mechanism."
        ),
    }

    report_dir = (
        args.report_dir.resolve()
        if args.report_dir is not None
        else PROJECT / "reports/phase3b_stage_a" / raw_dir.name
    )
    report_root = (PROJECT / "reports/phase3b_stage_a").resolve()
    if report_dir == report_root or report_root not in report_dir.parents:
        raise ValueError(f"Compact report must remain under {report_root}")
    if report_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite compact report: {report_dir}"
        )
    staging = report_dir.with_name(
        f".{report_dir.name}__tmp__pid{os.getpid()}"
    )
    staging.mkdir(parents=True)
    atomic_write_json(staging / "contract.json", contract)
    atomic_write_json(staging / "summary.json", summary)
    pd.DataFrame(rows).to_csv(staging / "condition_summary.csv", index=False)
    pd.DataFrame(pair_audits).to_csv(
        staging / "support_pair_geometry.csv", index=False
    )
    (staging / "README.md").write_text(
        "# v36 phase-separated grasp construction gate\n\n"
        "All six previously untouched open/grasped roots pass construction, "
        "strict grasp continuity, joint-support measurement, root stability, "
        "and fresh repeated-action state certificates at simulator timestep "
        "560. Drawer opening and bowl acquisition are separately composed; "
        "the latter uses the frozen bowl-registered episode-474 pre-grasp "
        "continuation. No goal proposal or VLA policy was executed.\n\n"
        "This compact report is derived from hash-verified local raw evidence. "
        "It is a causal smoke gate for root construction only, not task-oracle "
        "coverage or hidden-state evidence.\n"
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
    print(f"Compact construction-gate report: {report_dir}")


if __name__ == "__main__":
    main()
