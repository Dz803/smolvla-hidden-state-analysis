#!/usr/bin/env python
"""Test acquisition-settling choices without executing any proposal oracle."""

from __future__ import annotations

import argparse
import json
import traceback
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

try:
    from run_phase3b_stage_a import (
        PROJECT,
        _file_sha256,
        _load_config,
        _load_demos,
    )
except ModuleNotFoundError:
    from scripts.run_phase3b_stage_a import (
        PROJECT,
        _file_sha256,
        _load_config,
        _load_demos,
    )
from smolvla_analysis.phase3_crd import atomic_write_json
from smolvla_analysis.phase3b_libero import (
    build_support_reference_bank,
    certify_computational_state,
    construct_candidate,
    make_stage_a_environment,
)
from smolvla_analysis.phase3b_stage_a import (
    GOALS,
    candidate_spec,
    canonical_sha256,
    snapshot_sha256,
)


DEFAULT_CANDIDATE = (
    "stagea__drawer-open__possession-grasped__locus-drawer-side__"
    "support-demonstration-near__layout-a"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose grasp settling before the Stage A root route."
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "configs/phase3b_stage_a_v35.yaml"
    )
    parser.add_argument("--candidate-id", default=DEFAULT_CANDIDATE)
    parser.add_argument("--stability-steps", type=int, action="append", required=True)
    parser.add_argument(
        "--max-translation-action", type=float, action="append", required=True
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if any(value < 1 for value in args.stability_steps):
        raise ValueError("Stability steps must be positive")
    if any(not 0.0 < value <= 1.0 for value in args.max_translation_action):
        raise ValueError("Translation actions must be in (0, 1]")
    config_path = args.config.resolve()
    config = _load_config(config_path)
    spec = candidate_spec(args.candidate_id)
    if spec.possession != "grasped":
        raise ValueError("Grasp-stability diagnostics require a grasped candidate")
    demos = _load_demos(config)
    run_id = datetime.now(UTC).strftime(
        "grasp_stability_%Y%m%dT%H%M%SZ"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else PROJECT
        / "local/phase3b_stage_a/construction_diagnostics"
        / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    contract = {
        "schema_version": 1,
        "diagnostic_revision": "phase3b-grasp-stability-v1",
        "candidate_id": spec.candidate_id,
        "base_config": config_path.relative_to(PROJECT).as_posix(),
        "base_config_sha256": _file_sha256(config_path),
        "stability_steps": args.stability_steps,
        "max_translation_actions": args.max_translation_action,
        "runner_sha256": _file_sha256(Path(__file__).resolve()),
        "policy_loaded": False,
        "proposal_oracles_executed": 0,
    }
    contract_sha = canonical_sha256(contract)
    atomic_write_json(output_dir / "contract.json", contract)

    bank_environment = make_stage_a_environment(PROJECT, output_dir, config)
    try:
        support_bank = build_support_reference_bank(
            bank_environment,
            {goal: demos[goal] for goal in GOALS},
            config,
        )
    finally:
        bank_environment.close()

    results = []
    for stability_steps in args.stability_steps:
        for max_translation_action in args.max_translation_action:
            condition_config = deepcopy(config)
            condition_config["construction"][
                "construction_grasp_stability_steps"
            ] = stability_steps
            condition_config["construction"][
                "max_translation_action_grasped"
            ] = max_translation_action
            environment = make_stage_a_environment(
                PROJECT, output_dir, condition_config
            )
            try:
                constructed = construct_candidate(
                    environment,
                    spec,
                    demos,
                    condition_config,
                    support_reference_bank=support_bank,
                )
                certificate = certify_computational_state(
                    environment,
                    constructed.snapshot,
                    possession=spec.possession,
                    probe_actions=condition_config["certificate"]["actions"],
                )
                phases = constructed.construction["root_servo"]["phases"]
                results.append(
                    {
                        "stability_steps": stability_steps,
                        "max_translation_action": max_translation_action,
                        "construction_pass": True,
                        "certificate_pass": bool(certificate["pass"]),
                        "state_sha256": snapshot_sha256(constructed.snapshot),
                        "prefix": constructed.construction["prefix"],
                        "root_phase_results": [
                            {
                                "phase": item["phase"],
                                "result": item["result"],
                            }
                            for item in phases
                        ],
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "stability_steps": stability_steps,
                        "max_translation_action": max_translation_action,
                        "construction_pass": False,
                        "certificate_pass": False,
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
            finally:
                environment.close()

    result = {
        "schema_version": 1,
        "run_id": run_id,
        "contract_sha256": contract_sha,
        "support_reference_bank_sha256": support_bank.sha256,
        "condition_count": len(results),
        "passing_condition_count": sum(
            item["construction_pass"] and item["certificate_pass"]
            for item in results
        ),
        "results": results,
        "policy_loaded": False,
        "proposal_oracles_executed": 0,
    }
    atomic_write_json(output_dir / "result.json", result)
    artifact_sha256 = {
        name: _file_sha256(output_dir / name)
        for name in ("contract.json", "result.json")
    }
    atomic_write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "status": "complete",
            "contract_sha256": contract_sha,
            "artifact_sha256": artifact_sha256,
            "policy_loaded": False,
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Grasp-stability diagnostic complete: {output_dir}")


if __name__ == "__main__":
    main()
