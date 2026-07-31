#!/usr/bin/env python
"""Certify the v36 grasped-root construction before any proposal rollout."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path

try:
    from run_phase3b_stage_a import (
        PROJECT,
        _file_sha256,
        _load_config,
        _load_demos,
        _load_proposal_bank,
    )
    from run_phase3b_stage_a_completion import _validate_completion_config
except ModuleNotFoundError:  # imported as scripts.run_phase3b_stage_a_construction_gate
    from scripts.run_phase3b_stage_a import (
        PROJECT,
        _file_sha256,
        _load_config,
        _load_demos,
        _load_proposal_bank,
    )
    from scripts.run_phase3b_stage_a_completion import (
        _validate_completion_config,
    )

from smolvla_analysis.phase3_crd import atomic_write_json
from smolvla_analysis.phase3b_libero import (
    build_landmark_registered_action_phase_proposal_bank,
    build_support_reference_bank,
    certify_computational_state,
    compact_snapshot_metadata,
    construct_candidate,
    make_stage_a_environment,
)
from smolvla_analysis.phase3b_stage_a import (
    GOALS,
    canonical_sha256,
    candidate_spec,
    snapshot_sha256,
    validate_support_pair_geometry_records,
)


DEFAULT_CONFIG = PROJECT / "configs/phase3b_stage_a_v36.yaml"
RAW_ROOT = PROJECT / "local/phase3b_stage_a/construction_gates"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct and certify every v36 completion root without running "
            "a goal proposal."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--candidate-id",
        action="append",
        help="Restrict a diagnostic gate to selected completion candidates.",
    )
    return parser.parse_args()


def _output_dir(requested: Path | None) -> Path:
    if requested is None:
        run_id = datetime.now(UTC).strftime(
            "phase3b_stage_a_construction_gate_%Y%m%dT%H%M%SZ"
        )
        requested = RAW_ROOT / run_id
    output = requested.resolve()
    raw_root = RAW_ROOT.resolve()
    if output == raw_root or raw_root not in output.parents:
        raise ValueError(f"Construction gate must remain under {raw_root}")
    return output


def _construction_contract_checks(construction: dict, config: dict) -> dict:
    acquisition = construction.get("grasp_acquisition") or {}
    safe_lift = construction.get("safe_lift") or {}
    expected_timestep = int(config["construction"]["root_final_timestep"])
    return {
        "registered_acquisition_mode": acquisition.get("mode")
        == "registered_cabinet_phase_until_stable_grasp_v1",
        "acquisition_final_grasped": acquisition.get("final_grasped") is True,
        "acquisition_final_goal_free": acquisition.get("final_goals")
        == {"drawer": False, "cabinet": False},
        "safe_lift_early_stop_enabled": safe_lift.get("padded_to_budget")
        is False,
        "safe_lift_executed_positive_steps": 0
        < int(safe_lift.get("executed_action_steps", 0)),
        "safe_lift_stopped_before_budget": int(
            safe_lift.get("executed_action_steps", 0)
        )
        < int(safe_lift.get("budgeted_action_steps", 0)),
        "normalized_final_timestep": int(
            construction.get("final_timestep", -1)
        )
        == expected_timestep,
        "nonempty_construction_action_trace": 0
        < int(construction.get("action_count", -1))
        <= expected_timestep,
    }


def _compact_candidate(spec, constructed, certificate, config) -> dict:
    construction = constructed.construction
    expected_timestep = int(config["construction"]["root_final_timestep"])
    construction_contract_checks = _construction_contract_checks(
        construction, config
    )
    construction_contract_pass = all(construction_contract_checks.values())
    return {
        "schema_version": 1,
        "candidate_id": spec.candidate_id,
        "factors": spec.as_dict(),
        "policy_loaded": False,
        "proposal_oracles_executed": 0,
        "pass": bool(
            constructed.root_validation.get("pass") is True
            and constructed.support_measurement is not None
            and constructed.support_measurement.get("pass") is True
            and certificate.get("pass") is True
            and construction_contract_pass
        ),
        "state_sha256": snapshot_sha256(constructed.snapshot),
        "snapshot_metadata": compact_snapshot_metadata(constructed.snapshot),
        "construction_contract_pass": construction_contract_pass,
        "construction_contract_checks": construction_contract_checks,
        "initial_timestep_offset": expected_timestep
        - int(construction.get("action_count", -1)),
        "construction": construction,
        "root_validation": constructed.root_validation,
        "root_geometry": constructed.root_geometry,
        "support_measurement": constructed.support_measurement,
        "certificate": certificate,
    }


def main() -> None:
    args = _parse_args()
    config_path = args.config.resolve()
    config = _load_config(config_path)
    completion = _validate_completion_config(config)
    candidate_ids = tuple(completion["candidate_ids"])
    if args.candidate_id:
        requested = set(args.candidate_id)
        unknown = requested - set(candidate_ids)
        if unknown:
            raise ValueError(
                f"Unknown construction-gate candidates: {sorted(unknown)}"
            )
        candidate_ids = tuple(
            candidate_id
            for candidate_id in candidate_ids
            if candidate_id in requested
        )
    specs = tuple(candidate_spec(candidate_id) for candidate_id in candidate_ids)
    if (
        not specs
        or any(
            spec.drawer_aperture != "open" or spec.possession != "grasped"
            for spec in specs
        )
        or config["construction"].get("open_grasped_acquisition_mode")
        != "registered_cabinet_phase_v1"
        or completion["imports"]
    ):
        raise ValueError(
            "Construction gate requires a fresh registered open-grasped shard"
        )

    output_dir = _output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "candidates").mkdir()
    demos = _load_demos(config)
    proposal_banks = _load_proposal_bank(config)
    acquisition_episode = int(
        config["construction"]["registered_grasp_acquisition_episode_index"]
    )
    acquisition_matches = [
        (index, proposal)
        for index, proposal in enumerate(proposal_banks["cabinet"])
        if proposal.episode_index == acquisition_episode
    ]
    if len(acquisition_matches) != 1:
        raise ValueError("Registered acquisition source identity is not unique")
    acquisition_inventory_index, acquisition_source = acquisition_matches[0]
    contract = {
        "schema_version": 1,
        "gate_revision": "phase3b-stage-a-v36-construction-gate-v1",
        "config_path": config_path.relative_to(PROJECT).as_posix(),
        "config_sha256": _file_sha256(config_path),
        "construction_revision": config["construction_revision"],
        "construction": config["construction"],
        "candidate_ids": list(candidate_ids),
        "acquisition_source": {
            "episode_index": acquisition_source.episode_index,
            "task_index": acquisition_source.task_index,
            "action_sha256": acquisition_source.action_sha256,
            "full_inventory_proposal_index": acquisition_inventory_index,
        },
        "phase_bank_setup": (
            "Build only the locked acquisition source; the landmark, anchor, "
            "orientation, suffix, and physical execution are identical to its "
            "entry in the full bank."
        ),
        "source_sha256": {
            "runner": _file_sha256(Path(__file__).resolve()),
            "libero": _file_sha256(
                PROJECT / "src/smolvla_analysis/phase3b_libero.py"
            ),
            "stage_a_runner": _file_sha256(
                PROJECT / "scripts/run_phase3b_stage_a.py"
            ),
        },
        "policy_loaded": False,
        "proposal_oracles_executed": 0,
    }
    contract_sha = canonical_sha256(contract)
    atomic_write_json(output_dir / "contract.json", contract)
    atomic_write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "run_id": output_dir.name,
            "status": "in_progress",
            "contract_sha256": contract_sha,
            "candidate_count": 0,
            "expected_candidate_count": len(candidate_ids),
            "policy_loaded": False,
        },
    )

    bank_environment = make_stage_a_environment(PROJECT, output_dir, config)
    try:
        support_bank = build_support_reference_bank(
            bank_environment,
            {goal: demos[goal] for goal in GOALS},
            config,
        )
    finally:
        bank_environment.close()
    print(f"support bank {support_bank.sha256}", flush=True)

    phase_banks = {}
    for layout in sorted({spec.layout for spec in specs}):
        phase_environment = make_stage_a_environment(
            PROJECT, output_dir, config
        )
        try:
            phase_banks[layout] = (
                build_landmark_registered_action_phase_proposal_bank(
                    phase_environment,
                    target_layout=layout,
                    proposals=(acquisition_source,),
                    config=config,
                )
            )
        finally:
            phase_environment.close()
        print(f"registered acquisition anchor {layout}", flush=True)

    results = []
    records = {}
    for spec in specs:
        matches = [
            proposal
            for proposal in phase_banks[spec.layout]
            if proposal.source.episode_index == acquisition_episode
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Registered acquisition proposal changed for {spec.candidate_id}"
            )
        environment = make_stage_a_environment(PROJECT, output_dir, config)
        try:
            constructed = construct_candidate(
                environment,
                spec,
                demos,
                config,
                support_reference_bank=support_bank,
                registered_grasp_acquisition=matches[0],
            )
            certificate = certify_computational_state(
                environment,
                constructed.snapshot,
                possession=spec.possession,
                probe_actions=config["certificate"]["actions"],
            )
            record = _compact_candidate(
                spec, constructed, certificate, config
            )
            atomic_write_json(
                output_dir / "candidates" / f"{spec.candidate_id}.json",
                record,
            )
            if record["pass"] is not True:
                failed_checks = sorted(
                    key
                    for key, value in record[
                        "construction_contract_checks"
                    ].items()
                    if not value
                )
                raise RuntimeError(
                    f"Construction gate failed for {spec.candidate_id}: "
                    f"failed_checks={failed_checks}, "
                    f"root_validation={record['root_validation'].get('pass')}, "
                    f"support={record['support_measurement'].get('pass')}, "
                    f"certificate={record['certificate'].get('pass')}"
                )
            records[spec.candidate_id] = record
            results.append(
                {
                    "candidate_id": spec.candidate_id,
                    "pass": True,
                    "state_sha256": record["state_sha256"],
                    "final_timestep": record["construction"]["final_timestep"],
                    "construction_action_sha256": record["construction"][
                        "action_sha256"
                    ],
                }
            )
            print(f"construction pass {spec.candidate_id}", flush=True)
        except Exception as exc:
            results.append(
                {
                    "candidate_id": spec.candidate_id,
                    "pass": False,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(
                f"construction fail {spec.candidate_id}: {exc}", flush=True
            )
        finally:
            environment.close()
        atomic_write_json(
            output_dir / "result.json",
            {
                "schema_version": 1,
                "run_id": output_dir.name,
                "contract_sha256": contract_sha,
                "conditions": results,
                "policy_loaded": False,
                "proposal_oracles_executed": 0,
            },
        )

    pair_audits = []
    for pair_id in sorted({spec.support_pair_id for spec in specs}):
        pair_specs = [spec for spec in specs if spec.support_pair_id == pair_id]
        if len(pair_specs) != 2 or any(
            spec.candidate_id not in records for spec in pair_specs
        ):
            continue
        near = next(
            records[spec.candidate_id]
            for spec in pair_specs
            if spec.support_stratum == "demonstration_near"
        )
        low = next(
            records[spec.candidate_id]
            for spec in pair_specs
            if spec.support_stratum == "transverse_low_support"
        )
        audit = validate_support_pair_geometry_records(
            near,
            low,
            max_realized_goal_distance_mismatch=float(
                config["validation"][
                    "realized_goal_distance_mismatch_limit"
                ]
            ),
            max_planned_recovery_distance_mismatch=float(
                config["validation"][
                    "planned_recovery_distance_mismatch_limit"
                ]
            ),
        )
        pair_audits.append(audit)

    expected_pair_count = sum(
        len(
            {
                spec.support_stratum
                for spec in specs
                if spec.support_pair_id == pair_id
            }
        )
        == 2
        for pair_id in {spec.support_pair_id for spec in specs}
    )
    all_pass = bool(
        len(results) == len(candidate_ids)
        and all(item["pass"] for item in results)
        and len(records) == len(candidate_ids)
        and len(pair_audits) == expected_pair_count
        and len({record["state_sha256"] for record in records.values()})
        == len(candidate_ids)
    )
    result = {
        "schema_version": 1,
        "run_id": output_dir.name,
        "contract_sha256": contract_sha,
        "all_pass": all_pass,
        "conditions": results,
        "support_pair_geometry_audits": pair_audits,
        "support_reference_bank_sha256": support_bank.sha256,
        "policy_loaded": False,
        "proposal_oracles_executed": 0,
    }
    atomic_write_json(output_dir / "result.json", result)
    artifact_sha256 = {
        "contract.json": _file_sha256(output_dir / "contract.json"),
        "result.json": _file_sha256(output_dir / "result.json"),
        **{
            f"candidates/{candidate_id}.json": _file_sha256(
                output_dir / "candidates" / f"{candidate_id}.json"
            )
            for candidate_id in records
        },
    }
    atomic_write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "run_id": output_dir.name,
            "status": "complete" if all_pass else "failed",
            "all_pass": all_pass,
            "contract_sha256": contract_sha,
            "candidate_count": len(records),
            "expected_candidate_count": len(candidate_ids),
            "artifact_sha256": artifact_sha256,
            "policy_loaded": False,
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Construction gate finished: {output_dir}")
    if not all_pass:
        raise RuntimeError("Stage A v36 construction gate did not pass")


if __name__ == "__main__":
    main()
