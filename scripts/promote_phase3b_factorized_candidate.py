#!/usr/bin/env python
"""Promote the final Stage A root without replaying a completed proposal tail."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from run_phase3b_stage_a import (
        PROJECT,
        _file_sha256,
        _load_config,
        _load_demos,
        _load_proposal_bank,
    )
    from run_phase3b_stage_a_completion import (
        _assert_construction_gate_identity,
        _load_construction_gate_binding,
        _validate_completion_config,
    )
except ModuleNotFoundError:
    from scripts.run_phase3b_stage_a import (
        PROJECT,
        _file_sha256,
        _load_config,
        _load_demos,
        _load_proposal_bank,
    )
    from scripts.run_phase3b_stage_a_completion import (
        _assert_construction_gate_identity,
        _load_construction_gate_binding,
        _validate_completion_config,
    )
from smolvla_analysis.phase3_crd import atomic_write_json
from smolvla_analysis.phase3b_completion import (
    oracle_pair_comparability,
    proposal_inventory,
    summarize_exhaustive_negative_checkpoint,
    validate_imported_checkpoint,
)
from smolvla_analysis.phase3b_consolidation import _validate_candidate_record
from smolvla_analysis.phase3b_feasibility import (
    build_factorized_feasibility_evidence,
    proposal_feasibility_evidence,
)
from smolvla_analysis.phase3b_libero import (
    build_landmark_registered_action_phase_proposal_bank,
    build_support_reference_bank,
    candidate_record,
    certify_computational_state,
    compact_snapshot_metadata,
    construct_candidate,
    make_stage_a_environment,
    run_goal_oracle_bank,
)
from smolvla_analysis.phase3b_persistence import (
    persist_candidate_artifacts_transactional,
    recover_candidate_transactions,
)
from smolvla_analysis.phase3b_registered_validation import (
    validate_oracle_proposal_ledger_compatible,
)
from smolvla_analysis.phase3b_stage_a import (
    GOALS,
    canonical_sha256,
    candidate_spec,
    snapshot_sha256,
    validate_selection_lock,
    validate_support_pair_geometry_records,
)


CANDIDATE_ID = (
    "stagea__drawer-open__possession-grasped__locus-cabinet-side__"
    "support-transverse-low-support__layout-a"
)
NEAR_CANDIDATE_ID = (
    "stagea__drawer-open__possession-grasped__locus-cabinet-side__"
    "support-demonstration-near__layout-a"
)
DEFAULT_CONFIG = PROJECT / "configs/phase3b_stage_a_v37.yaml"
DEFAULT_SOURCE_RUN = (
    PROJECT
    / "local/phase3b_stage_a/"
    "phase3b_stage_a_completion_v37_20260803T101257Z"
)
DEFAULT_FACTOR_CERTIFICATE = (
    PROJECT
    / "local/phase3b_stage_a/factorized_certificates/"
    "factorized_certificate_20260803T042126Z"
)
PROMOTION_REVISION = "phase3b-stage-a-v38-factorized-promotion-v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one additive promotion record from existing proposal ledgers "
            "and a factorized physical-feasibility certificate."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument(
        "--factor-certificate", type=Path, default=DEFAULT_FACTOR_CERTIFICATE
    )
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def _output_dir(requested: Path | None) -> Path:
    if requested is None:
        stamp = datetime.now(UTC).strftime(
            "phase3b_stage_a_promotion_v38_%Y%m%dT%H%M%SZ"
        )
        requested = PROJECT / "local/phase3b_stage_a" / stamp
    output = requested.resolve()
    root = (PROJECT / "local/phase3b_stage_a").resolve()
    if output == root or root not in output.parents:
        raise ValueError(f"Promotion output must remain under {root}")
    return output


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _checkpoint_path(source_run: Path, goal: str) -> Path:
    return source_run / "checkpoints" / f"{CANDIDATE_ID}__{goal}.json"


def _checkpoint_results(checkpoint: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(row["proposal_index"]): row["result"]
        for row in checkpoint["results"]
    }


def _validate_negative_checkpoint_identity(
    checkpoint: dict[str, Any],
    *,
    root_state_sha256: str,
    contract_sha256: str,
    selection_lock_sha256: str,
    proposals: tuple[Any, ...],
    phase_proposals: tuple[Any, ...],
) -> dict[str, Any]:
    expected = {
        "candidate_id": CANDIDATE_ID,
        "goal": "cabinet",
        "root_state_sha256": root_state_sha256,
        "contract_sha256": contract_sha256,
        "selection_lock_sha256": selection_lock_sha256,
        "proposal_inventory_sha256": canonical_sha256(
            proposal_inventory(proposals)
        ),
        "proposal_execution_contract_sha256": canonical_sha256(
            [proposal.metadata for proposal in phase_proposals]
        ),
    }
    for field, value in expected.items():
        if checkpoint.get(field) != value:
            raise ValueError(f"Negative cabinet checkpoint mismatch: {field}")
    summary = summarize_exhaustive_negative_checkpoint(
        checkpoint,
        candidate_id=CANDIDATE_ID,
        goal="cabinet",
        root_state_sha256=root_state_sha256,
        proposal_count=len(proposals),
    )
    if (
        checkpoint.get("status") != "in_progress"
        or checkpoint.get("last_completed_proposal_index")
        != len(proposals) - 1
    ):
        raise ValueError("Negative checkpoint completion semantics changed")
    return summary


def _factorized_evidence(certificate_dir: Path) -> dict[str, Any]:
    paths = {
        name: certificate_dir / name
        for name in ("contract.json", "manifest.json", "result.json", "acquisition.json")
    }
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError("Factorized certificate artifact is incomplete")
    return build_factorized_feasibility_evidence(
        result=_read_json(paths["result.json"]),
        manifest=_read_json(paths["manifest.json"]),
        contract=_read_json(paths["contract.json"]),
        artifact_file_sha256={
            name: _file_sha256(path) for name, path in paths.items()
        },
    )


def _promotion_contract(
    *,
    config_path: Path,
    source_run: Path,
    source_contract_sha256: str,
    source_selection_lock_sha256: str,
    checkpoint_paths: dict[str, Path],
    factor_certificate: Path,
    factorized_evidence: dict[str, Any],
) -> dict[str, Any]:
    implementation = (
        Path(__file__).resolve(),
        PROJECT / "src/smolvla_analysis/phase3b_feasibility.py",
        PROJECT / "src/smolvla_analysis/phase3b_persistence.py",
        PROJECT / "src/smolvla_analysis/phase3b_consolidation.py",
        PROJECT / "src/smolvla_analysis/phase3b_libero.py",
        PROJECT / "src/smolvla_analysis/phase3b_stage_a.py",
    )
    return {
        "schema_version": 1,
        "promotion_revision": PROMOTION_REVISION,
        "candidate_id": CANDIDATE_ID,
        "base_config_path": config_path.relative_to(PROJECT).as_posix(),
        "base_config_file_sha256": _file_sha256(config_path),
        "source_run": source_run.relative_to(PROJECT).as_posix(),
        "source_contract_sha256": source_contract_sha256,
        "source_selection_lock_sha256": source_selection_lock_sha256,
        "source_artifact_file_sha256": {
            "contract.json": _file_sha256(source_run / "contract.json"),
            "manifest.json": _file_sha256(source_run / "manifest.json"),
            "selection_lock.json": _file_sha256(
                source_run / "selection_lock.json"
            ),
            **{
                f"checkpoint_{goal}.json": _file_sha256(path)
                for goal, path in checkpoint_paths.items()
            },
        },
        "factor_certificate": factor_certificate.relative_to(PROJECT).as_posix(),
        "factorized_feasibility_evidence_sha256": canonical_sha256(
            factorized_evidence
        ),
        "implementation_sha256": {
            path.relative_to(PROJECT).as_posix(): _file_sha256(path)
            for path in implementation
        },
        "execution_scope": {
            "policy_forwards": 0,
            "completed_proposal_suffixes_reexecuted": 0,
            "new_proposal_suffixes_executed": 0,
            "normalization_only_reconstructions": 2,
            "physical_root_reconstructions": 1,
            "factorized_branches_reexecuted": 0,
        },
        "evidence_model": {
            "physical_feasibility": "factorized_policy_free_path_v1",
            "proposal_compatibility": "exhaustive_registered_bank_0_of_46",
            "claims_must_remain_separate": True,
        },
    }


def main() -> None:
    args = _parse_args()
    run_dir = _output_dir(args.run_dir)
    if run_dir.exists():
        recovered = recover_candidate_transactions(run_dir)
        if recovered:
            print(json.dumps({"recovered_transactions": recovered}, indent=2))
        manifest = _read_json(run_dir / "manifest.json")
        if manifest.get("status") == "complete":
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return
        raise FileExistsError(f"Refusing ambiguous promotion resume: {run_dir}")

    config_path = args.config.resolve()
    source_run = args.source_run.resolve()
    factor_certificate = args.factor_certificate.resolve()
    config = _load_config(config_path)
    completion = _validate_completion_config(config)
    source_contract = _read_json(source_run / "contract.json")
    source_manifest = _read_json(source_run / "manifest.json")
    source_lock = _read_json(source_run / "selection_lock.json")
    source_contract_sha = canonical_sha256(source_contract)
    if (
        source_manifest.get("contract_sha256") != source_contract_sha
        or source_manifest.get("selection_lock_sha256")
        != source_lock.get("selection_lock_sha256")
        or source_manifest.get("candidate_count") != 5
        or source_manifest.get("failed_candidate") != CANDIDATE_ID
        or source_manifest.get("policy_loaded") is not False
    ):
        raise ValueError("Source v37 failure inventory changed")
    validate_selection_lock(
        source_lock,
        contract_sha256=source_contract_sha,
        construction_revision=source_contract["construction_revision"],
    )
    if (
        source_contract.get("config_sha256") != _file_sha256(config_path)
        or source_contract.get("policy_loaded") is not False
    ):
        raise ValueError("Source v37 config or policy boundary changed")

    checkpoint_paths = {goal: _checkpoint_path(source_run, goal) for goal in GOALS}
    checkpoints = {goal: _read_json(path) for goal, path in checkpoint_paths.items()}
    roots = {checkpoint.get("root_state_sha256") for checkpoint in checkpoints.values()}
    if len(roots) != 1:
        raise ValueError("Source goal checkpoints use different physical roots")
    root_sha = next(iter(roots))
    factorized = _factorized_evidence(factor_certificate)
    if factorized["root_state_sha256"] != root_sha:
        raise ValueError("Factorized certificate uses another physical root")

    promotion_contract = _promotion_contract(
        config_path=config_path,
        source_run=source_run,
        source_contract_sha256=source_contract_sha,
        source_selection_lock_sha256=source_lock["selection_lock_sha256"],
        checkpoint_paths=checkpoint_paths,
        factor_certificate=factor_certificate,
        factorized_evidence=factorized,
    )
    promotion_sha = canonical_sha256(promotion_contract)
    run_dir.mkdir(parents=True)
    for name in ("candidates", "audits", "errors"):
        (run_dir / name).mkdir()
    atomic_write_json(run_dir / "contract.json", source_contract)
    atomic_write_json(run_dir / "selection_lock.json", source_lock)
    atomic_write_json(run_dir / "promotion_contract.json", promotion_contract)
    manifest = {
        "schema_version": 2,
        "run_id": run_dir.name,
        "stage": "phase3b_stage_a_factorized_promotion",
        "status": "in_progress",
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "contract_sha256": source_contract_sha,
        "selection_lock_sha256": source_lock["selection_lock_sha256"],
        "promotion_contract_sha256": promotion_sha,
        "policy_loaded": False,
        "candidate_count": 0,
        "expected_candidate_count": 1,
    }
    atomic_write_json(run_dir / "manifest.json", manifest)

    try:
        demos = _load_demos(config)
        proposal_banks = _load_proposal_bank(config)
        support_environment = make_stage_a_environment(PROJECT, run_dir, config)
        try:
            support_bank = build_support_reference_bank(
                support_environment,
                {goal: demos[goal] for goal in GOALS},
                config,
            )
        finally:
            support_environment.close()
        if support_bank.sha256 != source_manifest.get(
            "support_reference_bank_sha256"
        ):
            raise RuntimeError("Support reference bank changed")

        phase_banks = {}
        for goal in GOALS:
            phase_environment = make_stage_a_environment(PROJECT, run_dir, config)
            try:
                phase_banks[goal] = (
                    build_landmark_registered_action_phase_proposal_bank(
                        phase_environment,
                        target_layout="A",
                        proposals=proposal_banks[goal],
                        config=config,
                    )
                )
            finally:
                phase_environment.close()
        acquisition_episode = int(
            config["construction"][
                "registered_grasp_acquisition_episode_index"
            ]
        )
        acquisition_matches = [
            proposal
            for proposal in phase_banks["cabinet"]
            if proposal.source.episode_index == acquisition_episode
        ]
        if len(acquisition_matches) != 1:
            raise ValueError("Registered construction acquisition changed")

        drawer_completed = validate_imported_checkpoint(
            checkpoints["drawer"],
            candidate_id=CANDIDATE_ID,
            goal="drawer",
            root_state_sha256=root_sha,
            source_contract_sha256=source_contract_sha,
            source_selection_lock_sha256=source_lock[
                "selection_lock_sha256"
            ],
            proposals=proposal_banks["drawer"],
            phase_proposals=phase_banks["drawer"],
        )
        negative_summary = _validate_negative_checkpoint_identity(
            checkpoints["cabinet"],
            root_state_sha256=root_sha,
            contract_sha256=source_contract_sha,
            selection_lock_sha256=source_lock["selection_lock_sha256"],
            proposals=proposal_banks["cabinet"],
            phase_proposals=phase_banks["cabinet"],
        )

        spec = candidate_spec(CANDIDATE_ID)
        environment = make_stage_a_environment(PROJECT, run_dir, config)
        try:
            constructed = construct_candidate(
                environment,
                spec,
                demos,
                config,
                support_reference_bank=support_bank,
                registered_grasp_acquisition=acquisition_matches[0],
            )
            if snapshot_sha256(constructed.snapshot) != root_sha:
                raise RuntimeError("Promoted physical root changed")
            certificate = certify_computational_state(
                environment,
                constructed.snapshot,
                possession=spec.possession,
                probe_actions=config["certificate"]["actions"],
            )
            if certificate.get("pass") is not True:
                raise RuntimeError("Promoted computational-state certificate failed")
            gate = _load_construction_gate_binding(completion, config=config)
            gate_identity = _assert_construction_gate_identity(
                constructed,
                candidate_id=CANDIDATE_ID,
                gate=gate,
            )
            drawer_oracle = run_goal_oracle_bank(
                environment,
                constructed.snapshot,
                spec=spec,
                goal="drawer",
                proposals=proposal_banks["drawer"],
                initial_bowl_position=constructed.initial_bowl_position,
                initial_eef_position=constructed.initial_eef_position,
                initial_eef_orientation=constructed.initial_eef_orientation,
                initial_joint_positions=constructed.initial_joint_positions,
                recovery_waypoints=constructed.recovery_waypoints,
                config=config,
                action_phase_proposals=phase_banks["drawer"],
                completed_results=drawer_completed,
            )
            if canonical_sha256(drawer_oracle) != checkpoints["drawer"].get(
                "oracle_sha256"
            ):
                raise RuntimeError("Drawer oracle reassembly changed")
            cabinet_oracle = run_goal_oracle_bank(
                environment,
                constructed.snapshot,
                spec=spec,
                goal="cabinet",
                proposals=proposal_banks["cabinet"],
                initial_bowl_position=constructed.initial_bowl_position,
                initial_eef_position=constructed.initial_eef_position,
                initial_eef_orientation=constructed.initial_eef_orientation,
                initial_joint_positions=constructed.initial_joint_positions,
                recovery_waypoints=constructed.recovery_waypoints,
                config=config,
                action_phase_proposals=phase_banks["cabinet"],
                completed_results=_checkpoint_results(checkpoints["cabinet"]),
                allow_exhaustive_failure=True,
            )
        finally:
            environment.close()

        drawer_validation = validate_oracle_proposal_ledger_compatible(
            drawer_oracle,
            candidate_id=CANDIDATE_ID,
            goal="drawer",
        )
        cabinet_validation = validate_oracle_proposal_ledger_compatible(
            cabinet_oracle,
            candidate_id=CANDIDATE_ID,
            goal="cabinet",
            allow_exhaustive_failure=True,
        )
        if (
            drawer_validation["exhaustive_failure"]
            or not cabinet_validation["exhaustive_failure"]
            or cabinet_oracle["shared_normalized_state_sha256"]
            != factorized["normalized_state_sha256"]
            or cabinet_oracle["shared_normalization_action_sha256"]
            != factorized["normalization_action_sha256"]
        ):
            raise RuntimeError("Promotion evidence classes were conflated")

        record = candidate_record(
            spec=spec,
            constructed=constructed,
            certificate=certificate,
            oracles={"drawer": drawer_oracle, "cabinet": cabinet_oracle},
            contract_sha256=source_contract_sha,
            selection_lock_sha256=source_lock["selection_lock_sha256"],
            construction_revision=source_contract["construction_revision"],
        )
        record.update(
            {
                "schema_version": 2,
                "snapshot_metadata": compact_snapshot_metadata(
                    constructed.snapshot
                ),
                "completion_revision": PROMOTION_REVISION,
                "promotion_contract_sha256": promotion_sha,
                "construction_gate_identity": gate_identity,
                "goal_feasibility_evidence": {
                    "drawer": proposal_feasibility_evidence(
                        drawer_oracle, goal="drawer"
                    ),
                    "cabinet": factorized,
                },
                "proposal_compatibility_provenance": {
                    "drawer": {
                        "source_checkpoint_file_sha256": _file_sha256(
                            checkpoint_paths["drawer"]
                        ),
                        "source_checkpoint_oracle_sha256": checkpoints[
                            "drawer"
                        ]["oracle_sha256"],
                    },
                    "cabinet": {
                        **negative_summary,
                        "source_checkpoint_file_sha256": _file_sha256(
                            checkpoint_paths["cabinet"]
                        ),
                    },
                },
                "evidence_separation": {
                    "physical_feasibility_pass": True,
                    "proposal_compatibility_success_count": 0,
                    "proposal_compatibility_attempt_count": len(
                        proposal_banks["cabinet"]
                    ),
                    "competence_compatibility_gap": True,
                    "claims_must_remain_separate": True,
                },
            }
        )
        validation = _validate_candidate_record(record)
        near_record = _read_json(
            source_run / "candidates" / f"{NEAR_CANDIDATE_ID}.json"
        )
        geometry = validate_support_pair_geometry_records(
            near_record,
            record,
            max_realized_goal_distance_mismatch=float(
                config["validation"]["realized_goal_distance_mismatch_limit"]
            ),
            max_planned_recovery_distance_mismatch=float(
                config["validation"]["planned_recovery_distance_mismatch_limit"]
            ),
        )
        comparability = oracle_pair_comparability(near_record, record)
        if (
            comparability["by_goal"]["cabinet"][
                "proposal_outcome_estimable"
            ]
            is not True
            or comparability["by_goal"]["cabinet"]["estimable"] is not False
        ):
            raise RuntimeError("Cabinet outcome/cost estimands were not separated")

        persist_candidate_artifacts_transactional(
            run_dir,
            candidate_id=CANDIDATE_ID,
            snapshot=constructed.snapshot,
            record=record,
        )
        audit = {
            "schema_version": 1,
            "pass": True,
            "candidate_validation": validation,
            "support_pair_geometry": geometry,
            "oracle_comparability": comparability,
            "drawer_proposal_success": {
                "count": drawer_oracle["proposal_success_count"],
                "attempts": drawer_oracle["proposal_attempt_count"],
                "oracle_sha256": canonical_sha256(drawer_oracle),
            },
            "cabinet_proposal_compatibility": {
                "count": cabinet_oracle["proposal_success_count"],
                "attempts": cabinet_oracle["proposal_attempt_count"],
                "ledger_sha256": canonical_sha256(cabinet_oracle),
            },
            "cabinet_physical_feasibility": {
                "kind": factorized["kind"],
                "pass": factorized["pass"],
                "evidence_sha256": canonical_sha256(factorized),
            },
            "execution_scope": promotion_contract["execution_scope"],
        }
        atomic_write_json(run_dir / "audits" / "promotion_audit.json", audit)
        manifest.update(
            {
                "status": "complete",
                "updated_at": datetime.now(UTC).isoformat(),
                "completed_at": datetime.now(UTC).isoformat(),
                "candidate_count": 1,
                "candidate_id": CANDIDATE_ID,
                "state_sha256": record["state_sha256"],
                "candidate_record_sha256": canonical_sha256(record),
                "support_reference_bank_sha256": support_bank.sha256,
                "promotion_audit_sha256": canonical_sha256(audit),
                "physical_feasibility_pass": True,
                "cabinet_proposal_success_count": 0,
                "cabinet_proposal_attempt_count": len(
                    proposal_banks["cabinet"]
                ),
            }
        )
        atomic_write_json(run_dir / "manifest.json", manifest)
        print(json.dumps(audit, indent=2, sort_keys=True))
        print(f"Factorized Stage A promotion complete: {run_dir}")
    except Exception as exc:
        error = {
            "timestamp": datetime.now(UTC).isoformat(),
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_write_json(run_dir / "errors" / "promotion_error.json", error)
        manifest.update(
            {
                "status": "failed",
                "updated_at": datetime.now(UTC).isoformat(),
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
            }
        )
        atomic_write_json(run_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
