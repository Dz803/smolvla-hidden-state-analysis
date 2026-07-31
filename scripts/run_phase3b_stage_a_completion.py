#!/usr/bin/env python
"""Complete the 13 missing Stage A roots without replaying finished evidence."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

try:
    from run_phase3b_stage_a import (
        PROJECT,
        _completed_record,
        _contract,
        _file_sha256,
        _load_config,
        _load_demos,
        _load_proposal_bank,
        _minimum_native_horizon,
        _persist_candidate_state,
        _validation_limits,
    )
except ModuleNotFoundError:  # imported as scripts.run_phase3b_stage_a_completion
    from scripts.run_phase3b_stage_a import (
        PROJECT,
        _completed_record,
        _contract,
        _file_sha256,
        _load_config,
        _load_demos,
        _load_proposal_bank,
        _minimum_native_horizon,
        _persist_candidate_state,
        _validation_limits,
    )
from smolvla_analysis.phase3_crd import atomic_write_json
from smolvla_analysis.phase3b_completion import (
    COMPLETION_REVISION,
    oracle_pair_comparability,
    proposal_inventory,
    summarize_exhaustive_negative_checkpoint,
    validate_completion_candidate_ids,
    validate_imported_checkpoint,
)
from smolvla_analysis.phase3b_libero import (
    build_action_phase_proposal_bank,
    build_landmark_registered_action_phase_proposal_bank,
    build_support_reference_bank,
    candidate_record,
    certify_computational_state,
    compact_snapshot_metadata,
    construct_candidate,
    make_stage_a_environment,
    run_goal_oracle_bank,
)
from smolvla_analysis.phase3b_stage_a import (
    GOALS,
    build_selection_lock,
    canonical_sha256,
    candidate_spec,
    iter_candidate_specs,
    snapshot_sha256,
    validate_selection_lock,
    validate_support_pair_geometry_records,
    validate_support_pair_records,
)


DEFAULT_CONFIG = PROJECT / "configs/phase3b_stage_a_v35.yaml"
DEFAULT_REGISTERED_SMOKE = (
    PROJECT
    / "local/phase3b_stage_a/registered_generalization_smokes/"
    "registered_smoke_20260731T073347Z"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the additive v35 Stage A completion shard."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--registered-smoke", type=Path, default=DEFAULT_REGISTERED_SMOKE
    )
    parser.add_argument("--candidate-id", action="append")
    parser.add_argument("--stop-after-candidates", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def _run_dir(requested: Path | None) -> Path:
    if requested is None:
        run_id = datetime.now(UTC).strftime(
            "phase3b_stage_a_completion_%Y%m%dT%H%M%SZ"
        )
        requested = PROJECT / "local/phase3b_stage_a" / run_id
    output = requested.resolve()
    raw_root = (PROJECT / "local/phase3b_stage_a").resolve()
    if output == raw_root or raw_root not in output.parents:
        raise ValueError(f"Completion shard must remain under {raw_root}")
    return output


def _validate_completion_config(config: dict[str, Any]) -> dict[str, Any]:
    completion = config.get("completion")
    if not isinstance(completion, dict):
        raise ValueError("v35 config has no completion block")
    expected_count = int(completion.get("expected_candidate_count", -1))
    candidate_ids = validate_completion_candidate_ids(
        completion.get("candidate_ids", ()), expected_count=expected_count
    )
    imports = completion.get("imports")
    if not isinstance(imports, list) or len(imports) != 1:
        raise ValueError("v35 completion requires exactly one prior goal import")
    imported = imports[0]
    required = {
        "candidate_id",
        "goal",
        "source_run",
        "source_config",
        "preserve_negative_goal",
    }
    if set(imported) != required:
        raise ValueError("v35 completion import contract changed")
    if (
        imported["candidate_id"] not in candidate_ids
        or imported["goal"] != "drawer"
        or imported["preserve_negative_goal"] != "cabinet"
    ):
        raise ValueError("v35 completion import identity changed")
    return {**completion, "candidate_ids": candidate_ids}


def _completion_contract(
    config_path: Path,
    config: dict[str, Any],
    demos: dict[str, Any],
    proposal_banks: dict[str, tuple[Any, ...]],
    completion: dict[str, Any],
    smoke_dir: Path,
) -> dict[str, Any]:
    base = _contract(config_path, config, demos, proposal_banks)
    smoke_files = {
        name: _file_sha256(smoke_dir / name)
        for name in ("contract.json", "result.json", "manifest.json")
    }
    return {
        **base,
        "stage": "phase3b_stage_a_completion",
        "completion_revision": COMPLETION_REVISION,
        "completion_assignment": {
            "expected_candidate_count": completion["expected_candidate_count"],
            "candidate_ids": list(completion["candidate_ids"]),
            "imports": completion["imports"],
        },
        "registered_smoke": {
            "run_dir": smoke_dir.relative_to(PROJECT).as_posix(),
            "artifact_sha256": smoke_files,
        },
        "completion_source_sha256": {
            "runner": _file_sha256(Path(__file__).resolve()),
            "completion": _file_sha256(
                PROJECT / "src/smolvla_analysis/phase3b_completion.py"
            ),
        },
        "reuse_policy": (
            "Import only exact root/contract/proposal-bound results; never "
            "re-execute an imported proposal attempt."
        ),
    }


def _initialize(
    run_dir: Path,
    contract: dict[str, Any],
    selection_lock: dict[str, Any],
    *,
    expected_count: int,
) -> dict[str, Any]:
    contract_sha = canonical_sha256(contract)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if json.loads((run_dir / "contract.json").read_text()) != contract:
            raise ValueError("Completion contract changed on resume")
        expected = {
            "contract_sha256": contract_sha,
            "selection_lock_sha256": selection_lock["selection_lock_sha256"],
            "expected_candidate_count": expected_count,
            "policy_loaded": False,
        }
        for field, value in expected.items():
            if manifest.get(field) != value:
                raise ValueError(f"Completion manifest mismatch: {field}")
        validate_selection_lock(
            json.loads((run_dir / "selection_lock.json").read_text()),
            contract_sha256=contract_sha,
            construction_revision=contract["construction_revision"],
        )
        return manifest
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("candidates", "states.zarr", "checkpoints", "errors", "audits"):
        (run_dir / name).mkdir()
    atomic_write_json(run_dir / "contract.json", contract)
    atomic_write_json(run_dir / "selection_lock.json", selection_lock)
    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "stage": "phase3b_stage_a_completion",
        "status": "in_progress",
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "contract_sha256": contract_sha,
        "selection_lock_sha256": selection_lock["selection_lock_sha256"],
        "candidate_count": 0,
        "expected_candidate_count": expected_count,
        "policy_loaded": False,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def _update_manifest(run_dir: Path, manifest: dict[str, Any], **updates: Any) -> None:
    manifest.update(updates)
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    atomic_write_json(run_dir / "manifest.json", manifest)


def _record_error(run_dir: Path, candidate_id: str, exc: Exception) -> None:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    atomic_write_json(
        run_dir / "errors" / f"{candidate_id}__{stamp}__pid{os.getpid()}.json",
        {
            "candidate_id": candidate_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    )


def _load_registered_smoke(
    smoke_dir: Path,
    *,
    base_contract_sha256: str,
) -> dict[str, dict[str, Any]]:
    smoke_dir = smoke_dir.resolve()
    contract_path = smoke_dir / "contract.json"
    result_path = smoke_dir / "result.json"
    manifest_path = smoke_dir / "manifest.json"
    contract = json.loads(contract_path.read_text())
    result = json.loads(result_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    contract_sha = canonical_sha256(contract)
    if (
        manifest.get("status") != "complete"
        or manifest.get("all_pass") is not True
        or result.get("all_pass") is not True
        or {manifest.get("contract_sha256"), result.get("contract_sha256")}
        != {contract_sha}
    ):
        raise ValueError("Registered smoke is not complete and passing")
    hashes = {
        "contract.json": _file_sha256(contract_path),
        "result.json": _file_sha256(result_path),
    }
    if manifest.get("artifact_sha256") != hashes:
        raise ValueError("Registered smoke artifact hashes changed")
    if contract.get("v35_stage_a_contract_sha256") != base_contract_sha256:
        raise ValueError("Registered smoke used a different v35 base contract")
    conditions = {
        item["candidate_id"]: item for item in result["conditions"]
    }
    if len(conditions) != 2 or not all(item["pass"] for item in conditions.values()):
        raise ValueError("Registered smoke condition inventory changed")
    return conditions


def _checkpoint(
    run_dir: Path,
    *,
    candidate_id: str,
    goal: str,
    root_state_sha256: str,
    contract_sha256: str,
    selection_lock_sha256: str,
    proposals: tuple[Any, ...],
    phase_proposals: tuple[Any, ...],
    imported_results: dict[int, dict[str, Any]],
    imported_provenance: dict[int, dict[str, Any]],
) -> tuple[
    dict[int, dict[str, Any]],
    Callable[[int, dict[str, Any]], None],
    Callable[[dict[str, Any]], None],
]:
    path = run_dir / "checkpoints" / f"{candidate_id}__{goal}.json"
    expected = {
        "schema_version": 1,
        "checkpoint_revision": "phase3b-v35-sparse-import-v1",
        "candidate_id": candidate_id,
        "goal": goal,
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
    if path.is_file():
        state = json.loads(path.read_text())
        for field, value in expected.items():
            if state.get(field) != value:
                raise ValueError(
                    f"Completion checkpoint mismatch for {candidate_id}/{goal}: {field}"
                )
    else:
        state = {
            **expected,
            "status": "in_progress",
            "result_count": 0,
            "results": [],
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
        }
    rows = state.get("results")
    if not isinstance(rows, list):
        raise ValueError("Completion checkpoint result rows are invalid")
    by_index = {int(row["proposal_index"]): row for row in rows}
    if len(by_index) != len(rows) or any(
        index not in range(len(proposals)) for index in by_index
    ):
        raise ValueError("Completion checkpoint indices are invalid")
    for index, result in imported_results.items():
        if index not in range(len(proposals)):
            raise ValueError("Imported smoke proposal index is invalid")
        expected_row = {
            "proposal_index": index,
            "result": result,
            "provenance": imported_provenance[index],
        }
        if index in by_index and by_index[index] != expected_row:
            raise ValueError("Imported smoke checkpoint row changed")
        by_index[index] = expected_row
    state["results"] = [by_index[index] for index in sorted(by_index)]
    state["result_count"] = len(state["results"])
    state["updated_at"] = datetime.now(UTC).isoformat()
    atomic_write_json(path, state)

    def record(index: int, result: dict[str, Any]) -> None:
        if index in by_index:
            raise ValueError(
                f"Refusing to rerun completed {candidate_id}/{goal}/{index}"
            )
        by_index[index] = {
            "proposal_index": index,
            "result": result,
            "provenance": {"kind": "simulated_in_completion_shard"},
        }
        state["results"] = [by_index[key] for key in sorted(by_index)]
        state["result_count"] = len(state["results"])
        state["last_completed_proposal_index"] = index
        state["updated_at"] = datetime.now(UTC).isoformat()
        atomic_write_json(path, state)

    def finish(oracle: dict[str, Any]) -> None:
        if set(by_index) != set(range(len(proposals))):
            raise ValueError(f"Cannot finish sparse {candidate_id}/{goal} checkpoint")
        state["status"] = "complete"
        state["oracle_sha256"] = canonical_sha256(oracle)
        state["updated_at"] = datetime.now(UTC).isoformat()
        atomic_write_json(path, state)

    return (
        {index: row["result"] for index, row in by_index.items()},
        record,
        finish,
    )


def _source_import(
    imported: dict[str, Any],
    *,
    environment,
    runtime_dir: Path,
    constructed: Any,
    proposals: tuple[Any, ...],
    spec: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_run = (PROJECT / imported["source_run"]).resolve()
    source_config_path = (PROJECT / imported["source_config"]).resolve()
    source_config = _load_config(source_config_path)
    source_contract_path = source_run / "contract.json"
    source_manifest_path = source_run / "manifest.json"
    source_lock_path = source_run / "selection_lock.json"
    source_contract = json.loads(source_contract_path.read_text())
    source_manifest = json.loads(source_manifest_path.read_text())
    source_lock = json.loads(source_lock_path.read_text())
    source_contract_sha = canonical_sha256(source_contract)
    if (
        source_manifest.get("contract_sha256") != source_contract_sha
        or source_contract.get("config_sha256") != _file_sha256(source_config_path)
    ):
        raise ValueError("Imported source run contract changed")
    validate_selection_lock(
        source_lock,
        contract_sha256=source_contract_sha,
        construction_revision=source_contract["construction_revision"],
    )
    if source_manifest.get("selection_lock_sha256") != source_lock.get(
        "selection_lock_sha256"
    ):
        raise ValueError("Imported source selection lock changed")
    phase_environment = make_stage_a_environment(
        PROJECT, runtime_dir, source_config
    )
    try:
        source_phases = build_action_phase_proposal_bank(
            phase_environment,
            layout=spec.layout,
            proposals=proposals,
            config=source_config,
        )
    finally:
        phase_environment.close()
    goal = imported["goal"]
    checkpoint_path = (
        source_run / "checkpoints" / f"{spec.candidate_id}__{goal}.json"
    )
    checkpoint = json.loads(checkpoint_path.read_text())
    completed = validate_imported_checkpoint(
        checkpoint,
        candidate_id=spec.candidate_id,
        goal=goal,
        root_state_sha256=snapshot_sha256(constructed.snapshot),
        source_contract_sha256=source_contract_sha,
        source_selection_lock_sha256=source_lock["selection_lock_sha256"],
        proposals=proposals,
        phase_proposals=source_phases,
    )
    oracle = run_goal_oracle_bank(
        environment,
        constructed.snapshot,
        spec=spec,
        goal=goal,
        proposals=proposals,
        initial_bowl_position=constructed.initial_bowl_position,
        initial_eef_position=constructed.initial_eef_position,
        initial_eef_orientation=constructed.initial_eef_orientation,
        initial_joint_positions=constructed.initial_joint_positions,
        recovery_waypoints=constructed.recovery_waypoints,
        config=source_config,
        action_phase_proposals=source_phases,
        completed_results=completed,
    )
    reassembled_sha = canonical_sha256(oracle)
    if reassembled_sha != checkpoint["oracle_sha256"]:
        raise ValueError("Imported source oracle does not reassemble exactly")
    oracle["import_provenance"] = {
        "kind": "complete_goal_checkpoint_reassembly_no_proposal_simulation",
        "source_run": source_run.relative_to(PROJECT).as_posix(),
        "source_contract_sha256": source_contract_sha,
        "source_checkpoint_file_sha256": _file_sha256(checkpoint_path),
        "source_oracle_sha256": checkpoint["oracle_sha256"],
        "reassembled_oracle_sha256": reassembled_sha,
        "proposal_attempts_reexecuted": 0,
    }

    negative_goal = imported["preserve_negative_goal"]
    negative_path = (
        source_run
        / "checkpoints"
        / f"{spec.candidate_id}__{negative_goal}.json"
    )
    negative = json.loads(negative_path.read_text())
    negative_phases = build_action_phase_proposal_bank(
        environment,
        layout=spec.layout,
        proposals=_load_proposal_bank(source_config)[negative_goal],
        config=source_config,
    )
    negative_proposals = _load_proposal_bank(source_config)[negative_goal]
    expected_negative = {
        "contract_sha256": source_contract_sha,
        "selection_lock_sha256": source_lock["selection_lock_sha256"],
        "proposal_inventory_sha256": canonical_sha256(
            proposal_inventory(negative_proposals)
        ),
        "proposal_execution_contract_sha256": canonical_sha256(
            [proposal.metadata for proposal in negative_phases]
        ),
    }
    if any(negative.get(key) != value for key, value in expected_negative.items()):
        raise ValueError("Imported negative checkpoint binding changed")
    negative_summary = summarize_exhaustive_negative_checkpoint(
        negative,
        candidate_id=spec.candidate_id,
        goal=negative_goal,
        root_state_sha256=snapshot_sha256(constructed.snapshot),
        proposal_count=len(negative_proposals),
    )
    negative_summary.update(
        {
            "source_run": source_run.relative_to(PROJECT).as_posix(),
            "source_contract_sha256": source_contract_sha,
            "source_checkpoint_file_sha256": _file_sha256(negative_path),
            "proposal_execution_mode": source_config["action_phase_oracle"][
                "execution_mode"
            ],
        }
    )
    return oracle, negative_summary


def _smoke_seed(
    smoke_conditions: dict[str, dict[str, Any]],
    *,
    spec: Any,
    constructed: Any,
    proposal: Any,
    phase_proposal: Any,
    smoke_dir: Path,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    condition = smoke_conditions.get(spec.candidate_id)
    if condition is None:
        return {}, {}
    index = int(condition["proposal_index"])
    attempt = condition["attempt"]
    if (
        condition["root_state_sha256"] != snapshot_sha256(constructed.snapshot)
        or index != int(phase_proposal.metadata["proposal_index"])
        or attempt.get("demo_episode_index") != proposal.episode_index
        or attempt.get("demo_task_index") != proposal.task_index
        or attempt.get("demo_action_sha256") != proposal.action_sha256
        or attempt.get("phase_proposal") != phase_proposal.metadata
    ):
        raise ValueError("Registered-smoke attempt cannot be imported exactly")
    return (
        {index: attempt},
        {
            index: {
                "kind": "registered_held_root_smoke_import",
                "source_run": smoke_dir.relative_to(PROJECT).as_posix(),
                "source_result_file_sha256": _file_sha256(
                    smoke_dir / "result.json"
                ),
                "proposal_attempts_reexecuted": 0,
            }
        },
    )


def main() -> None:
    args = _parse_args()
    if args.stop_after_candidates is not None and args.stop_after_candidates < 1:
        raise ValueError("--stop-after-candidates must be positive")
    config_path = args.config.resolve()
    smoke_dir = args.registered_smoke.resolve()
    config = _load_config(config_path)
    completion = _validate_completion_config(config)
    demos = _load_demos(config)
    proposal_banks = _load_proposal_bank(config)
    minimum_horizon = _minimum_native_horizon(config, demos, proposal_banks)
    if int(config["environment"]["episode_length"]) < minimum_horizon:
        raise ValueError("v35 completion native horizon is too short")
    base_contract_sha = canonical_sha256(
        _contract(config_path, config, demos, proposal_banks)
    )
    smoke_conditions = _load_registered_smoke(
        smoke_dir, base_contract_sha256=base_contract_sha
    )
    contract = _completion_contract(
        config_path,
        config,
        demos,
        proposal_banks,
        completion,
        smoke_dir,
    )
    contract_sha = canonical_sha256(contract)
    selection_lock = build_selection_lock(
        contract_sha256=contract_sha,
        construction_revision=config["construction_revision"],
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "completion_revision": COMPLETION_REVISION,
                    "contract_sha256": contract_sha,
                    "selection_lock_sha256": selection_lock[
                        "selection_lock_sha256"
                    ],
                    "expected_candidate_count": completion[
                        "expected_candidate_count"
                    ],
                    "candidate_ids": list(completion["candidate_ids"]),
                    "minimum_native_horizon": minimum_horizon,
                    "configured_native_horizon": config["environment"][
                        "episode_length"
                    ],
                    "registered_smoke_candidate_ids": sorted(smoke_conditions),
                    "prior_goal_imports": completion["imports"],
                    "policy_loaded": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    run_dir = _run_dir(args.run_dir)
    manifest = _initialize(
        run_dir,
        contract,
        selection_lock,
        expected_count=int(completion["expected_candidate_count"]),
    )
    selected = tuple(completion["candidate_ids"])
    if args.candidate_id:
        requested = set(args.candidate_id)
        unknown = requested - set(selected)
        if unknown:
            raise ValueError(f"Unknown completion candidates: {sorted(unknown)}")
        selected = tuple(item for item in selected if item in requested)
    completed_count = sum(
        _completed_record(
            run_dir,
            candidate_id,
            contract_sha256=contract_sha,
            selection_lock_sha256=selection_lock["selection_lock_sha256"],
        )
        is not None
        for candidate_id in completion["candidate_ids"]
    )
    support_bank = None
    phase_banks: dict[tuple[str, str], tuple[Any, ...]] = {}
    import_spec = completion["imports"][0]
    for candidate_id in selected:
        existing = _completed_record(
            run_dir,
            candidate_id,
            contract_sha256=contract_sha,
            selection_lock_sha256=selection_lock["selection_lock_sha256"],
        )
        if existing is not None:
            print(f"skip complete {candidate_id}", flush=True)
            continue
        if (
            args.stop_after_candidates is not None
            and completed_count >= args.stop_after_candidates
        ):
            break
        spec = candidate_spec(candidate_id)
        if support_bank is None:
            bank_environment = make_stage_a_environment(PROJECT, run_dir, config)
            try:
                support_bank = build_support_reference_bank(
                    bank_environment,
                    {goal: demos[goal] for goal in GOALS},
                    config,
                )
            finally:
                bank_environment.close()
            print(f"support bank {support_bank.sha256}", flush=True)
        for goal in GOALS:
            key = (goal, spec.layout)
            if key not in phase_banks:
                phase_environment = make_stage_a_environment(
                    PROJECT, run_dir, config
                )
                try:
                    phase_banks[key] = (
                        build_landmark_registered_action_phase_proposal_bank(
                            phase_environment,
                            target_layout=spec.layout,
                            proposals=proposal_banks[goal],
                            config=config,
                        )
                    )
                finally:
                    phase_environment.close()
        print(f"construct {candidate_id}", flush=True)
        environment = make_stage_a_environment(PROJECT, run_dir, config)
        try:
            constructed = construct_candidate(
                environment,
                spec,
                demos,
                config,
                support_reference_bank=support_bank,
            )
            certificate = certify_computational_state(
                environment,
                constructed.snapshot,
                possession=spec.possession,
                probe_actions=config["certificate"]["actions"],
            )
            if certificate["pass"] is not True:
                raise RuntimeError(f"v35 certificate failed for {candidate_id}")
            oracles = {}
            imports = {}
            prior_negative = {}
            for goal in GOALS:
                if candidate_id == import_spec["candidate_id"] and goal == import_spec[
                    "goal"
                ]:
                    oracle, negative = _source_import(
                        import_spec,
                        environment=environment,
                        runtime_dir=run_dir,
                        constructed=constructed,
                        proposals=proposal_banks[goal],
                        spec=spec,
                    )
                    oracles[goal] = oracle
                    imports[goal] = oracle["import_provenance"]
                    prior_negative[import_spec["preserve_negative_goal"]] = negative
                    print(f"import complete {candidate_id}/{goal}", flush=True)
                    continue
                phases = phase_banks[(goal, spec.layout)]
                smoke_results: dict[int, dict[str, Any]] = {}
                smoke_provenance: dict[int, dict[str, Any]] = {}
                if goal == "cabinet" and candidate_id in smoke_conditions:
                    smoke_index = int(
                        smoke_conditions[candidate_id]["proposal_index"]
                    )
                    smoke_results, smoke_provenance = _smoke_seed(
                        smoke_conditions,
                        spec=spec,
                        constructed=constructed,
                        proposal=proposal_banks[goal][smoke_index],
                        phase_proposal=phases[smoke_index],
                        smoke_dir=smoke_dir,
                    )
                    imports[goal] = smoke_provenance[smoke_index]
                completed, record_result, finish = _checkpoint(
                    run_dir,
                    candidate_id=candidate_id,
                    goal=goal,
                    root_state_sha256=snapshot_sha256(constructed.snapshot),
                    contract_sha256=contract_sha,
                    selection_lock_sha256=selection_lock[
                        "selection_lock_sha256"
                    ],
                    proposals=proposal_banks[goal],
                    phase_proposals=phases,
                    imported_results=smoke_results,
                    imported_provenance=smoke_provenance,
                )
                oracles[goal] = run_goal_oracle_bank(
                    environment,
                    constructed.snapshot,
                    spec=spec,
                    goal=goal,
                    proposals=proposal_banks[goal],
                    initial_bowl_position=constructed.initial_bowl_position,
                    initial_eef_position=constructed.initial_eef_position,
                    initial_eef_orientation=constructed.initial_eef_orientation,
                    initial_joint_positions=constructed.initial_joint_positions,
                    recovery_waypoints=constructed.recovery_waypoints,
                    config=config,
                    action_phase_proposals=phases,
                    completed_results=completed,
                    result_callback=record_result,
                )
                finish(oracles[goal])
                print(
                    f"complete {candidate_id}/{goal}: "
                    f"{oracles[goal]['proposal_success_count']}/"
                    f"{oracles[goal]['proposal_attempt_count']}",
                    flush=True,
                )
            record = candidate_record(
                spec=spec,
                constructed=constructed,
                certificate=certificate,
                oracles=oracles,
                contract_sha256=contract_sha,
                selection_lock_sha256=selection_lock["selection_lock_sha256"],
                construction_revision=config["construction_revision"],
            )
            record["snapshot_metadata"] = compact_snapshot_metadata(
                constructed.snapshot
            )
            record["completion_revision"] = COMPLETION_REVISION
            record["oracle_imports"] = imports
            record["prior_negative_oracle_evidence"] = prior_negative
            counterpart_spec = next(
                item
                for item in iter_candidate_specs()
                if item.support_pair_id == spec.support_pair_id
                and item.support_stratum != spec.support_stratum
            )
            counterpart = _completed_record(
                run_dir,
                counterpart_spec.candidate_id,
                contract_sha256=contract_sha,
                selection_lock_sha256=selection_lock["selection_lock_sha256"],
            )
            if counterpart is not None:
                near, low = (
                    (record, counterpart)
                    if spec.support_stratum == "demonstration_near"
                    else (counterpart, record)
                )
                limits = _validation_limits(config)
                geometry = validate_support_pair_geometry_records(
                    near,
                    low,
                    max_realized_goal_distance_mismatch=limits[
                        "max_realized_goal_distance_mismatch"
                    ],
                    max_planned_recovery_distance_mismatch=limits[
                        "max_planned_recovery_distance_mismatch"
                    ],
                )
                comparability = oracle_pair_comparability(near, low)
                strict = None
                if comparability["all_goals_estimable"]:
                    strict = validate_support_pair_records(
                        near, low, **limits
                    )
                atomic_write_json(
                    run_dir
                    / "audits"
                    / f"{spec.support_pair_id}__pair.json",
                    {
                        "schema_version": 1,
                        "support_pair_id": spec.support_pair_id,
                        "geometry": geometry,
                        "oracle_comparability": comparability,
                        "strict_oracle_balance": strict,
                    },
                )
            _persist_candidate_state(
                run_dir,
                candidate_id,
                constructed.snapshot,
                record["state_sha256"],
            )
            atomic_write_json(
                run_dir / "candidates" / f"{candidate_id}.json", record
            )
            completed_count += 1
            _update_manifest(
                run_dir,
                manifest,
                status="in_progress",
                candidate_count=completed_count,
                last_completed_candidate=candidate_id,
                support_reference_bank_sha256=support_bank.sha256,
            )
            print(f"complete {candidate_id}", flush=True)
        except Exception as exc:
            _record_error(run_dir, candidate_id, exc)
            _update_manifest(
                run_dir,
                manifest,
                status="failed",
                candidate_count=completed_count,
                failed_candidate=candidate_id,
                failure_type=type(exc).__name__,
                failure_message=str(exc),
            )
            raise
        finally:
            environment.close()
    actual_count = sum(
        _completed_record(
            run_dir,
            candidate_id,
            contract_sha256=contract_sha,
            selection_lock_sha256=selection_lock["selection_lock_sha256"],
        )
        is not None
        for candidate_id in completion["candidate_ids"]
    )
    if actual_count == int(completion["expected_candidate_count"]):
        records = [
            _completed_record(
                run_dir,
                candidate_id,
                contract_sha256=contract_sha,
                selection_lock_sha256=selection_lock["selection_lock_sha256"],
            )
            for candidate_id in completion["candidate_ids"]
        ]
        _update_manifest(
            run_dir,
            manifest,
            status="complete",
            candidate_count=actual_count,
            candidate_records_sha256=canonical_sha256(records),
            completed_at=datetime.now(UTC).isoformat(),
        )
        print(f"v35 completion shard complete: {run_dir}")
    else:
        _update_manifest(
            run_dir,
            manifest,
            status="in_progress",
            candidate_count=actual_count,
        )
        print(f"v35 completion shard paused at {actual_count}/13")


if __name__ == "__main__":
    main()
