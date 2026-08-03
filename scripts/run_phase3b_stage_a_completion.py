#!/usr/bin/env python
"""Complete an additive Stage A root shard without replaying finished evidence."""

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
from smolvla_analysis.phase3b_registered_validation import (
    validate_support_pair_records_compatible,
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
    run_action_phase_oracle_from_prepared_root,
    run_goal_oracle,
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
)


DEFAULT_CONFIG = PROJECT / "configs/phase3b_stage_a_v37.yaml"
DEFAULT_REGISTERED_SMOKE = (
    PROJECT
    / "local/phase3b_stage_a/registered_generalization_smokes/"
    "registered_smoke_20260731T073347Z"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an additive Stage A completion shard."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--registered-smoke", type=Path, default=DEFAULT_REGISTERED_SMOKE
    )
    parser.add_argument("--candidate-id", action="append")
    parser.add_argument("--stop-after-candidates", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--smoke-candidate-id",
        help=(
            "Construct one candidate and checkpoint only the two locked smoke "
            "proposals; a later normal resume reuses them."
        ),
    )
    parser.add_argument("--smoke-drawer-episode", type=int)
    parser.add_argument("--smoke-cabinet-episode", type=int)
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
        raise ValueError("Stage A config has no completion block")
    revision = str(completion.get("revision", COMPLETION_REVISION))
    if not revision.startswith("phase3b-stage-a-"):
        raise ValueError("Completion revision is not namespaced")
    expected_count = int(completion.get("expected_candidate_count", -1))
    candidate_ids = validate_completion_candidate_ids(
        completion.get("candidate_ids", ()), expected_count=expected_count
    )
    imports = completion.get("imports")
    if not isinstance(imports, list) or len(imports) > 1:
        raise ValueError("Completion accepts at most one prior goal import")
    gate_report = completion.get("construction_gate_report")
    if gate_report is not None and (
        not isinstance(gate_report, str) or not gate_report.strip()
    ):
        raise ValueError("Construction-gate report path is invalid")
    causal_smoke = completion.get("causal_smoke")
    if causal_smoke is not None:
        required_smoke = {
            "candidate_id",
            "proposal_episode_by_goal",
            "require_all_pass",
        }
        if (
            not isinstance(causal_smoke, dict)
            or set(causal_smoke) != required_smoke
            or causal_smoke.get("candidate_id") not in candidate_ids
            or set(causal_smoke.get("proposal_episode_by_goal", {}))
            != set(GOALS)
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in causal_smoke["proposal_episode_by_goal"].values()
            )
            or causal_smoke.get("require_all_pass") is not True
        ):
            raise ValueError("Completion causal-smoke contract is invalid")
    if not imports:
        return {
            **completion,
            "revision": revision,
            "candidate_ids": candidate_ids,
        }
    imported = imports[0]
    required = {
        "candidate_id",
        "goal",
        "source_run",
        "source_config",
        "preserve_negative_goal",
    }
    if set(imported) != required:
        raise ValueError("Completion import contract changed")
    if (
        imported["candidate_id"] not in candidate_ids
        or imported["goal"] != "drawer"
        or imported["preserve_negative_goal"] != "cabinet"
    ):
        raise ValueError("Completion import identity changed")
    return {
        **completion,
        "revision": revision,
        "candidate_ids": candidate_ids,
    }


def _completion_contract(
    config_path: Path,
    config: dict[str, Any],
    demos: dict[str, Any],
    proposal_banks: dict[str, tuple[Any, ...]],
    completion: dict[str, Any],
    smoke_dir: Path | None,
    construction_gate: dict[str, Any] | None,
) -> dict[str, Any]:
    base = _contract(config_path, config, demos, proposal_banks)
    registered_smoke = None
    if smoke_dir is not None:
        smoke_files = {
            name: _file_sha256(smoke_dir / name)
            for name in ("contract.json", "result.json", "manifest.json")
        }
        registered_smoke = {
            "run_dir": smoke_dir.relative_to(PROJECT).as_posix(),
            "artifact_sha256": smoke_files,
        }
    return {
        **base,
        "stage": "phase3b_stage_a_completion",
        "completion_revision": completion["revision"],
        "completion_assignment": {
            "expected_candidate_count": completion["expected_candidate_count"],
            "candidate_ids": list(completion["candidate_ids"]),
            "imports": completion["imports"],
            "causal_smoke": completion.get("causal_smoke"),
        },
        "registered_smoke": registered_smoke,
        "construction_gate": construction_gate,
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


def _load_construction_gate_binding(
    completion: dict[str, Any],
    *,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    """Load a compact, Git-backed construction gate as an identity contract."""

    configured = completion.get("construction_gate_report")
    if configured is None:
        return None
    report_dir = (PROJECT / configured).resolve()
    report_root = (PROJECT / "reports/phase3b_stage_a").resolve()
    if report_dir == report_root or report_root not in report_dir.parents:
        raise ValueError("Construction-gate report must remain under reports")
    paths = {
        name: report_dir / name
        for name in ("contract.json", "summary.json", "manifest.json")
    }
    if not all(path.is_file() for path in paths.values()):
        raise ValueError("Construction-gate report is incomplete")
    report_contract = json.loads(paths["contract.json"].read_text())
    summary = json.loads(paths["summary.json"].read_text())
    manifest = json.loads(paths["manifest.json"].read_text())
    artifact_hashes = {
        name: _file_sha256(path)
        for name, path in paths.items()
        if name != "manifest.json"
    }
    expected_ids = list(completion["candidate_ids"])
    conditions = summary.get("conditions")
    by_candidate = {
        item.get("candidate_id"): item
        for item in conditions
        if isinstance(item, dict)
    } if isinstance(conditions, list) else {}
    expected_timestep = int(config["construction"]["root_final_timestep"])
    if (
        manifest.get("status") != "complete"
        or manifest.get("policy_loaded") is not False
        or summary.get("all_pass") is not True
        or int(summary.get("candidate_count", -1)) != len(expected_ids)
        or report_contract.get("candidate_ids") != expected_ids
        or report_contract.get("construction") != config["construction"]
        or set(by_candidate) != set(expected_ids)
        or any(
            item.get("root_pass") is not True
            or item.get("support_pass") is not True
            or item.get("certificate_pass") is not True
            for item in by_candidate.values()
        )
        or any(
            int(item.get("final_timestep", -1)) != expected_timestep
            for item in by_candidate.values()
        )
        or any(
            len(str(item.get("state_sha256", ""))) != 64
            or len(str(item.get("construction_action_sha256", ""))) != 64
            for item in by_candidate.values()
        )
        or manifest.get("artifact_sha256", {}).get("contract.json")
        != artifact_hashes["contract.json"]
        or manifest.get("artifact_sha256", {}).get("summary.json")
        != artifact_hashes["summary.json"]
        or manifest.get("summary_sha256") != canonical_sha256(summary)
    ):
        raise ValueError("Construction-gate identity contract changed")
    return {
        "report_dir": report_dir.relative_to(PROJECT).as_posix(),
        "artifact_sha256": {
            **artifact_hashes,
            "manifest.json": _file_sha256(paths["manifest.json"]),
        },
        "report_contract_sha256": canonical_sha256(report_contract),
        "summary_sha256": canonical_sha256(summary),
        "expected_candidates": {
            candidate_id: {
                "state_sha256": by_candidate[candidate_id]["state_sha256"],
                "construction_action_sha256": by_candidate[candidate_id][
                    "construction_action_sha256"
                ],
                "final_timestep": expected_timestep,
            }
            for candidate_id in expected_ids
        },
    }


def _assert_construction_gate_identity(
    constructed: Any,
    *,
    candidate_id: str,
    gate: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if gate is None:
        return None
    expected = gate["expected_candidates"][candidate_id]
    observed = {
        "state_sha256": snapshot_sha256(constructed.snapshot),
        "construction_action_sha256": constructed.construction[
            "action_sha256"
        ],
        "final_timestep": int(constructed.construction["final_timestep"]),
    }
    if observed != expected:
        raise RuntimeError(
            f"Construction-gate identity mismatch for {candidate_id}"
        )
    return {
        "pass": True,
        "report_dir": gate["report_dir"],
        "expected": expected,
        "observed": observed,
    }


def _smoke_episode_indices(
    proposal_banks: dict[str, tuple[Any, ...]],
    *,
    drawer_episode: int,
    cabinet_episode: int,
) -> dict[str, int]:
    requested = {"drawer": drawer_episode, "cabinet": cabinet_episode}
    indices: dict[str, int] = {}
    for goal, episode in requested.items():
        matches = [
            index
            for index, proposal in enumerate(proposal_banks[goal])
            if proposal.episode_index == episode
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Smoke proposal episode is not unique: {goal}/{episode}"
            )
        indices[goal] = matches[0]
    return indices


def _run_sparse_causal_smoke(
    environment,
    *,
    run_dir: Path,
    constructed: Any,
    spec: Any,
    proposal_banks: dict[str, tuple[Any, ...]],
    phase_banks: dict[tuple[str, str], tuple[Any, ...]],
    config: dict[str, Any],
    contract_sha256: str,
    selection_lock_sha256: str,
    smoke_indices: dict[str, int],
    construction_gate_identity: dict[str, Any] | None,
) -> dict[str, Any]:
    """Checkpoint two prospective attempts without promoting a candidate."""

    root_sha = snapshot_sha256(constructed.snapshot)
    results: dict[str, dict[str, Any]] = {}
    simulated_now = 0
    for goal in GOALS:
        proposals = proposal_banks[goal]
        phases = phase_banks[(goal, spec.layout)]
        proposal_index = smoke_indices[goal]
        completed, record_result, _ = _checkpoint(
            run_dir,
            candidate_id=spec.candidate_id,
            goal=goal,
            root_state_sha256=root_sha,
            contract_sha256=contract_sha256,
            selection_lock_sha256=selection_lock_sha256,
            proposals=proposals,
            phase_proposals=phases,
            imported_results={},
            imported_provenance={},
        )
        if proposal_index in completed:
            result = completed[proposal_index]
            provenance = "reused_same_run_checkpoint"
        else:
            preparation = run_goal_oracle(
                environment,
                constructed.snapshot,
                spec=spec,
                goal=goal,
                demo=proposals[proposal_index],
                initial_bowl_position=constructed.initial_bowl_position,
                initial_eef_position=constructed.initial_eef_position,
                initial_eef_orientation=constructed.initial_eef_orientation,
                initial_joint_positions=constructed.initial_joint_positions,
                recovery_waypoints=constructed.recovery_waypoints,
                config=config,
                raise_on_failure=False,
                return_prepared_root=True,
                normalization_only=True,
            )
            prepared = preparation.pop("_prepared_oracle_root")
            result = run_action_phase_oracle_from_prepared_root(
                environment,
                constructed.snapshot,
                prepared,
                spec=spec,
                proposal=phases[proposal_index],
                config=config,
            )
            record_result(proposal_index, result)
            simulated_now += 1
            provenance = "simulated_now_and_checkpointed"
        proposal = proposals[proposal_index]
        bridge = result.get("phases", {}).get("action_phase_bridge", {})
        registration = bridge.get("root_landmark_registration", {})
        if (
            result.get("goal") != goal
            or result.get("demo_episode_index") != proposal.episode_index
            or result.get("demo_task_index") != proposal.task_index
            or result.get("demo_action_sha256") != proposal.action_sha256
            or result.get("phase_proposal") != phases[proposal_index].metadata
            or result.get("pass") is not True
            or bridge.get("pass") is not True
            or registration.get("mode")
            != "normalized_bowl_translation_v1"
            or float(registration.get("normalized_bowl_residual_norm_m", -1.0))
            < 0.0
            or float(registration.get("normalized_bowl_residual_norm_m", -1.0))
            > float(registration.get("tolerance_m", -1.0))
        ):
            raise RuntimeError(
                f"Sparse causal smoke failed for {spec.candidate_id}/{goal}/"
                f"{proposal_index}"
            )
        results[goal] = {
            "goal": goal,
            "proposal_index": proposal_index,
            "proposal_episode_index": proposal.episode_index,
            "pass": True,
            "bridge_pass": True,
            "root_landmark_registration": registration,
            "normalized_state_sha256": result["normalized_state_sha256"],
            "normalization_action_sha256": result[
                "normalization_action_sha256"
            ],
            "result_sha256": canonical_sha256(result),
            "provenance": provenance,
        }
    normalized_states = {
        item["normalized_state_sha256"] for item in results.values()
    }
    normalization_actions = {
        item["normalization_action_sha256"] for item in results.values()
    }
    if len(normalized_states) != 1 or len(normalization_actions) != 1:
        raise RuntimeError("Sparse causal-smoke goals used different normalized roots")
    checkpoint_hashes = {
        goal: _file_sha256(
            run_dir
            / "checkpoints"
            / f"{spec.candidate_id}__{goal}.json"
        )
        for goal in GOALS
    }
    audit = {
        "schema_version": 1,
        "audit_revision": "normalized-root-registration-smoke-v1",
        "candidate_id": spec.candidate_id,
        "root_state_sha256": root_sha,
        "construction_gate_identity": construction_gate_identity,
        "all_pass": True,
        "candidate_promotions": 0,
        "proposal_attempts_in_smoke": len(GOALS),
        "proposal_attempts_simulated_now": simulated_now,
        "proposal_attempts_reused_from_same_run": len(GOALS) - simulated_now,
        "shared_normalized_state_sha256": next(iter(normalized_states)),
        "shared_normalization_action_sha256": next(
            iter(normalization_actions)
        ),
        "conditions": [results[goal] for goal in GOALS],
        "checkpoint_file_sha256": checkpoint_hashes,
        "policy_loaded": False,
    }
    audit_path = run_dir / "audits/normalized_root_registration_smoke.json"
    atomic_write_json(audit_path, audit)
    return {
        **audit,
        "audit_file_sha256": _file_sha256(audit_path),
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
        "checkpoint_revision": "phase3b-additive-sparse-import-v1",
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
    smoke_arguments = (
        args.smoke_candidate_id,
        args.smoke_drawer_episode,
        args.smoke_cabinet_episode,
    )
    smoke_mode = all(value is not None for value in smoke_arguments)
    if any(value is not None for value in smoke_arguments) and not smoke_mode:
        raise ValueError("Sparse smoke requires a candidate and both episodes")
    if smoke_mode and (
        args.candidate_id
        or args.stop_after_candidates is not None
        or args.preflight_only
    ):
        raise ValueError("Sparse smoke cannot be combined with run filters")
    config_path = args.config.resolve()
    config = _load_config(config_path)
    completion = _validate_completion_config(config)
    smoke_plan = completion.get("causal_smoke")
    if smoke_mode:
        observed_smoke_plan = {
            "candidate_id": args.smoke_candidate_id,
            "proposal_episode_by_goal": {
                "drawer": args.smoke_drawer_episode,
                "cabinet": args.smoke_cabinet_episode,
            },
            "require_all_pass": True,
        }
        if observed_smoke_plan != smoke_plan:
            raise ValueError("CLI sparse smoke does not match the config contract")
    demos = _load_demos(config)
    proposal_banks = _load_proposal_bank(config)
    minimum_horizon = _minimum_native_horizon(config, demos, proposal_banks)
    if int(config["environment"]["episode_length"]) < minimum_horizon:
        raise ValueError("Completion native horizon is too short")
    base_contract_sha = canonical_sha256(
        _contract(config_path, config, demos, proposal_banks)
    )
    smoke_dir = args.registered_smoke.resolve() if completion["imports"] else None
    smoke_conditions = (
        _load_registered_smoke(
            smoke_dir, base_contract_sha256=base_contract_sha
        )
        if smoke_dir is not None
        else {}
    )
    construction_gate = _load_construction_gate_binding(
        completion, config=config
    )
    contract = _completion_contract(
        config_path,
        config,
        demos,
        proposal_banks,
        completion,
        smoke_dir,
        construction_gate,
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
                    "completion_revision": completion["revision"],
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
                    "causal_smoke": smoke_plan,
                    "construction_gate_report": (
                        construction_gate["report_dir"]
                        if construction_gate is not None
                        else None
                    ),
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
    smoke_indices = None
    if smoke_mode:
        selected = (str(args.smoke_candidate_id),)
        smoke_indices = _smoke_episode_indices(
            proposal_banks,
            drawer_episode=int(args.smoke_drawer_episode),
            cabinet_episode=int(args.smoke_cabinet_episode),
        )
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
    import_spec = completion["imports"][0] if completion["imports"] else None
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
        registered_grasp_acquisition = None
        if (
            spec.drawer_aperture == "open"
            and spec.possession == "grasped"
            and config["construction"].get("open_grasped_acquisition_mode")
            == "registered_cabinet_phase_v1"
        ):
            acquisition_episode = int(
                config["construction"][
                    "registered_grasp_acquisition_episode_index"
                ]
            )
            matches = [
                proposal
                for proposal in phase_banks[("cabinet", spec.layout)]
                if proposal.source.episode_index == acquisition_episode
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Registered grasp proposal changed for {candidate_id}"
                )
            registered_grasp_acquisition = matches[0]
        print(f"construct {candidate_id}", flush=True)
        environment = make_stage_a_environment(PROJECT, run_dir, config)
        try:
            constructed = construct_candidate(
                environment,
                spec,
                demos,
                config,
                support_reference_bank=support_bank,
                registered_grasp_acquisition=registered_grasp_acquisition,
            )
            certificate = certify_computational_state(
                environment,
                constructed.snapshot,
                possession=spec.possession,
                probe_actions=config["certificate"]["actions"],
            )
            if certificate["pass"] is not True:
                raise RuntimeError(
                    f"Completion certificate failed for {candidate_id}"
                )
            construction_gate_identity = _assert_construction_gate_identity(
                constructed,
                candidate_id=candidate_id,
                gate=construction_gate,
            )
            if smoke_mode:
                if smoke_indices is None:
                    raise RuntimeError("Sparse smoke indices were not resolved")
                smoke_audit = _run_sparse_causal_smoke(
                    environment,
                    run_dir=run_dir,
                    constructed=constructed,
                    spec=spec,
                    proposal_banks=proposal_banks,
                    phase_banks=phase_banks,
                    config=config,
                    contract_sha256=contract_sha,
                    selection_lock_sha256=selection_lock[
                        "selection_lock_sha256"
                    ],
                    smoke_indices=smoke_indices,
                    construction_gate_identity=construction_gate_identity,
                )
                _update_manifest(
                    run_dir,
                    manifest,
                    status="causal_smoke_complete",
                    candidate_count=completed_count,
                    causal_smoke_all_pass=True,
                    causal_smoke_audit_sha256=smoke_audit[
                        "audit_file_sha256"
                    ],
                    causal_smoke_attempt_count=smoke_audit[
                        "proposal_attempts_in_smoke"
                    ],
                    causal_smoke_candidate_promotions=0,
                    support_reference_bank_sha256=support_bank.sha256,
                )
                print(
                    "Sparse causal smoke complete; resume this same run "
                    "without smoke flags.",
                    flush=True,
                )
                return
            oracles = {}
            imports = {}
            prior_negative = {}
            for goal in GOALS:
                if (
                    import_spec is not None
                    and candidate_id == import_spec["candidate_id"]
                    and goal == import_spec["goal"]
                ):
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
                    if smoke_dir is None:
                        raise RuntimeError(
                            "Registered smoke condition exists without a source"
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
            record["completion_revision"] = completion["revision"]
            record["oracle_imports"] = imports
            record["prior_negative_oracle_evidence"] = prior_negative
            record["construction_gate_identity"] = construction_gate_identity
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
                    strict = validate_support_pair_records_compatible(
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
        print(f"Completion shard complete: {run_dir}")
    else:
        _update_manifest(
            run_dir,
            manifest,
            status="in_progress",
            candidate_count=actual_count,
        )
        print(
            "Completion shard paused at "
            f"{actual_count}/{completion['expected_candidate_count']}"
        )


if __name__ == "__main__":
    main()
