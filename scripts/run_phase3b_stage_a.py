#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml
import zarr

from smolvla_analysis.phase2_storage import read_libero_snapshot, write_libero_snapshot
from smolvla_analysis.phase3_crd import atomic_write_json
from smolvla_analysis.phase3b_libero import (
    action_phase_suffix,
    build_action_phase_proposal_bank,
    build_support_reference_bank,
    candidate_record,
    certify_computational_state,
    compact_snapshot_metadata,
    construct_candidate,
    list_demo_trace_inventory,
    load_demo_trace,
    make_stage_a_environment,
    run_goal_oracle_bank,
)
from smolvla_analysis.phase3b_stage_a import (
    FACTOR_LEVELS,
    GOALS,
    build_selection_lock,
    candidate_spec,
    canonical_sha256,
    iter_candidate_specs,
    snapshot_sha256,
    validate_selection_lock,
    validate_stage_a_records,
    validate_support_pair_records,
)


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT / "configs/phase3b_stage_a.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and certify the policy-independent Phase 3b Stage A lattice."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--candidate-id",
        action="append",
        help="Construct only this locked candidate; repeat to request multiple candidates.",
    )
    parser.add_argument(
        "--stop-after-candidates",
        type=int,
        help="Stop after this many total candidate records for a resumable smoke.",
    )
    parser.add_argument(
        "--audit-candidate-id",
        help="Reconstruct one completed candidate in this fresh process and compare its state hash.",
    )
    parser.add_argument(
        "--oracle-goal",
        action="append",
        choices=GOALS,
        help=(
            "Evaluate and checkpoint only this goal without promoting a candidate; "
            "repeat for multiple goals. Intended for bounded causal smokes."
        ),
    )
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    config = yaml.safe_load(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("Stage A config must have schema_version=1")
    if int(config["environment"]["task_id"]) != 3:
        raise ValueError("Stage A must use the shared LIBERO task-3 scene")
    if [
        int(config["environment"]["observation_height"]),
        int(config["environment"]["observation_width"]),
    ] != [256, 256]:
        raise ValueError("Stage A image provenance is locked to 256x256")
    expected_demo_tasks = {
        "drawer_construction": 12,
        "grasp_construction": 18,
        "drawer": 12,
        "cabinet": 18,
    }
    missing_roles = expected_demo_tasks.keys() - config["demonstrations"].keys()
    if missing_roles:
        raise ValueError(
            f"Stage A demonstration roles are incomplete: {sorted(missing_roles)}"
        )
    for role, expected_task in expected_demo_tasks.items():
        if int(config["demonstrations"][role]["task_index"]) != expected_task:
            raise ValueError(
                f"Stage A demonstration role {role} must use task {expected_task}"
            )
    proposal_tasks = {"drawer": 12, "cabinet": 18}
    proposal_config = config.get("oracle_proposals", {})
    if set(proposal_config) != set(proposal_tasks):
        raise ValueError("Stage A oracle proposal banks must cover both goals")
    for goal, expected_task in proposal_tasks.items():
        proposal = proposal_config[goal]
        expected_proposal = {
            "source": "all_cached_task_demonstrations",
            "task_index": expected_task,
            "order_by": ["frame_count", "episode_index"],
        }
        if proposal != expected_proposal:
            raise ValueError(
                f"Stage A {goal} proposal inventory must be {expected_proposal}"
            )
    if config.get("oracle", {}).get("proposal_selection_rule") != (
        "minimum_executed_steps_then_path_effort_index"
    ):
        raise ValueError("Stage A proposal selection rule is not locked")
    if int(config["oracle"]["setdown_reverse_root_budget"]) != int(
        config["construction"]["root_servo_budget"]
    ):
        raise ValueError(
            "Reverse-recovery budget must equal the fixed root-servo budget"
        )
    transit_budgets = config["construction"].get(
        "grasped_root_transit_budgets", {}
    )
    expected_transit_phases = {
        "clearance_lift",
        "clearance_transit",
        "target_descent",
    }
    if set(transit_budgets) != expected_transit_phases or any(
        int(value) < 1 for value in transit_budgets.values()
    ):
        raise ValueError(
            "Grasped-root transit requires three positive phase budgets"
        )
    if sum(int(value) for value in transit_budgets.values()) != int(
        config["construction"]["root_servo_budget"]
    ):
        raise ValueError(
            "Grasped-root transit budgets must sum to root_servo_budget"
        )
    clearance_margin = float(
        config["construction"]["grasped_root_clearance_margin_m"]
    )
    waypoint_tolerance = float(
        config["construction"]["grasped_root_waypoint_tolerance_m"]
    )
    if (
        not np.isfinite(clearance_margin)
        or not np.isfinite(waypoint_tolerance)
        or waypoint_tolerance <= 0.0
        or clearance_margin <= waypoint_tolerance
    ):
        raise ValueError(
            "Grasped-root clearance must exceed its positive waypoint tolerance"
        )
    action_phase = config.get("action_phase_oracle", {})
    expected_phase_labels = {
        "applicability": "all_goals_all_drawer_apertures",
        "execution_mode": "action_intrinsic_pregrasp_phase_continuation_v2",
        "anchor_rule": (
            "capped_pregrasp_lead_before_first_gripper_close_transition"
        ),
        "bridge_mode": "three_leg_clearance_lift_transit_descent",
        "normalization_preparation": (
            "normalization_only_no_source_replay"
        ),
    }
    for field, expected in expected_phase_labels.items():
        if action_phase.get(field) != expected:
            raise ValueError(
                f"Stage A action-phase field {field} must be {expected!r}"
            )
    phase_budgets = action_phase.get("bridge_phase_budgets", {})
    if set(phase_budgets) != expected_transit_phases or any(
        int(value) < 1 for value in phase_budgets.values()
    ):
        raise ValueError(
            "Drawer phase bridge requires three positive phase budgets"
        )
    if (
        int(action_phase.get("maximum_pregrasp_lead_frames", 0)) < 1
        or int(action_phase.get("minimum_anchor_prefix_frames", 0)) < 1
    ):
        raise ValueError("Action-phase lead/prefix bounds must be positive")
    phase_scalars = {
        field: float(action_phase.get(field, np.nan))
        for field in (
            "gripper_close_threshold",
            "gripper_action",
            "clearance_margin_m",
            "intermediate_tolerance_m",
            "final_tolerance_m",
            "orientation_tolerance_rad",
            "max_translation_action",
            "bowl_drift_tolerance_m",
        )
    }
    if not all(np.isfinite(value) for value in phase_scalars.values()):
        raise ValueError("Action-phase scalar parameters must be finite")
    if not -1.0 <= phase_scalars["gripper_close_threshold"] <= 1.0:
        raise ValueError("Action-phase gripper threshold is outside [-1, 1]")
    if phase_scalars["gripper_action"] != -1.0:
        raise ValueError("Action-phase bridge must keep the gripper open")
    if not (
        phase_scalars["clearance_margin_m"]
        > phase_scalars["intermediate_tolerance_m"]
        > 0.0
        and phase_scalars["final_tolerance_m"] > 0.0
        and phase_scalars["orientation_tolerance_rad"] > 0.0
        and 0.0 < phase_scalars["max_translation_action"] <= 1.0
        and phase_scalars["bowl_drift_tolerance_m"] > 0.0
    ):
        raise ValueError("Action-phase bridge tolerances are invalid")
    workspace = action_phase.get("workspace_bounds_m", {})
    if set(workspace) != {"x", "y", "z"}:
        raise ValueError("Action-phase workspace bounds are incomplete")
    for axis, bounds in workspace.items():
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or not all(np.isfinite(float(value)) for value in bounds)
            or float(bounds[0]) >= float(bounds[1])
        ):
            raise ValueError(f"Invalid action-phase {axis} workspace bounds")
    probe_actions = config.get("certificate", {}).get("actions", [])
    if not probe_actions or any(len(action) != 6 for action in probe_actions):
        raise ValueError(
            "Stage A requires non-empty six-axis certificate actions"
        )
    return config


def _dataset_root(config: dict[str, Any]) -> Path:
    path = Path(config["demonstrations"]["dataset_root"])
    return path if path.is_absolute() else PROJECT / path


def _load_demos(config: dict[str, Any]):
    root = _dataset_root(config)
    roles = {
        "drawer_construction": "drawer",
        "grasp_construction": "cabinet",
        "drawer": "drawer",
        "cabinet": "cabinet",
    }
    return {
        role: load_demo_trace(
            root,
            goal=goal,
            episode_index=int(config["demonstrations"][role]["episode_index"]),
            task_index=int(config["demonstrations"][role]["task_index"]),
        )
        for role, goal in roles.items()
    }


def _load_proposal_bank(config: dict[str, Any]):
    root = _dataset_root(config)
    banks = {}
    for goal in GOALS:
        task_index = int(config["oracle_proposals"][goal]["task_index"])
        inventory = list_demo_trace_inventory(root, task_index=task_index)
        banks[goal] = tuple(
            load_demo_trace(
                root,
                goal=goal,
                episode_index=item["episode_index"],
                task_index=task_index,
            )
            for item in inventory
        )
        observed = tuple(
            (len(demo.actions), demo.episode_index) for demo in banks[goal]
        )
        expected = tuple(
            (item["frame_count"], item["episode_index"])
            for item in inventory
        )
        if observed != expected:
            raise RuntimeError(f"Stage A {goal} proposal inventory changed while loading")
    return banks


def _oracle_demos(demos: dict[str, Any]) -> dict[str, Any]:
    return {goal: demos[goal] for goal in GOALS}


def _minimum_native_horizon(
    config: dict[str, Any], demos: dict[str, Any], proposal_bank: dict[str, Any]
) -> int:
    oracle_cfg = config["oracle"]
    recovery_budget = sum(
        int(oracle_cfg[key])
        for key in (
            "setdown_reverse_root_budget",
            "setdown_return_lifted_budget",
            "setdown_return_grasp_budget",
            "setdown_table_safe_budget",
            "setdown_descend_budget",
            "setdown_release_budget",
            "setdown_retreat_budget",
        )
    )
    longest_demo = max(
        [len(demo.actions) for demo in demos.values()]
        + [
            len(proposal.actions)
            for proposals in proposal_bank.values()
            for proposal in proposals
        ]
    )
    full_trajectory_horizon = (
        int(config["construction"]["root_final_timestep"])
        + recovery_budget
        + int(oracle_cfg["home_budget"])
        + int(longest_demo)
        + 1
    )
    phase_cfg = config["action_phase_oracle"]
    longest_goal_suffix = max(
        len(
            action_phase_suffix(
                proposal,
                maximum_pregrasp_lead_frames=int(
                    phase_cfg["maximum_pregrasp_lead_frames"]
                ),
                minimum_anchor_prefix_frames=int(
                    phase_cfg["minimum_anchor_prefix_frames"]
                ),
                gripper_close_threshold=float(
                    phase_cfg["gripper_close_threshold"]
                ),
            )[0].actions
        )
        for proposals in proposal_bank.values()
        for proposal in proposals
    )
    phase_horizon = (
        int(config["construction"]["root_final_timestep"])
        + recovery_budget
        + int(oracle_cfg["home_budget"])
        + sum(
            int(value)
            for value in phase_cfg["bridge_phase_budgets"].values()
        )
        + int(longest_goal_suffix)
        + 1
    )
    return max(full_trajectory_horizon, phase_horizon)


def _contract(
    config_path: Path,
    config: dict[str, Any],
    demos: dict[str, Any],
    proposal_bank: dict[str, Any],
) -> dict[str, Any]:
    relative_dataset_root = Path(config["demonstrations"]["dataset_root"]).as_posix()
    return {
        "schema_version": 1,
        "stage": "phase3b_stage_a",
        "construction_revision": config["construction_revision"],
        "config_path": config_path.resolve().relative_to(PROJECT).as_posix(),
        "config_sha256": _file_sha256(config_path.resolve()),
        "environment": config["environment"],
        "minimum_native_horizon": _minimum_native_horizon(
            config, demos, proposal_bank
        ),
        "dataset_root": relative_dataset_root,
        "demonstrations": {
            role: {
                "trace_goal": demo.goal,
                "purpose": (
                    "state_construction"
                    if role.endswith("_construction")
                    else "support_reference"
                ),
                "episode_index": demo.episode_index,
                "task_index": demo.task_index,
                "frame_count": int(len(demo.actions)),
                "action_sha256": demo.action_sha256,
            }
            for role, demo in demos.items()
        },
        "oracle_proposal_inventory_spec": config["oracle_proposals"],
        "oracle_proposal_selection_rule": config["oracle"][
            "proposal_selection_rule"
        ],
        "action_phase_oracle": config["action_phase_oracle"],
        "action_phase_proposal_slices": {
            goal: [
                {
                    "proposal_index": index,
                    "episode_index": proposal.episode_index,
                    "task_index": proposal.task_index,
                    "source_action_sha256": proposal.action_sha256,
                    **action_phase_suffix(
                        proposal,
                        maximum_pregrasp_lead_frames=int(
                            config["action_phase_oracle"][
                                "maximum_pregrasp_lead_frames"
                            ]
                        ),
                        minimum_anchor_prefix_frames=int(
                            config["action_phase_oracle"][
                                "minimum_anchor_prefix_frames"
                            ]
                        ),
                        gripper_close_threshold=float(
                            config["action_phase_oracle"][
                                "gripper_close_threshold"
                            ]
                        ),
                    )[1],
                }
                for index, proposal in enumerate(proposals)
            ]
            for goal, proposals in proposal_bank.items()
        },
        "oracle_proposal_bank": {
            goal: [
                {
                    "proposal_index": index,
                    "purpose": "goal_feasibility_proposal",
                    "episode_index": proposal.episode_index,
                    "task_index": proposal.task_index,
                    "frame_count": int(len(proposal.actions)),
                    "action_sha256": proposal.action_sha256,
                }
                for index, proposal in enumerate(proposals)
            ]
            for goal, proposals in proposal_bank.items()
        },
        "candidate_ids": [spec.candidate_id for spec in iter_candidate_specs()],
        "goals": {
            "drawer": [
                "In",
                "akita_black_bowl_1",
                "wooden_cabinet_1_top_region",
            ],
            "cabinet": [
                "On",
                "akita_black_bowl_1",
                "wooden_cabinet_1_top_side",
            ],
        },
        "certificate": config["certificate"],
        "support_metric": config["support_metric"],
        "validation": config["validation"],
        "source_sha256": {
            "runner": _file_sha256(Path(__file__).resolve()),
            "runtime": _file_sha256(
                PROJECT / "src/smolvla_analysis/phase3b_libero.py"
            ),
            "lattice": _file_sha256(
                PROJECT / "src/smolvla_analysis/phase3b_stage_a.py"
            ),
            "libero_state": _file_sha256(
                PROJECT / "src/smolvla_analysis/libero_state.py"
            ),
            "phase3_crd": _file_sha256(
                PROJECT / "src/smolvla_analysis/phase3_crd.py"
            ),
            "phase2_storage": _file_sha256(
                PROJECT / "src/smolvla_analysis/phase2_storage.py"
            ),
            "runtime_config": _file_sha256(
                PROJECT / "src/smolvla_analysis/runtime.py"
            ),
        },
        "policy_loaded": False,
        "canonical_rollout_reused": False,
    }


def _run_directory(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        run_dir = args.run_dir.resolve()
    else:
        run_id = datetime.now(UTC).strftime("phase3b_stage_a_%Y%m%dT%H%M%SZ")
        run_dir = (PROJECT / "local/phase3b_stage_a" / run_id).resolve()
    raw_root = (PROJECT / "local/phase3b_stage_a").resolve()
    if run_dir != raw_root and raw_root not in run_dir.parents:
        raise ValueError(
            "Stage A raw output must remain under local/phase3b_stage_a"
        )
    return run_dir


def _initialize_run(
    run_dir: Path,
    contract: dict[str, Any],
    selection_lock: dict[str, Any],
) -> dict[str, Any]:
    contract_sha = canonical_sha256(contract)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        contract_path = run_dir / "contract.json"
        if not contract_path.is_file() or json.loads(contract_path.read_text()) != contract:
            raise ValueError("Existing Stage A contract payload is missing or changed")
        if manifest.get("contract_sha256") != contract_sha:
            raise ValueError("Existing Stage A run has a different scientific contract")
        expected_manifest = {
            "schema_version": 1,
            "run_id": run_dir.name,
            "stage": "phase3b_stage_a",
            "selection_lock_sha256": selection_lock["selection_lock_sha256"],
            "policy_loaded": False,
            "expected_candidate_count": 32,
        }
        for key, expected in expected_manifest.items():
            if manifest.get(key) != expected:
                raise ValueError(
                    f"Existing Stage A manifest mismatch for {key}: "
                    f"{manifest.get(key)!r} != {expected!r}"
                )
        validate_selection_lock(
            json.loads((run_dir / "selection_lock.json").read_text()),
            contract_sha256=contract_sha,
            construction_revision=contract["construction_revision"],
        )
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        return manifest

    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("candidates", "errors", "audits", "checkpoints"):
        (run_dir / name).mkdir()
    atomic_write_json(run_dir / "contract.json", contract)
    atomic_write_json(run_dir / "selection_lock.json", selection_lock)
    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "stage": "phase3b_stage_a",
        "status": "in_progress",
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "contract_sha256": contract_sha,
        "selection_lock_sha256": selection_lock["selection_lock_sha256"],
        "policy_loaded": False,
        "candidate_count": 0,
        "expected_candidate_count": 32,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def _update_manifest(run_dir: Path, manifest: dict[str, Any], **updates: Any) -> None:
    manifest.update(updates)
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    atomic_write_json(run_dir / "manifest.json", manifest)


def _candidate_paths(run_dir: Path, candidate_id: str) -> tuple[Path, Path]:
    return (
        run_dir / "candidates" / f"{candidate_id}.json",
        run_dir / "states.zarr" / candidate_id,
    )


def _validate_run_inventory(run_dir: Path) -> None:
    expected = {spec.candidate_id for spec in iter_candidate_specs()}
    candidate_dir = run_dir / "candidates"
    observed_records = {
        path.stem for path in candidate_dir.glob("*.json") if path.is_file()
    }
    unknown_records = observed_records - expected
    if unknown_records:
        raise ValueError(
            f"Unknown Stage A candidate records: {sorted(unknown_records)}"
        )
    state_root_path = run_dir / "states.zarr"
    if state_root_path.exists():
        state_root = zarr.open_group(str(state_root_path), mode="r")
        observed_states = set(state_root.group_keys())
        temporary_states = sorted(
            key for key in observed_states if key.startswith("__tmp__")
        )
        unknown_states = observed_states - expected - set(temporary_states)
        if temporary_states or unknown_states:
            raise ValueError(
                "Partial or unknown Stage A state groups: "
                f"temporary={temporary_states}, unknown={sorted(unknown_states)}"
            )


def _completed_record(
    run_dir: Path,
    candidate_id: str,
    *,
    contract_sha256: str,
    selection_lock_sha256: str,
) -> dict[str, Any] | None:
    record_path, state_path = _candidate_paths(run_dir, candidate_id)
    if not record_path.exists() and not state_path.exists():
        return None
    if not record_path.is_file() or not state_path.is_dir():
        raise ValueError(f"Partial Stage A candidate artifact: {candidate_id}")
    record = json.loads(record_path.read_text())
    if record.get("contract_sha256") != contract_sha256:
        raise ValueError(f"Candidate contract mismatch: {candidate_id}")
    if record.get("selection_lock_sha256") != selection_lock_sha256:
        raise ValueError(f"Candidate selection-lock mismatch: {candidate_id}")
    state_attrs = zarr.open_group(str(state_path), mode="r").attrs
    if not state_attrs.get("complete"):
        raise ValueError(f"Candidate state group is incomplete: {candidate_id}")
    if state_attrs.get("state_sha256") != record.get("state_sha256"):
        raise ValueError(f"Candidate state hash mismatch: {candidate_id}")
    state_root = zarr.open_group(str(run_dir / "states.zarr"), mode="r")
    persisted_snapshot = read_libero_snapshot(state_root, candidate_id)
    if snapshot_sha256(persisted_snapshot) != record.get("state_sha256"):
        raise ValueError(f"Persisted Stage A state payload changed: {candidate_id}")
    return record


def _persist_candidate_state(
    run_dir: Path,
    candidate_id: str,
    snapshot,
    state_sha: str,
) -> None:
    state_root_path = run_dir / "states.zarr"
    state_root = zarr.open_group(str(state_root_path), mode="a")
    temporary_key = f"__tmp__{candidate_id}__pid{os.getpid()}"
    if temporary_key in state_root or candidate_id in state_root:
        raise ValueError(f"Refusing to overwrite Stage A state group: {candidate_id}")
    write_libero_snapshot(state_root, temporary_key, snapshot)
    temporary = state_root[temporary_key]
    temporary.attrs["state_sha256"] = state_sha
    temporary.attrs["complete"] = True
    temporary_path = state_root_path / temporary_key
    final_path = state_root_path / candidate_id
    os.replace(temporary_path, final_path)


def _candidate_error(run_dir: Path, candidate_id: str, exc: Exception) -> None:
    timestamp = datetime.now(UTC)
    error_id = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    atomic_write_json(
        run_dir / "errors" / f"{candidate_id}__{error_id}__pid{os.getpid()}.json",
        {
            "candidate_id": candidate_id,
            "timestamp": timestamp.isoformat(),
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    )


def _oracle_checkpoint(
    run_dir: Path,
    *,
    candidate_id: str,
    goal: str,
    root_state_sha256: str,
    contract_sha256: str,
    selection_lock_sha256: str,
    proposals: tuple[Any, ...],
    phase_proposals: tuple[Any, ...],
) -> tuple[
    dict[int, dict[str, Any]],
    Callable[[int, dict[str, Any]], None],
    Callable[[dict[str, Any]], None],
]:
    checkpoint_path = (
        run_dir / "checkpoints" / f"{candidate_id}__{goal}.json"
    )
    proposal_inventory = [
        {
            "proposal_index": index,
            "episode_index": proposal.episode_index,
            "task_index": proposal.task_index,
            "action_sha256": proposal.action_sha256,
        }
        for index, proposal in enumerate(proposals)
    ]
    expected = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "goal": goal,
        "root_state_sha256": root_state_sha256,
        "contract_sha256": contract_sha256,
        "selection_lock_sha256": selection_lock_sha256,
        "proposal_inventory_sha256": canonical_sha256(proposal_inventory),
        "proposal_execution_contract_sha256": canonical_sha256(
            [proposal.metadata for proposal in phase_proposals]
        ),
    }
    if checkpoint_path.is_file():
        state = json.loads(checkpoint_path.read_text())
        for field, value in expected.items():
            if state.get(field) != value:
                raise ValueError(
                    f"Oracle checkpoint mismatch for {candidate_id}/{goal}: "
                    f"{field}"
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
        atomic_write_json(checkpoint_path, state)
    rows = state.get("results")
    if not isinstance(rows, list):
        raise ValueError(f"Invalid oracle checkpoint rows for {candidate_id}/{goal}")
    indices = [row.get("proposal_index") for row in rows]
    if indices != list(range(len(rows))) or len(rows) > len(proposals):
        raise ValueError(
            f"Non-contiguous oracle checkpoint for {candidate_id}/{goal}"
        )
    if int(state.get("result_count", -1)) != len(rows):
        raise ValueError(
            f"Oracle checkpoint count mismatch for {candidate_id}/{goal}"
        )
    completed = {
        int(row["proposal_index"]): row["result"] for row in rows
    }

    def record(proposal_index: int, result: dict[str, Any]) -> None:
        expected_index = len(state["results"])
        if proposal_index != expected_index:
            raise ValueError(
                f"Oracle checkpoint write is out of order for "
                f"{candidate_id}/{goal}: {proposal_index} != {expected_index}"
            )
        state["results"].append(
            {"proposal_index": proposal_index, "result": result}
        )
        state["result_count"] = len(state["results"])
        state["last_completed_proposal_index"] = proposal_index
        state["updated_at"] = datetime.now(UTC).isoformat()
        atomic_write_json(checkpoint_path, state)

    def finish(oracle: dict[str, Any]) -> None:
        if len(state["results"]) != len(proposals):
            raise ValueError(
                f"Cannot complete partial oracle checkpoint for "
                f"{candidate_id}/{goal}"
            )
        state["status"] = "complete"
        state["oracle_sha256"] = canonical_sha256(oracle)
        state["updated_at"] = datetime.now(UTC).isoformat()
        atomic_write_json(checkpoint_path, state)

    return completed, record, finish


def _validation_limits(config: dict[str, Any]) -> dict[str, float]:
    validation = config["validation"]
    return {
        "max_oracle_cost_mismatch": float(
            validation["oracle_cost_mismatch_limit"]
        ),
        "max_realized_goal_distance_mismatch": float(
            validation["realized_goal_distance_mismatch_limit"]
        ),
        "max_planned_recovery_distance_mismatch": float(
            validation["planned_recovery_distance_mismatch_limit"]
        ),
        "max_executed_step_mismatch": float(
            validation["executed_step_mismatch_limit"]
        ),
        "max_active_step_mismatch": float(
            validation["active_step_mismatch_limit"]
        ),
        "max_eef_path_mismatch": float(validation["eef_path_mismatch_limit"]),
        "max_motion_control_effort_mismatch": float(
            validation["motion_control_effort_mismatch_limit"]
        ),
    }


def _proposal_coverage_frames(
    records: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows = []
    for record in records:
        spec = candidate_spec(record["candidate_id"])
        factors = spec.as_dict()
        for goal in GOALS:
            oracle = record["oracles"][goal]
            selected_index = int(oracle["selected_proposal_index"])
            for proposal, attempt in zip(
                oracle["proposal_bank"],
                oracle["proposal_attempts"],
                strict=True,
            ):
                rows.append(
                    {
                        "candidate_id": spec.candidate_id,
                        "family_id": spec.family_id,
                        "support_pair_id": spec.support_pair_id,
                        **{
                            factor: factors[factor]
                            for factor in FACTOR_LEVELS
                        },
                        "goal": goal,
                        "proposal_execution_mode": oracle[
                            "proposal_execution_mode"
                        ],
                        "proposal_execution_contract_sha256": oracle[
                            "proposal_execution_contract_sha256"
                        ],
                        "proposal_index": int(proposal["proposal_index"]),
                        "episode_index": int(proposal["episode_index"]),
                        "task_index": int(proposal["task_index"]),
                        "frame_count": int(proposal["frame_count"]),
                        "action_sha256": proposal["action_sha256"],
                        "pass": bool(attempt["pass"]),
                        "selected": int(proposal["proposal_index"])
                        == selected_index,
                        "first_goal_demo_frame": attempt[
                            "first_goal_demo_frame"
                        ],
                        "wrong_goal_ever_achieved": bool(
                            attempt["wrong_goal_ever_achieved"]
                        ),
                        "unexpected_done_before_goal": bool(
                            attempt["unexpected_done_before_goal"]
                        ),
                        "action_phase_bridge_pass": bool(
                            attempt["action_phase_bridge"]["pass"]
                        ),
                        "source_suffix_action_steps": attempt["cost"].get(
                            "source_suffix_action_steps"
                        ),
                        "executed_source_action_steps": attempt["cost"].get(
                            "executed_source_action_steps",
                            attempt["cost"][
                                "executed_demonstration_action_steps"
                            ],
                        ),
                        "executed_demonstration_action_steps": int(
                            attempt["cost"][
                                "executed_demonstration_action_steps"
                            ]
                        ),
                        "eef_path_length_m": float(
                            attempt["cost"]["eef_path_length_m"]
                        ),
                        "motion_control_effort": float(
                            attempt["cost"]["motion_control_effort"]
                        ),
                        "candidate_proposal_success_count": int(
                            oracle["proposal_success_count"]
                        ),
                        "candidate_proposal_success_fraction": float(
                            oracle["proposal_success_fraction"]
                        ),
                    }
                )
    coverage = pd.DataFrame(rows).sort_values(
        ["goal", "candidate_id", "proposal_index"]
    )
    generality = (
        coverage.groupby(
            [
                "goal",
                "proposal_execution_mode",
                "proposal_index",
                "episode_index",
                "task_index",
                "frame_count",
                "action_sha256",
            ],
            as_index=False,
        )
        .agg(
            successful_candidate_count=("pass", "sum"),
            candidate_count=("candidate_id", "nunique"),
            selected_candidate_count=("selected", "sum"),
        )
        .sort_values(["goal", "proposal_index"])
    )
    generality["successful_candidate_fraction"] = (
        generality["successful_candidate_count"] / generality["candidate_count"]
    )

    candidate_coverage = coverage.drop_duplicates(["candidate_id", "goal"])
    factor_rows = []
    for goal in GOALS:
        goal_frame = candidate_coverage[candidate_coverage["goal"] == goal]
        for factor, levels in FACTOR_LEVELS.items():
            means = [
                float(
                    goal_frame[goal_frame[factor] == level][
                        "candidate_proposal_success_fraction"
                    ].mean()
                )
                for level in levels
            ]
            factor_rows.append(
                {
                    "goal": goal,
                    "factor": factor,
                    "reference_level": levels[0],
                    "contrast_level": levels[1],
                    "reference_mean_success_fraction": means[0],
                    "contrast_mean_success_fraction": means[1],
                    "contrast_minus_reference": means[1] - means[0],
                }
            )
    factor_effects = pd.DataFrame(factor_rows).sort_values(["goal", "factor"])

    coverage_summary = {}
    for goal in GOALS:
        values = candidate_coverage[candidate_coverage["goal"] == goal][
            "candidate_proposal_success_fraction"
        ].to_numpy(dtype=float)
        goal_generality = generality[generality["goal"] == goal][
            "successful_candidate_fraction"
        ].to_numpy(dtype=float)
        coverage_summary[goal] = {
            "proposal_execution_mode": str(
                candidate_coverage[
                    candidate_coverage["goal"] == goal
                ]["proposal_execution_mode"].iloc[0]
            ),
            "candidate_success_fraction_min": float(np.min(values)),
            "candidate_success_fraction_median": float(np.median(values)),
            "candidate_success_fraction_mean": float(np.mean(values)),
            "candidate_success_fraction_max": float(np.max(values)),
            "proposal_generality_fraction_median": float(
                np.median(goal_generality)
            ),
            "proposal_generality_fraction_max": float(
                np.max(goal_generality)
            ),
            "never_successful_proposal_count": int(
                np.sum(goal_generality == 0.0)
            ),
        }
    return coverage, generality, factor_effects, coverage_summary


def _compact_report(
    run_dir: Path,
    manifest: dict[str, Any],
    contract: dict[str, Any],
    selection_lock: dict[str, Any],
    records: list[dict[str, Any]],
    config: dict[str, Any],
) -> Path:
    summary = validate_stage_a_records(records, **_validation_limits(config))
    ordered_records = sorted(records, key=lambda record: record["candidate_id"])
    candidate_records_sha256 = canonical_sha256(ordered_records)
    state_hash_inventory = [
        {
            "candidate_id": record["candidate_id"],
            "state_sha256": record["state_sha256"],
        }
        for record in ordered_records
    ]
    state_hash_inventory_sha256 = canonical_sha256(state_hash_inventory)
    (
        proposal_coverage,
        proposal_generality,
        proposal_factor_effects,
        proposal_coverage_summary,
    ) = _proposal_coverage_frames(ordered_records)
    support_pairs = []
    frame = []
    records_by_id = {
        record["candidate_id"]: record for record in ordered_records
    }
    for record in ordered_records:
        spec = candidate_spec(record["candidate_id"])
        frame.append(
            {
                **{
                    key: value
                    for key, value in spec.as_dict().items()
                    if key not in {"family_id", "support_pair_id"}
                },
                "family_id": spec.family_id,
                "support_pair_id": spec.support_pair_id,
                "state_sha256": record["state_sha256"],
                "scripted_path_distance_m": record["root_geometry"][
                    "scripted_path_distance_m"
                ],
                "support_pair_separation_m": record["root_geometry"][
                    "support_pair_separation_m"
                ],
                "selected_transverse_rotation_degrees": record[
                    "root_geometry"
                ]["selected_transverse_rotation_degrees"],
                "selected_planned_recovery_mismatch": record[
                    "root_geometry"
                ]["selected_planned_recovery_mismatch"],
                "drawer_goal_distance_m": record["root_geometry"][
                    "realized_goal_distances_m"
                ]["drawer"],
                "cabinet_goal_distance_m": record["root_geometry"][
                    "realized_goal_distances_m"
                ]["cabinet"],
                "planned_recovery_distance_m": record["root_geometry"][
                    "planned_recovery_distance_m"
                ],
                "realized_recovery_distance_m": record["root_geometry"][
                    "realized_recovery_distance_m"
                ],
                "drawer_oracle_steps": record["oracles"]["drawer"]["cost"][
                    "budgeted_action_steps"
                ],
                "cabinet_oracle_steps": record["oracles"]["cabinet"]["cost"][
                    "budgeted_action_steps"
                ],
                **{
                    f"{goal}_oracle_{field}": record["oracles"][goal]["cost"][
                        source
                    ]
                    for goal in GOALS
                    for field, source in (
                        ("executed_steps", "executed_action_steps"),
                        ("active_steps", "active_servo_steps"),
                        ("eef_path_m", "eef_path_length_m"),
                        ("motion_effort", "motion_control_effort"),
                    )
                },
                **{
                    f"{goal}_oracle_{field}": record["oracles"][goal][source]
                    for goal in GOALS
                    for field, source in (
                        ("selected_proposal_index", "selected_proposal_index"),
                        ("proposal_attempt_count", "proposal_attempt_count"),
                        ("proposal_success_count", "proposal_success_count"),
                        (
                            "proposal_success_fraction",
                            "proposal_success_fraction",
                        ),
                        ("proposal_bank_sha256", "proposal_bank_sha256"),
                        (
                            "total_attempted_action_steps",
                            "total_attempted_action_steps",
                        ),
                    )
                },
                **{
                    f"{goal}_oracle_selected_episode_index": record["oracles"][
                        goal
                    ]["demo_episode_index"]
                    for goal in GOALS
                },
                "certificate_pass": record["certificate"]["pass"],
                "root_pass": record["root_validation"]["pass"],
                "joint_support_distance": record["support_measurement"][
                    "nearest"
                ]["distance"],
                "event_matched_support_distance": (
                    record["support_measurement"]["event_matched_nearest"][
                        "distance"
                    ]
                    if record["support_measurement"]["event_matched_nearest"]
                    is not None
                    else None
                ),
                "event_matching_reference_count": record[
                    "support_measurement"
                ]["event_matching_reference_count"],
                "nearest_anchor_locus": record["support_measurement"][
                    "locus_semantics"
                ]["nearest_anchor_locus"],
                "nearest_anchor_matches_locked_route": record[
                    "support_measurement"
                ]["locus_semantics"]["nearest_anchor_matches_locked_route"],
            }
        )
    table = pd.DataFrame(frame).sort_values("candidate_id")
    for pair_id, pair in table.groupby("support_pair_id", sort=True):
        if len(pair) != 2:
            raise ValueError(f"Compact report found incomplete support pair {pair_id}")
        near = pair[pair["support_stratum"] == "demonstration_near"].iloc[0]
        low = pair[pair["support_stratum"] == "transverse_low_support"].iloc[0]
        near_record = records_by_id[str(near["candidate_id"])]
        low_record = records_by_id[str(low["candidate_id"])]
        pair_metrics = validate_support_pair_records(
            near_record,
            low_record,
            **_validation_limits(config),
        )
        matched_cost_differences = {}
        matched_proposal_episode_indices = {}
        for goal, proposal_index in zip(
            GOALS,
            pair_metrics["matched_cost_proposal_indices"],
            strict=True,
        ):
            if proposal_index is None:
                matched_proposal_episode_indices[goal] = None
                for field in (
                    "executed_steps",
                    "active_steps",
                    "eef_path_m",
                    "motion_effort",
                ):
                    matched_cost_differences[(goal, field)] = None
                continue
            near_attempt = near_record["oracles"][goal]["proposal_attempts"][
                proposal_index
            ]
            low_attempt = low_record["oracles"][goal]["proposal_attempts"][
                proposal_index
            ]
            matched_proposal_episode_indices[goal] = int(
                near_attempt["episode_index"]
            )
            for field, source in (
                ("executed_steps", "executed_action_steps"),
                ("active_steps", "active_servo_steps"),
                ("eef_path_m", "eef_path_length_m"),
                ("motion_effort", "motion_control_effort"),
            ):
                matched_cost_differences[(goal, field)] = float(
                    low_attempt["cost"][source]
                    - near_attempt["cost"][source]
                )
        support_pairs.append(
            {
                "support_pair_id": pair_id,
                "support_pair_separation_m": float(
                    pair["support_pair_separation_m"].mean()
                ),
                "selected_transverse_rotation_degrees": float(
                    pair["selected_transverse_rotation_degrees"].mean()
                ),
                "controlled_perturbation_m": float(
                    low["scripted_path_distance_m"]
                    - near["scripted_path_distance_m"]
                ),
                "drawer_goal_distance_abs_difference_m": float(
                    abs(low["drawer_goal_distance_m"] - near["drawer_goal_distance_m"])
                ),
                "cabinet_goal_distance_abs_difference_m": float(
                    abs(low["cabinet_goal_distance_m"] - near["cabinet_goal_distance_m"])
                ),
                "planned_recovery_distance_abs_difference_m": float(
                    abs(
                        low["planned_recovery_distance_m"]
                        - near["planned_recovery_distance_m"]
                    )
                ),
                "realized_recovery_distance_abs_difference_m": float(
                    abs(
                        low["realized_recovery_distance_m"]
                        - near["realized_recovery_distance_m"]
                    )
                ),
                "drawer_selected_oracle_budgeted_step_difference": int(
                    low["drawer_oracle_steps"] - near["drawer_oracle_steps"]
                ),
                "cabinet_selected_oracle_budgeted_step_difference": int(
                    low["cabinet_oracle_steps"] - near["cabinet_oracle_steps"]
                ),
                "joint_support_distance_difference": float(
                    low["joint_support_distance"] - near["joint_support_distance"]
                ),
                **{
                    f"{goal}_same_selected_proposal": bool(
                        near[f"{goal}_oracle_selected_proposal_index"]
                        == low[f"{goal}_oracle_selected_proposal_index"]
                    )
                    for goal in GOALS
                },
                **{
                    f"{goal}_near_selected_episode_index": int(
                        near[f"{goal}_oracle_selected_episode_index"]
                    )
                    for goal in GOALS
                },
                **{
                    f"{goal}_low_selected_episode_index": int(
                        low[f"{goal}_oracle_selected_episode_index"]
                    )
                    for goal in GOALS
                },
                **{
                    f"{goal}_matched_cost_episode_index": (
                        matched_proposal_episode_indices[goal]
                    )
                    for goal in GOALS
                },
                **{
                    f"{goal}_shared_success_count": int(count)
                    for goal, count in zip(
                        GOALS,
                        pair_metrics["shared_success_counts"],
                        strict=True,
                    )
                },
                **{
                    f"{goal}_success_set_jaccard": float(jaccard)
                    for goal, jaccard in zip(
                        GOALS,
                        pair_metrics["success_set_jaccards"],
                        strict=True,
                    )
                },
                "measured_support_direction_matches_label": bool(
                    low["joint_support_distance"] > near["joint_support_distance"]
                ),
                "event_matched_support_distance_difference": (
                    float(
                        low["event_matched_support_distance"]
                        - near["event_matched_support_distance"]
                    )
                    if pd.notna(low["event_matched_support_distance"])
                    and pd.notna(near["event_matched_support_distance"])
                    else None
                ),
                **{
                    f"{goal}_selected_oracle_{field}_difference": float(
                        low[f"{goal}_oracle_{field}"]
                        - near[f"{goal}_oracle_{field}"]
                    )
                    for goal in GOALS
                    for field in (
                        "executed_steps",
                        "active_steps",
                        "eef_path_m",
                        "motion_effort",
                    )
                },
                **{
                    f"{goal}_matched_oracle_{field}_difference": (
                        matched_cost_differences[(goal, field)]
                    )
                    for goal in GOALS
                    for field in (
                        "executed_steps",
                        "active_steps",
                        "eef_path_m",
                        "motion_effort",
                    )
                },
            }
        )
    summary["support_pairs"] = support_pairs
    summary["exact_event_supported_candidate_count"] = int(
        (table["event_matching_reference_count"] > 0).sum()
    )
    summary["exact_event_supported_candidate_fraction"] = float(
        (table["event_matching_reference_count"] > 0).mean()
    )
    summary["exact_event_supported_pair_count"] = int(
        sum(
            1
            for _, pair in table.groupby("support_pair_id")
            if bool((pair["event_matching_reference_count"] > 0).all())
        )
    )
    summary["construction_revision"] = contract["construction_revision"]
    summary["demonstration_action_sha256_by_role"] = {
        role: values["action_sha256"]
        for role, values in contract["demonstrations"].items()
    }
    summary["oracle_proposal_bank"] = contract["oracle_proposal_bank"]
    summary["proposal_coverage_summary_by_goal"] = proposal_coverage_summary
    summary["candidate_records_sha256"] = candidate_records_sha256
    summary["state_hash_inventory_sha256"] = state_hash_inventory_sha256
    summary["scientific_boundary"] = (
        "Policy-independent physical feasibility and deterministic restoration only; "
        "no VLA behaviour or causal hidden-state mechanism was evaluated."
    )

    report_dir = PROJECT / "reports/phase3b_stage_a" / run_dir.name
    if report_dir.exists():
        existing_manifest_path = report_dir / "manifest.json"
        if existing_manifest_path.is_file():
            existing_manifest = json.loads(existing_manifest_path.read_text())
            if (
                existing_manifest.get("status") == "complete"
                and existing_manifest.get("contract_sha256")
                == manifest["contract_sha256"]
                and existing_manifest.get("candidate_count") == len(records)
            ):
                return report_dir
        raise FileExistsError(f"Refusing to overwrite compact report {report_dir}")
    staging_dir = report_dir.with_name(
        f".{report_dir.name}__tmp__pid{os.getpid()}"
    )
    if staging_dir.exists():
        raise FileExistsError(f"Stale compact-report staging path: {staging_dir}")
    staging_dir.mkdir(parents=True)
    atomic_write_json(staging_dir / "summary.json", summary)
    atomic_write_json(staging_dir / "contract.json", contract)
    atomic_write_json(staging_dir / "selection_lock.json", selection_lock)
    atomic_write_json(staging_dir / "candidates.json", ordered_records)
    table.to_csv(staging_dir / "candidate_summary.csv", index=False)
    pd.DataFrame(support_pairs).to_csv(
        staging_dir / "support_pair_balance.csv", index=False
    )
    proposal_coverage.to_csv(staging_dir / "proposal_coverage.csv", index=False)
    proposal_generality.to_csv(
        staging_dir / "proposal_generality.csv", index=False
    )
    proposal_factor_effects.to_csv(
        staging_dir / "proposal_coverage_factor_effects.csv", index=False
    )
    (staging_dir / "README.md").write_text(
        "# Phase 3b Stage A\n\n"
        "This compact report certifies the locked 32-cell, policy-independent "
        "LIBERO affordance lattice. Every root has both goals false, both fixed "
        "human-demonstration continuations feasible, and an exact repeated-action "
        "computational-state certificate. No SmolVLA checkpoint or branch was loaded.\n\n"
        "Demonstration support is measured jointly over robot pose, object and "
        "drawer geometry, discrete events, and motion against the locked replay "
        "bank; the construction label is not treated as empirical support by fiat.\n\n"
        "Goal feasibility exhaustively evaluates the complete locked drawer/cabinet "
        "human-demonstration inventories from one shared normalized root. Proposal "
        "success fractions, deterministic minimum-cost selections, pairwise "
        "success-set overlap, and disjoint proposal basins are reported explicitly "
        "rather than folded into physical difficulty.\n\n"
        "`proposal_coverage.csv` preserves every candidate-by-proposal outcome; "
        "the generality and factor-effect tables expose whether feasibility is "
        "carried by a narrow trajectory template.\n\n"
        "Exact event-support coverage and any reversals of the geometric support "
        "label are reported explicitly; unsupported cells do not justify a causal "
        "occupancy claim.\n\n"
        "Raw MuJoCo states remain workstation-only under `local/phase3b_stage_a`; "
        "this directory contains only Git-safe manifests, hashes, certificates, "
        "geometry, and aggregate costs.\n"
    )
    artifact_sha256 = {
        path.name: _file_sha256(path)
        for path in sorted(staging_dir.iterdir())
        if path.is_file()
    }
    atomic_write_json(
        staging_dir / "manifest.json",
        {
            "schema_version": 1,
            "run_id": run_dir.name,
            "status": "complete",
            "contract_sha256": manifest["contract_sha256"],
            "selection_lock_sha256": manifest["selection_lock_sha256"],
            "candidate_count": len(ordered_records),
            "candidate_records_sha256": candidate_records_sha256,
            "state_hash_inventory_sha256": state_hash_inventory_sha256,
            "artifact_sha256": artifact_sha256,
            "policy_loaded": False,
            "support_reference_bank_sha256": summary[
                "support_reference_bank_sha256"
            ],
            "raw_evidence_location": "workstation-only local/phase3b_stage_a",
        },
    )
    os.replace(staging_dir, report_dir)
    return report_dir


def _audit_candidate(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    manifest: dict[str, Any],
    selection_lock: dict[str, Any],
    demos: dict[str, Any],
    config: dict[str, Any],
) -> None:
    spec = candidate_spec(args.audit_candidate_id)
    expected = _completed_record(
        run_dir,
        spec.candidate_id,
        contract_sha256=manifest["contract_sha256"],
        selection_lock_sha256=selection_lock["selection_lock_sha256"],
    )
    if expected is None:
        raise ValueError(f"Cannot audit incomplete candidate {spec.candidate_id}")
    bank_environment = make_stage_a_environment(PROJECT, run_dir, config)
    try:
        support_bank = build_support_reference_bank(
            bank_environment, _oracle_demos(demos), config
        )
    finally:
        bank_environment.close()
    environment = make_stage_a_environment(PROJECT, run_dir, config)
    try:
        reconstructed = construct_candidate(
            environment,
            spec,
            demos,
            config,
            support_reference_bank=support_bank,
        )
        actual_hash = snapshot_sha256(reconstructed.snapshot)
        certificate = certify_computational_state(
            environment,
            reconstructed.snapshot,
            possession=spec.possession,
            probe_actions=config["certificate"]["actions"],
        )
        construction_match = canonical_sha256(reconstructed.construction) == canonical_sha256(
            expected["construction"]
        )
        root_geometry_match = canonical_sha256(reconstructed.root_geometry) == canonical_sha256(
            expected["root_geometry"]
        )
        route_provenance_match = bool(
            reconstructed.construction["root_transit_action_count"]
            == expected["construction"]["root_transit_action_count"]
            and reconstructed.construction["root_transit_action_sha256"]
            == expected["construction"]["root_transit_action_sha256"]
        )
        support_bank_match = bool(
            support_bank.sha256
            == expected["support_measurement"]["reference_bank_sha256"]
        )
        support_measurement_match = canonical_sha256(
            reconstructed.support_measurement
        ) == canonical_sha256(expected["support_measurement"])
        audit = {
            "schema_version": 1,
            "candidate_id": spec.candidate_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "expected_state_sha256": expected["state_sha256"],
            "reconstructed_state_sha256": actual_hash,
            "state_hash_match": actual_hash == expected["state_sha256"],
            "support_reference_bank_sha256": support_bank.sha256,
            "support_reference_bank_match": support_bank_match,
            "support_measurement_match": support_measurement_match,
            "construction_match": construction_match,
            "root_geometry_match": root_geometry_match,
            "route_provenance_match": route_provenance_match,
            "certificate": certificate,
            "pass": bool(
                actual_hash == expected["state_sha256"]
                and certificate["pass"]
                and support_bank_match
                and support_measurement_match
                and construction_match
                and root_geometry_match
                and route_provenance_match
            ),
        }
        name = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        atomic_write_json(
            run_dir / "audits" / f"{spec.candidate_id}__{name}.json", audit
        )
        print(json.dumps(audit, indent=2, sort_keys=True))
        if not audit["pass"]:
            raise RuntimeError(f"Fresh-process reconstruction failed: {audit}")
    finally:
        environment.close()


def main() -> None:
    args = _parse_args()
    if args.stop_after_candidates is not None and args.stop_after_candidates < 1:
        raise ValueError("--stop-after-candidates must be positive")
    config_path = args.config.resolve()
    config = _load_config(config_path)
    demos = _load_demos(config)
    proposal_bank = _load_proposal_bank(config)
    requested_goals = tuple(
        goal for goal in GOALS if not args.oracle_goal or goal in args.oracle_goal
    )
    minimum_horizon = _minimum_native_horizon(config, demos, proposal_bank)
    if int(config["environment"]["episode_length"]) < minimum_horizon:
        raise ValueError(
            "Stage A episode_length is below the worst-case fixed oracle budget: "
            f"{config['environment']['episode_length']} < {minimum_horizon}"
        )
    contract = _contract(config_path, config, demos, proposal_bank)
    contract_sha = canonical_sha256(contract)
    selection_lock = build_selection_lock(
        contract_sha256=contract_sha,
        construction_revision=config["construction_revision"],
    )
    run_dir = _run_directory(args)
    manifest = _initialize_run(run_dir, contract, selection_lock)
    _validate_run_inventory(run_dir)
    print(f"Stage A run directory: {run_dir}")

    if args.audit_candidate_id:
        _audit_candidate(
            args=args,
            run_dir=run_dir,
            manifest=manifest,
            selection_lock=selection_lock,
            demos=demos,
            config=config,
        )
        return

    specs = iter_candidate_specs()
    if args.candidate_id:
        requested = set(args.candidate_id)
        unknown = requested - {spec.candidate_id for spec in specs}
        if unknown:
            raise ValueError(f"Unknown requested candidate IDs: {sorted(unknown)}")
        specs = tuple(spec for spec in specs if spec.candidate_id in requested)

    completed_count = sum(
        _completed_record(
            run_dir,
            existing_spec.candidate_id,
            contract_sha256=contract_sha,
            selection_lock_sha256=selection_lock["selection_lock_sha256"],
        )
        is not None
        for existing_spec in iter_candidate_specs()
    )
    support_bank = None
    action_phase_banks: dict[tuple[str, str], tuple[Any, ...]] = {}
    action_phase_bank_hashes: dict[str, dict[str, str]] = {
        goal: {} for goal in GOALS
    }
    for spec in specs:
        existing = _completed_record(
            run_dir,
            spec.candidate_id,
            contract_sha256=contract_sha,
            selection_lock_sha256=selection_lock["selection_lock_sha256"],
        )
        if existing is not None:
            print(f"skip complete {spec.candidate_id}")
            continue
        if (
            args.stop_after_candidates is not None
            and completed_count >= args.stop_after_candidates
        ):
            break
        if support_bank is None:
            try:
                bank_environment = make_stage_a_environment(
                    PROJECT, run_dir, config
                )
                try:
                    support_bank = build_support_reference_bank(
                        bank_environment, _oracle_demos(demos), config
                    )
                finally:
                    bank_environment.close()
            except Exception as exc:
                _candidate_error(run_dir, spec.candidate_id, exc)
                _update_manifest(
                    run_dir,
                    manifest,
                    status="failed",
                    candidate_count=completed_count,
                    failed_candidate=spec.candidate_id,
                    failure_stage="support_reference_bank",
                    failure_type=type(exc).__name__,
                    failure_message=str(exc),
                )
                raise
            print(
                "support reference bank "
                f"{support_bank.sha256} ({len(support_bank.entries)} states)",
                flush=True,
            )
        for goal in requested_goals:
            phase_key = (goal, spec.layout)
            if phase_key in action_phase_banks:
                continue
            try:
                phase_environment = make_stage_a_environment(
                    PROJECT, run_dir, config
                )
                try:
                    action_phase_banks[phase_key] = (
                        build_action_phase_proposal_bank(
                            phase_environment,
                            layout=spec.layout,
                            proposals=proposal_bank[goal],
                            config=config,
                        )
                    )
                finally:
                    phase_environment.close()
                action_phase_bank_hashes[goal][spec.layout] = canonical_sha256(
                    [
                        proposal.metadata
                        for proposal in action_phase_banks[phase_key]
                    ]
                )
            except Exception as exc:
                _candidate_error(run_dir, spec.candidate_id, exc)
                _update_manifest(
                    run_dir,
                    manifest,
                    status="failed",
                    candidate_count=completed_count,
                    failed_candidate=spec.candidate_id,
                    failure_stage=f"{goal}_action_phase_proposal_bank",
                    failure_type=type(exc).__name__,
                    failure_message=str(exc),
                )
                raise
            print(
                f"{goal} action-phase bank layout {spec.layout} "
                f"{action_phase_bank_hashes[goal][spec.layout]} "
                f"({len(action_phase_banks[phase_key])} proposals)",
                flush=True,
            )
        print(f"construct {spec.candidate_id}", flush=True)
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
            if not certificate["pass"]:
                raise RuntimeError(
                    f"Certificate failed for {spec.candidate_id}: {certificate}"
                )
            oracles = {}
            for goal in requested_goals:
                completed_results, record_result, finish_checkpoint = (
                    _oracle_checkpoint(
                        run_dir,
                        candidate_id=spec.candidate_id,
                        goal=goal,
                        root_state_sha256=snapshot_sha256(
                            constructed.snapshot
                        ),
                        contract_sha256=contract_sha,
                        selection_lock_sha256=selection_lock[
                            "selection_lock_sha256"
                        ],
                        proposals=proposal_bank[goal],
                        phase_proposals=action_phase_banks[
                            (goal, spec.layout)
                        ],
                    )
                )
                oracles[goal] = run_goal_oracle_bank(
                    environment,
                    constructed.snapshot,
                    spec=spec,
                    goal=goal,
                    proposals=proposal_bank[goal],
                    initial_bowl_position=constructed.initial_bowl_position,
                    initial_eef_position=constructed.initial_eef_position,
                    initial_eef_orientation=constructed.initial_eef_orientation,
                    initial_joint_positions=constructed.initial_joint_positions,
                    recovery_waypoints=constructed.recovery_waypoints,
                    config=config,
                    action_phase_proposals=action_phase_banks[
                        (goal, spec.layout)
                    ],
                    completed_results=completed_results,
                    result_callback=record_result,
                )
                finish_checkpoint(oracles[goal])
                print(
                    f"complete {goal} oracle for {spec.candidate_id}: "
                    f"{oracles[goal]['proposal_success_count']}/"
                    f"{oracles[goal]['proposal_attempt_count']}",
                    flush=True,
                )
            if set(requested_goals) != set(GOALS):
                _update_manifest(
                    run_dir,
                    manifest,
                    status="in_progress",
                    candidate_count=completed_count,
                    last_checkpointed_candidate=spec.candidate_id,
                    last_checkpointed_goals=list(requested_goals),
                    action_phase_proposal_bank_sha256_by_goal_layout=(
                        action_phase_bank_hashes
                    ),
                )
                print(
                    f"checkpoint-only smoke complete for {spec.candidate_id}: "
                    f"goals={list(requested_goals)}",
                    flush=True,
                )
                continue
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
                pair_metrics = validate_support_pair_records(
                    near,
                    low,
                    **_validation_limits(config),
                )
                print(
                    "support pair gate "
                    f"{spec.support_pair_id}: "
                    f"support_delta={pair_metrics['support_distance_difference']:.6f}",
                    flush=True,
                )
            _persist_candidate_state(
                run_dir,
                spec.candidate_id,
                constructed.snapshot,
                record["state_sha256"],
            )
            atomic_write_json(
                run_dir / "candidates" / f"{spec.candidate_id}.json", record
            )
            completed_count += 1
            _update_manifest(
                run_dir,
                manifest,
                status="in_progress",
                candidate_count=completed_count,
                last_completed_candidate=spec.candidate_id,
                support_reference_bank_sha256=support_bank.sha256,
                action_phase_proposal_bank_sha256_by_goal_layout=(
                    action_phase_bank_hashes
                ),
            )
            print(f"complete {spec.candidate_id}", flush=True)
        except Exception as exc:
            _candidate_error(run_dir, spec.candidate_id, exc)
            _update_manifest(
                run_dir,
                manifest,
                status="failed",
                candidate_count=completed_count,
                failed_candidate=spec.candidate_id,
                failure_type=type(exc).__name__,
                failure_message=str(exc),
            )
            raise
        finally:
            environment.close()

    all_records = []
    for spec in iter_candidate_specs():
        record = _completed_record(
            run_dir,
            spec.candidate_id,
            contract_sha256=contract_sha,
            selection_lock_sha256=selection_lock["selection_lock_sha256"],
        )
        if record is not None:
            all_records.append(record)
    if len(all_records) == 32:
        ordered_records = sorted(
            all_records, key=lambda record: record["candidate_id"]
        )
        candidate_records_sha256 = canonical_sha256(ordered_records)
        state_hash_inventory_sha256 = canonical_sha256(
            [
                {
                    "candidate_id": record["candidate_id"],
                    "state_sha256": record["state_sha256"],
                }
                for record in ordered_records
            ]
        )
        report_dir = _compact_report(
            run_dir,
            manifest,
            contract,
            selection_lock,
            all_records,
            config,
        )
        _update_manifest(
            run_dir,
            manifest,
            status="complete",
            candidate_count=32,
            candidate_records_sha256=candidate_records_sha256,
            state_hash_inventory_sha256=state_hash_inventory_sha256,
            compact_report=report_dir.relative_to(PROJECT).as_posix(),
            completed_at=datetime.now(UTC).isoformat(),
        )
        print(f"Stage A complete: {report_dir}")
    else:
        _update_manifest(
            run_dir,
            manifest,
            status="in_progress",
            candidate_count=len(all_records),
        )
        print(f"Stage A paused with {len(all_records)}/32 candidates")


if __name__ == "__main__":
    main()
