#!/usr/bin/env python
"""Causally decompose the matched v37 normalized-root outcome reversal."""

from __future__ import annotations

import argparse
import json
import traceback
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from diagnose_phase3b_normalized_state import (
        DEFAULT_CONFIG,
        DEFAULT_SOURCE_RUN,
        LOW_CANDIDATE,
        NEAR_CANDIDATE,
        PROPOSAL_EPISODE,
        PROPOSAL_INDEX,
        _checkpoint_evidence,
        _state_descriptor,
        reconstruct_normalized_candidate,
    )
    from run_phase3b_stage_a import (
        PROJECT,
        _file_sha256,
        _load_config,
        _load_demos,
        _load_proposal_bank,
    )
except ModuleNotFoundError:
    from scripts.diagnose_phase3b_normalized_state import (
        DEFAULT_CONFIG,
        DEFAULT_SOURCE_RUN,
        LOW_CANDIDATE,
        NEAR_CANDIDATE,
        PROPOSAL_EPISODE,
        PROPOSAL_INDEX,
        _checkpoint_evidence,
        _state_descriptor,
        reconstruct_normalized_candidate,
    )
    from scripts.run_phase3b_stage_a import (
        PROJECT,
        _file_sha256,
        _load_config,
        _load_demos,
        _load_proposal_bank,
    )
from smolvla_analysis.libero_state import (
    capture_libero_state,
    restore_libero_state,
)
from smolvla_analysis.phase3_crd import atomic_write_json, evaluate_common_goals
from smolvla_analysis.phase3b_libero import (
    BOWL_NAME,
    PolicyFreeController,
    build_landmark_registered_action_phase_proposal_bank,
    build_support_reference_bank,
    make_stage_a_environment,
    run_action_phase_oracle_from_prepared_root,
)
from smolvla_analysis.phase3b_stage_a import (
    canonical_sha256,
    candidate_spec,
    snapshot_sha256,
)


RAW_ROOT = PROJECT / "local/phase3b_stage_a/state_swap_diagnostics"
DIAGNOSTIC_REVISION = "phase3b-normalized-state-swap-v1"
CONDITIONS = (
    "near_physics_low_runtime",
    "low_physics_near_runtime",
    "low_plus_near_bowl_orientation",
    "low_plus_near_robot_state",
    "low_plus_near_bowl_orientation_and_robot_state",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run five predeclared normalized-state interventions without "
            "replaying either completed baseline."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _output_directory(requested: Path | None) -> Path:
    if requested is None:
        stamp = datetime.now(UTC).strftime("state_swap_%Y%m%dT%H%M%SZ")
        requested = RAW_ROOT / stamp
    output = requested.resolve()
    raw_root = RAW_ROOT.resolve()
    if output == raw_root or raw_root not in output.parents:
        raise ValueError(f"State-swap diagnostic must remain under {raw_root}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic: {output}")
    output.mkdir(parents=True)
    return output


def _runtime_cross(physics: Any, runtime: Any) -> Any:
    """Use one physical state with the other's non-MuJoCo runtime state."""

    return replace(
        physics,
        runtime_state=deepcopy(runtime.runtime_state),
    )


def _address_slice(address: Any) -> slice:
    if isinstance(address, tuple):
        return slice(*address)
    return slice(int(address), int(address) + 1)


def _partial_near_splice(
    environment: Any,
    *,
    low: Any,
    near: Any,
    bowl_orientation: bool,
    robot_state: bool,
) -> Any:
    """Replace declared low-state blocks with their matched near values."""

    restore_libero_state(environment, near)
    near_controller = PolicyFreeController(environment)
    near_problem = near_controller.problem
    near_robot = near_problem.robots[0]
    bowl_joint = tuple(getattr(near_problem.get_object(BOWL_NAME), "joints", ()))
    if len(bowl_joint) != 1:
        raise ValueError(f"Expected one bowl joint, found {bowl_joint}")
    bowl_qpos_slice = _address_slice(
        near_problem.sim.model.get_joint_qpos_addr(bowl_joint[0])
    )
    near_bowl_qpos = np.asarray(
        near_problem.sim.data.qpos[bowl_qpos_slice], dtype=np.float64
    ).copy()
    near_robot_qpos = near_controller.joint_positions()
    near_robot_qvel = np.asarray(
        near_problem.sim.data.qvel[near_robot._ref_joint_vel_indexes],
        dtype=np.float64,
    ).copy()
    near_gripper_qpos = np.asarray(
        near_problem.sim.data.qpos[near_robot._ref_gripper_joint_pos_indexes],
        dtype=np.float64,
    ).copy()
    near_gripper_qvel = np.asarray(
        near_problem.sim.data.qvel[near_robot._ref_gripper_joint_vel_indexes],
        dtype=np.float64,
    ).copy()

    restore_libero_state(environment, low)
    low_controller = PolicyFreeController(environment)
    problem = low_controller.problem
    robot = problem.robots[0]
    if bowl_orientation:
        low_bowl_qpos = np.asarray(
            problem.sim.data.qpos[bowl_qpos_slice], dtype=np.float64
        ).copy()
        if low_bowl_qpos.shape != (7,) or near_bowl_qpos.shape != (7,):
            raise ValueError("Bowl free-joint state changed shape")
        low_bowl_qpos[3:] = near_bowl_qpos[3:]
        problem.sim.data.qpos[bowl_qpos_slice] = low_bowl_qpos
    if robot_state:
        problem.sim.data.qpos[robot._ref_joint_pos_indexes] = near_robot_qpos
        problem.sim.data.qvel[robot._ref_joint_vel_indexes] = near_robot_qvel
        problem.sim.data.qpos[
            robot._ref_gripper_joint_pos_indexes
        ] = near_gripper_qpos
        problem.sim.data.qvel[
            robot._ref_gripper_joint_vel_indexes
        ] = near_gripper_qvel
    problem.sim.forward()
    hybrid = capture_libero_state(environment)
    if robot_state:
        runtime = deepcopy(hybrid.runtime_state)
        runtime["robots"] = deepcopy(near.runtime_state.get("robots", []))
        near_sim_data = near.runtime_state.get("sim_data", {})
        for field in ("act", "ctrl"):
            if field in near_sim_data:
                runtime.setdefault("sim_data", {})[field] = deepcopy(
                    near_sim_data[field]
                )
        hybrid = replace(hybrid, runtime_state=runtime)
    return hybrid


def build_state_swap_conditions(
    environment: Any, *, near: Any, low: Any
) -> dict[str, Any]:
    """Build the fixed intervention lattice without executing a proposal."""

    conditions = {
        "near_physics_low_runtime": _runtime_cross(near, low),
        "low_physics_near_runtime": _runtime_cross(low, near),
        "low_plus_near_bowl_orientation": _partial_near_splice(
            environment,
            low=low,
            near=near,
            bowl_orientation=True,
            robot_state=False,
        ),
        "low_plus_near_robot_state": _partial_near_splice(
            environment,
            low=low,
            near=near,
            bowl_orientation=False,
            robot_state=True,
        ),
        "low_plus_near_bowl_orientation_and_robot_state": (
            _partial_near_splice(
                environment,
                low=low,
                near=near,
                bowl_orientation=True,
                robot_state=True,
            )
        ),
    }
    if tuple(conditions) != CONDITIONS:
        raise RuntimeError("State-swap condition order changed")
    baseline_hashes = {snapshot_sha256(near), snapshot_sha256(low)}
    condition_hashes = [snapshot_sha256(conditions[name]) for name in CONDITIONS]
    if len(set(condition_hashes)) != len(CONDITIONS) or any(
        value in baseline_hashes for value in condition_hashes
    ):
        raise RuntimeError("State-swap intervention duplicated a baseline or branch")
    return conditions


def _validate_intervention(
    environment: Any,
    snapshot: Any,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    restore_libero_state(environment, snapshot)
    controller = PolicyFreeController(environment)
    goals = evaluate_common_goals(environment)
    descriptor = _state_descriptor(environment, snapshot)
    minimum_contact_distance = min(
        (float(item["distance"]) for item in snapshot.contacts),
        default=0.0,
    )
    checks = {
        "no_goal": not any(goals.values()),
        "bowl_released": not controller.bowl_grasped(),
        "drawer_open": controller.top_drawer_position()
        <= float(config["construction"]["open_drawer_threshold"]),
        "penetration_within_limit": minimum_contact_distance
        >= -float(config["validation"]["penetration_limit_m"]),
        "finite_mujoco_state": bool(np.isfinite(snapshot.mujoco_state).all()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Invalid state-swap intervention: {checks}")
    return {
        "pass": True,
        "checks": checks,
        "minimum_contact_distance_m": minimum_contact_distance,
        "descriptor": descriptor,
    }


def _outcome_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "pass": bool(result["pass"]),
        "goal_ever_achieved": bool(result["goal_ever_achieved"]),
        "first_goal_demo_frame": result["first_goal_demo_frame"],
        "wrong_goal_ever_achieved": bool(result["wrong_goal_ever_achieved"]),
        "unexpected_done_before_goal": bool(
            result["unexpected_done_before_goal"]
        ),
        "final_goals": result["final_goals"],
        "bridge_pass": bool(result["phases"]["action_phase_bridge"]["pass"]),
        "normalized_state_sha256": result["normalized_state_sha256"],
        "action_sha256": result["cost"]["action_sha256"],
        "executed_action_steps": result["cost"]["executed_action_steps"],
    }


def main() -> None:
    args = _parse_args()
    output_dir = _output_directory(args.output_dir)
    config_path = args.config.resolve()
    source_run = args.source_run.resolve()
    config = _load_config(config_path)
    demos = _load_demos(config)
    proposal_banks = _load_proposal_bank(config)
    proposal = proposal_banks["cabinet"][PROPOSAL_INDEX]
    if proposal.episode_index != PROPOSAL_EPISODE:
        raise ValueError("Locked proposal index no longer maps to episode 474")
    evidence = {
        "near": _checkpoint_evidence(source_run, NEAR_CANDIDATE),
        "low": _checkpoint_evidence(source_run, LOW_CANDIDATE),
    }
    source_contract_path = source_run / "contract.json"
    source_manifest_path = source_run / "manifest.json"
    source_contract = json.loads(source_contract_path.read_text())
    source_manifest = json.loads(source_manifest_path.read_text())
    source_contract_sha = canonical_sha256(source_contract)
    if (
        source_contract_sha != evidence["near"]["contract_sha256"]
        or source_contract_sha != evidence["low"]["contract_sha256"]
    ):
        raise ValueError("Source run contract no longer matches its checkpoints")
    expected_support_bank_sha = source_manifest.get(
        "support_reference_bank_sha256"
    )
    if not isinstance(expected_support_bank_sha, str):
        raise ValueError("Source manifest has no support-bank identity")
    expected_baselines = {
        "near": {"pass": True, "goal_ever_achieved": True},
        "low": {"pass": False, "goal_ever_achieved": False},
    }
    for name, expected in expected_baselines.items():
        observed = {
            key: evidence[name]["completed_outcome"][key] for key in expected
        }
        if observed != expected:
            raise ValueError(f"Completed {name} baseline outcome changed")
    implementation_files = (
        Path(__file__).resolve(),
        PROJECT / "scripts/diagnose_phase3b_normalized_state.py",
        PROJECT / "scripts/run_phase3b_stage_a.py",
        PROJECT / "src/smolvla_analysis/phase3b_libero.py",
        PROJECT / "src/smolvla_analysis/libero_state.py",
        PROJECT / "src/smolvla_analysis/phase3b_stage_a.py",
    )
    contract = {
        "schema_version": 1,
        "diagnostic_revision": DIAGNOSTIC_REVISION,
        "config_path": config_path.relative_to(PROJECT).as_posix(),
        "config_file_sha256": _file_sha256(config_path),
        "source_run": source_run.relative_to(PROJECT).as_posix(),
        "source_contract_sha256": source_contract_sha,
        "source_artifact_sha256": {
            "contract.json": _file_sha256(source_contract_path),
            "manifest.json": _file_sha256(source_manifest_path),
        },
        "expected_support_reference_bank_sha256": expected_support_bank_sha,
        "candidate_evidence": evidence,
        "proposal_episode": PROPOSAL_EPISODE,
        "proposal_index": PROPOSAL_INDEX,
        "proposal_action_sha256": proposal.action_sha256,
        "conditions": list(CONDITIONS),
        "implementation_sha256": {
            path.relative_to(PROJECT).as_posix(): _file_sha256(path)
            for path in implementation_files
        },
        "execution_scope": {
            "completed_baselines_reexecuted": 0,
            "new_intervention_branches": len(CONDITIONS),
            "policy_forwards": 0,
        },
    }
    contract_sha = canonical_sha256(contract)
    atomic_write_json(output_dir / "contract.json", contract)
    atomic_write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "status": "in_progress",
            "created_at": datetime.now(UTC).isoformat(),
            "contract_sha256": contract_sha,
        },
    )
    try:
        bank_environment = make_stage_a_environment(PROJECT, output_dir, config)
        try:
            support_bank = build_support_reference_bank(
                bank_environment,
                {goal: demos[goal] for goal in ("drawer", "cabinet")},
                config,
            )
        finally:
            bank_environment.close()
        if support_bank.sha256 != expected_support_bank_sha:
            raise RuntimeError("Reconstructed support-reference bank changed")
        phase_environment = make_stage_a_environment(PROJECT, output_dir, config)
        try:
            phase_bank = build_landmark_registered_action_phase_proposal_bank(
                phase_environment,
                target_layout="A",
                proposals=proposal_banks["cabinet"],
                config=config,
            )
        finally:
            phase_environment.close()
        phase_proposal = phase_bank[PROPOSAL_INDEX]
        if phase_proposal.source.action_sha256 != proposal.action_sha256:
            raise ValueError("Action-phase proposal identity changed")
        acquisition_matches = [
            item
            for item in phase_bank
            if item.source.episode_index == PROPOSAL_EPISODE
        ]
        if len(acquisition_matches) != 1:
            raise ValueError("Registered acquisition identity changed")

        prepared_roots = {}
        reconstruction_payloads = {}
        for name, candidate_id in (
            ("near", NEAR_CANDIDATE),
            ("low", LOW_CANDIDATE),
        ):
            environment = make_stage_a_environment(PROJECT, output_dir, config)
            try:
                payload, prepared = reconstruct_normalized_candidate(
                    environment=environment,
                    candidate_id=candidate_id,
                    evidence=evidence[name],
                    demos=demos,
                    proposal=proposal,
                    config=config,
                    support_bank=support_bank,
                    registered_acquisition=acquisition_matches[0],
                )
            finally:
                environment.close()
            reconstruction_payloads[name] = payload
            prepared_roots[name] = prepared

        intervention_environment = make_stage_a_environment(
            PROJECT, output_dir, config
        )
        try:
            snapshots = build_state_swap_conditions(
                intervention_environment,
                near=prepared_roots["near"].snapshot,
                low=prepared_roots["low"].snapshot,
            )
            intervention_results = {}
            low_spec = candidate_spec(LOW_CANDIDATE)
            for condition in CONDITIONS:
                snapshot = snapshots[condition]
                validation = _validate_intervention(
                    intervention_environment, snapshot, config=config
                )
                prepared = replace(
                    prepared_roots["low"],
                    snapshot=snapshot,
                )
                result = run_action_phase_oracle_from_prepared_root(
                    intervention_environment,
                    prepared_roots["low"].snapshot,
                    prepared,
                    spec=low_spec,
                    proposal=phase_proposal,
                    config=config,
                )
                if result["normalized_state_sha256"] != snapshot_sha256(snapshot):
                    raise RuntimeError(f"Intervention hash changed for {condition}")
                intervention_results[condition] = {
                    "intervention_state_sha256": snapshot_sha256(snapshot),
                    "validation": validation,
                    "outcome": _outcome_summary(result),
                    "raw_result": result,
                }
                atomic_write_json(
                    output_dir / f"condition__{condition}.json",
                    intervention_results[condition],
                )
        finally:
            intervention_environment.close()

        result = {
            "schema_version": 1,
            "status": "complete",
            "diagnostic_revision": DIAGNOSTIC_REVISION,
            "contract_sha256": contract_sha,
            "support_reference_bank_sha256": support_bank.sha256,
            "policy_loaded": False,
            "completed_baselines_reexecuted": 0,
            "baseline_evidence": {
                name: evidence[name]["completed_outcome"]
                for name in ("near", "low")
            },
            "reconstruction": reconstruction_payloads,
            "interventions": {
                condition: {
                    key: value
                    for key, value in intervention_results[condition].items()
                    if key != "raw_result"
                }
                for condition in CONDITIONS
            },
            "scientific_boundary": (
                "Privileged simulator state swaps identify causal state-block "
                "dependence for this controller/proposal/root pair. They are "
                "diagnostic interventions, not feasible policy continuations "
                "or hidden-state mechanisms."
            ),
        }
        atomic_write_json(output_dir / "result.json", result)
        artifact_sha = {
            path.name: _file_sha256(path)
            for path in sorted(output_dir.iterdir())
            if path.is_file() and path.name != "manifest.json"
        }
        condition_pass = {
            condition: intervention_results[condition]["outcome"]["pass"]
            for condition in CONDITIONS
        }
        atomic_write_json(
            output_dir / "manifest.json",
            {
                "schema_version": 1,
                "status": "complete",
                "created_at": datetime.now(UTC).isoformat(),
                "contract_sha256": contract_sha,
                "artifact_sha256": artifact_sha,
                "condition_pass": condition_pass,
                "completed_baselines_reexecuted": 0,
                "policy_loaded": False,
            },
        )
        print(json.dumps(condition_pass, indent=2, sort_keys=True))
        print(f"State-swap diagnostic complete: {output_dir}")
    except Exception as exc:
        atomic_write_json(
            output_dir / "error.json",
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        atomic_write_json(
            output_dir / "manifest.json",
            {
                "schema_version": 1,
                "status": "failed",
                "created_at": datetime.now(UTC).isoformat(),
                "contract_sha256": contract_sha,
                "artifact_sha256": {
                    path.name: _file_sha256(path)
                    for path in sorted(output_dir.iterdir())
                    if path.is_file() and path.name != "manifest.json"
                },
                "completed_baselines_reexecuted": 0,
                "policy_loaded": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
