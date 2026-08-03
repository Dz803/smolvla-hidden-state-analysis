#!/usr/bin/env python
"""Compare two v37 normalized roots without replaying a goal proposal."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from run_phase3b_stage_a import (
        PROJECT,
        _file_sha256,
        _load_config,
        _load_demos,
        _load_proposal_bank,
    )
except ModuleNotFoundError:
    from scripts.run_phase3b_stage_a import (
        PROJECT,
        _file_sha256,
        _load_config,
        _load_demos,
        _load_proposal_bank,
    )
from smolvla_analysis.libero_state import restore_libero_state
from smolvla_analysis.phase3_crd import atomic_write_json
from smolvla_analysis.phase3b_libero import (
    BOWL_NAME,
    TOP_DRAWER_JOINT,
    PolicyFreeController,
    build_landmark_registered_action_phase_proposal_bank,
    build_support_reference_bank,
    certify_computational_state,
    construct_candidate,
    make_stage_a_environment,
    run_goal_oracle,
)
from smolvla_analysis.phase3b_normalized_state import (
    compare_normalized_descriptors,
)
from smolvla_analysis.phase3b_stage_a import (
    canonical_sha256,
    candidate_spec,
    snapshot_sha256,
)


DEFAULT_CONFIG = PROJECT / "configs/phase3b_stage_a_v37.yaml"
DEFAULT_SOURCE_RUN = (
    PROJECT
    / "local/phase3b_stage_a/"
    "phase3b_stage_a_completion_v37_20260803T101257Z"
)
RAW_ROOT = PROJECT / "local/phase3b_stage_a/normalized_state_diagnostics"
NEAR_CANDIDATE = (
    "stagea__drawer-open__possession-grasped__locus-cabinet-side__"
    "support-demonstration-near__layout-a"
)
LOW_CANDIDATE = (
    "stagea__drawer-open__possession-grasped__locus-cabinet-side__"
    "support-transverse-low-support__layout-a"
)
PROPOSAL_EPISODE = 474
PROPOSAL_INDEX = 31
DIAGNOSTIC_REVISION = "phase3b-normalized-state-v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct and compare matched normalized roots without "
            "executing any source proposal suffix."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _output_directory(requested: Path | None) -> Path:
    if requested is None:
        stamp = datetime.now(UTC).strftime(
            "normalized_state_%Y%m%dT%H%M%SZ"
        )
        requested = RAW_ROOT / stamp
    output = requested.resolve()
    raw_root = RAW_ROOT.resolve()
    if output == raw_root or raw_root not in output.parents:
        raise ValueError(f"Diagnostic must remain under {raw_root}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic: {output}")
    output.mkdir(parents=True)
    return output


def _slice(values: np.ndarray, address: Any) -> np.ndarray:
    if isinstance(address, tuple):
        return np.asarray(values[slice(*address)], dtype=np.float64).copy()
    return np.asarray([values[address]], dtype=np.float64)


def _named_joint_state(problem: Any, joint_name: str) -> tuple[np.ndarray, np.ndarray]:
    model = problem.sim.model
    return (
        _slice(problem.sim.data.qpos, model.get_joint_qpos_addr(joint_name)),
        _slice(problem.sim.data.qvel, model.get_joint_qvel_addr(joint_name)),
    )


def _indices(values: np.ndarray, owner: Any, name: str) -> np.ndarray:
    indexes = getattr(owner, name, ())
    return np.asarray(values[indexes], dtype=np.float64).copy()


def _state_descriptor(environment: Any, snapshot: Any) -> dict[str, Any]:
    restore_libero_state(environment, snapshot)
    controller = PolicyFreeController(environment)
    problem = controller.problem
    robot = problem.robots[0]
    bowl = problem.get_object(BOWL_NAME)
    bowl_joints = tuple(getattr(bowl, "joints", ()))
    if len(bowl_joints) != 1:
        raise ValueError(f"Expected one bowl joint, found {bowl_joints}")
    bowl_qpos, bowl_qvel = _named_joint_state(problem, bowl_joints[0])
    drawer_qpos, drawer_qvel = _named_joint_state(problem, TOP_DRAWER_JOINT)
    bowl_body_id = problem.obj_body_id[BOWL_NAME]
    bowl_orientation = np.asarray(
        problem.sim.data.body_xmat[bowl_body_id], dtype=np.float64
    ).reshape(3, 3)
    bowl_position = controller.bowl_position()
    eef_position = controller.eef_position()
    bowl_geoms = set(getattr(bowl, "contact_geoms", ()))
    contacts = [
        item
        for item in snapshot.contacts
        if item.get("geom1") in bowl_geoms or item.get("geom2") in bowl_geoms
    ]
    contact_pairs = sorted(
        [sorted((str(item.get("geom1")), str(item.get("geom2")))) for item in contacts]
    )
    runtime = snapshot.runtime_state
    return {
        "state_sha256": snapshot_sha256(snapshot),
        "timestep": int(runtime.get("environment", {}).get("timestep", -1)),
        "simulation_time": float(problem.sim.data.time),
        "bowl_position": bowl_position.tolist(),
        "bowl_orientation": bowl_orientation.tolist(),
        "bowl_joint_qpos": bowl_qpos.tolist(),
        "bowl_joint_qvel": bowl_qvel.tolist(),
        "eef_position": eef_position.tolist(),
        "eef_orientation": controller.eef_orientation().tolist(),
        "eef_bowl_relative_position": (eef_position - bowl_position).tolist(),
        "robot_joint_positions": controller.joint_positions().tolist(),
        "robot_joint_velocities": _indices(
            problem.sim.data.qvel, robot, "_ref_joint_vel_indexes"
        ).tolist(),
        "gripper_joint_positions": _indices(
            problem.sim.data.qpos, robot, "_ref_gripper_joint_pos_indexes"
        ).tolist(),
        "gripper_joint_velocities": _indices(
            problem.sim.data.qvel, robot, "_ref_gripper_joint_vel_indexes"
        ).tolist(),
        "drawer_joint": float(drawer_qpos[0]),
        "drawer_velocity": float(drawer_qvel[0]),
        "bowl_grasped": controller.bowl_grasped(),
        "bowl_contact_pairs": contact_pairs,
        "bowl_contacts": contacts,
        "runtime_state_sha256": canonical_sha256(runtime),
        "sim_data_sha256": canonical_sha256(runtime.get("sim_data", {})),
    }


def _checkpoint_evidence(source_run: Path, candidate_id: str) -> dict[str, Any]:
    path = source_run / "checkpoints" / f"{candidate_id}__cabinet.json"
    checkpoint = json.loads(path.read_text())
    matches = [
        row
        for row in checkpoint.get("results", ())
        if int(row.get("proposal_index", -1)) == PROPOSAL_INDEX
    ]
    if len(matches) != 1:
        raise ValueError(f"Checkpoint has no unique proposal {PROPOSAL_INDEX}")
    result = matches[0]["result"]
    if (
        int(result.get("demo_episode_index", -1)) != PROPOSAL_EPISODE
        or result.get("proposal_execution_mode")
        != "action_intrinsic_pregrasp_bowl_registered_v1"
    ):
        raise ValueError("Matched checkpoint proposal identity changed")
    return {
        "checkpoint_path": path.relative_to(PROJECT).as_posix(),
        "checkpoint_file_sha256": _file_sha256(path),
        "contract_sha256": checkpoint["contract_sha256"],
        "selection_lock_sha256": checkpoint["selection_lock_sha256"],
        "root_state_sha256": checkpoint["root_state_sha256"],
        "normalized_state_sha256": result["normalized_state_sha256"],
        "normalization_action_sha256": result[
            "normalization_action_sha256"
        ],
        "normalization_action_steps": result["normalization_action_steps"],
        "proposal_action_sha256": result["demo_action_sha256"],
        "completed_outcome": {
            "pass": result["pass"],
            "goal_ever_achieved": result["goal_ever_achieved"],
            "first_goal_demo_frame": result["first_goal_demo_frame"],
            "final_goals": result["final_goals"],
            "bridge_pass": result["phases"]["action_phase_bridge"]["pass"],
            "normalized_bowl_position_error_m": result[
                "normalized_bowl_position_error_m"
            ],
        },
    }


def reconstruct_normalized_candidate(
    *,
    environment: Any,
    candidate_id: str,
    evidence: dict[str, Any],
    demos: dict[str, Any],
    proposal: Any,
    config: dict[str, Any],
    support_bank: Any,
    registered_acquisition: Any,
) -> tuple[dict[str, Any], Any]:
    """Reconstruct one checkpoint-bound normalized root and return its payload."""

    spec = candidate_spec(candidate_id)
    constructed = construct_candidate(
        environment,
        spec,
        demos,
        config,
        support_reference_bank=support_bank,
        registered_grasp_acquisition=registered_acquisition,
    )
    if snapshot_sha256(constructed.snapshot) != evidence["root_state_sha256"]:
        raise RuntimeError(f"Reconstructed root changed for {candidate_id}")
    certificate = certify_computational_state(
        environment,
        constructed.snapshot,
        possession=spec.possession,
        probe_actions=config["certificate"]["actions"],
    )
    if certificate.get("pass") is not True:
        raise RuntimeError(f"Certificate failed for {candidate_id}")
    normalization = run_goal_oracle(
        environment,
        constructed.snapshot,
        spec=spec,
        goal="cabinet",
        demo=proposal,
        initial_bowl_position=constructed.initial_bowl_position,
        initial_eef_position=constructed.initial_eef_position,
        initial_eef_orientation=constructed.initial_eef_orientation,
        initial_joint_positions=constructed.initial_joint_positions,
        recovery_waypoints=constructed.recovery_waypoints,
        config=config,
        normalization_only=True,
        return_prepared_root=True,
    )
    prepared = normalization.pop("_prepared_oracle_root")
    expected = {
        "normalized_state_sha256": evidence["normalized_state_sha256"],
        "normalization_action_sha256": evidence[
            "normalization_action_sha256"
        ],
        "normalization_action_steps": evidence["normalization_action_steps"],
    }
    observed = {key: normalization[key] for key in expected}
    if observed != expected or normalization["source_proposal_replayed"] is not False:
        raise RuntimeError(f"Normalization identity changed for {candidate_id}")
    payload = {
        "candidate_id": candidate_id,
        "root_state_sha256": evidence["root_state_sha256"],
        "certificate_pass": True,
        "completed_outcome": evidence["completed_outcome"],
        "normalization": normalization,
        "root_descriptor": _state_descriptor(environment, constructed.snapshot),
        "normalized_descriptor": _state_descriptor(
            environment, prepared.snapshot
        ),
    }
    return payload, prepared


def _prepare_candidate(
    *,
    output_dir: Path,
    candidate_id: str,
    evidence: dict[str, Any],
    demos: dict[str, Any],
    proposal: Any,
    config: dict[str, Any],
    support_bank: Any,
    registered_acquisition: Any,
) -> dict[str, Any]:
    environment = make_stage_a_environment(PROJECT, output_dir, config)
    try:
        payload, _ = reconstruct_normalized_candidate(
            environment=environment,
            candidate_id=candidate_id,
            evidence=evidence,
            demos=demos,
            proposal=proposal,
            config=config,
            support_bank=support_bank,
            registered_acquisition=registered_acquisition,
        )
        return payload
    finally:
        environment.close()


def main() -> None:
    args = _parse_args()
    output_dir = _output_directory(args.output_dir)
    config_path = args.config.resolve()
    source_run = args.source_run.resolve()
    config = _load_config(config_path)
    demos = _load_demos(config)
    proposal_banks = _load_proposal_bank(config)
    proposal_matches = [
        item
        for item in proposal_banks["cabinet"]
        if item.episode_index == PROPOSAL_EPISODE
    ]
    if len(proposal_matches) != 1:
        raise ValueError("Cabinet episode 474 proposal identity changed")
    proposal = proposal_matches[0]
    evidence = {
        "near": _checkpoint_evidence(source_run, NEAR_CANDIDATE),
        "low": _checkpoint_evidence(source_run, LOW_CANDIDATE),
    }
    if evidence["near"]["proposal_action_sha256"] != proposal.action_sha256 or (
        evidence["low"]["proposal_action_sha256"] != proposal.action_sha256
    ):
        raise ValueError("Source proposal hash changed")
    if evidence["near"]["contract_sha256"] != evidence["low"]["contract_sha256"]:
        raise ValueError("Compared checkpoints do not share a contract")
    contract = {
        "schema_version": 1,
        "diagnostic_revision": DIAGNOSTIC_REVISION,
        "config_path": config_path.relative_to(PROJECT).as_posix(),
        "config_file_sha256": _file_sha256(config_path),
        "source_run": source_run.relative_to(PROJECT).as_posix(),
        "source_contract_sha256": evidence["near"]["contract_sha256"],
        "proposal_episode": PROPOSAL_EPISODE,
        "proposal_index": PROPOSAL_INDEX,
        "proposal_action_sha256": proposal.action_sha256,
        "candidate_evidence": evidence,
        "execution_scope": {
            "candidate_reconstructions": 2,
            "normalization_preparations": 2,
            "source_proposal_suffixes_executed": 0,
            "policies_loaded": 0,
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
        acquisition_matches = [
            item
            for item in phase_bank
            if item.source.episode_index == PROPOSAL_EPISODE
        ]
        if len(acquisition_matches) != 1:
            raise ValueError("Registered acquisition identity changed")
        candidates = {
            "near": _prepare_candidate(
                output_dir=output_dir,
                candidate_id=NEAR_CANDIDATE,
                evidence=evidence["near"],
                demos=demos,
                proposal=proposal,
                config=config,
                support_bank=support_bank,
                registered_acquisition=acquisition_matches[0],
            ),
            "low": _prepare_candidate(
                output_dir=output_dir,
                candidate_id=LOW_CANDIDATE,
                evidence=evidence["low"],
                demos=demos,
                proposal=proposal,
                config=config,
                support_bank=support_bank,
                registered_acquisition=acquisition_matches[0],
            ),
        }
        result = {
            "schema_version": 1,
            "status": "complete",
            "diagnostic_revision": DIAGNOSTIC_REVISION,
            "contract_sha256": contract_sha,
            "support_reference_bank_sha256": support_bank.sha256,
            "policy_loaded": False,
            "source_proposal_suffixes_executed": 0,
            "candidates": candidates,
            "normalized_comparison": compare_normalized_descriptors(
                candidates["near"]["normalized_descriptor"],
                candidates["low"]["normalized_descriptor"],
            ),
            "scientific_boundary": (
                "This diagnostic localizes observational state differences "
                "after exact normalization reconstruction. It does not by "
                "itself identify which difference causes proposal success."
            ),
        }
        atomic_write_json(output_dir / "result.json", result)
        artifact_sha = {
            path.name: _file_sha256(path)
            for path in sorted(output_dir.iterdir())
            if path.is_file() and path.name != "manifest.json"
        }
        atomic_write_json(
            output_dir / "manifest.json",
            {
                "schema_version": 1,
                "status": "complete",
                "created_at": datetime.now(UTC).isoformat(),
                "contract_sha256": contract_sha,
                "artifact_sha256": artifact_sha,
                "policy_loaded": False,
                "source_proposal_suffixes_executed": 0,
            },
        )
        print(json.dumps(result["normalized_comparison"], indent=2, sort_keys=True))
        print(f"Normalized-state diagnostic complete: {output_dir}")
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
                "policy_loaded": False,
                "source_proposal_suffixes_executed": 0,
            },
        )
        raise


if __name__ == "__main__":
    main()
