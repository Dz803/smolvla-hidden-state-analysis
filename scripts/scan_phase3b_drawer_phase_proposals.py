#!/usr/bin/env python
"""Audit drawer proposals with a uniform action-intrinsic phase slice.

This is a policy-free, raw-only diagnostic.  It never writes candidate states and
does not modify an existing Stage A run.  Every proposal is evaluated from the
same certified normalized root.  The only proposal-specific preparation is a
fixed three-leg Cartesian bridge to the end-effector pose fifty recorded frames
before the first open-to-close gripper transition.  The fixed five-second lead
retains the late drawer interaction and complete bowl-approach context without
requiring source object state, which the cached LeRobot dataset does not store.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from run_phase3b_stage_a import (
    DEFAULT_CONFIG,
    PROJECT,
    _file_sha256,
    _load_config,
    _load_demos,
    _load_proposal_bank,
)
from smolvla_analysis.libero_state import restore_libero_state
from smolvla_analysis.phase3_crd import atomic_write_json, evaluate_common_goals
from smolvla_analysis.phase3b_libero import (
    DemoTrace,
    PolicyFreeController,
    PreparedOracleRoot,
    _action_sha256,
    certify_computational_state,
    construct_candidate,
    grasped_root_transit_plan,
    make_stage_a_environment,
    replay_goal_proposal,
    run_goal_oracle,
)
from smolvla_analysis.phase3b_stage_a import (
    canonical_sha256,
    candidate_spec,
    snapshot_sha256,
)


RAW_ROOT = PROJECT / "local/phase3b_stage_a/drawer_phase_scans"
DEFAULT_CANDIDATE = (
    "stagea__drawer-open__possession-on-table__locus-drawer-side__"
    "support-demonstration-near__layout-a"
)
PREGRASP_LEAD_FRAMES = 50
ANCHOR_RULE = "fixed_50_frame_lead_before_first_gripper_close_transition"
BRIDGE_MODE = "three_leg_clearance_lift_transit_descent"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan phase-matched drawer continuations from one Stage A root."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--candidate-id", default=DEFAULT_CANDIDATE)
    parser.add_argument("--scan-dir", type=Path)
    return parser.parse_args()


def _scan_directory(requested: Path | None) -> Path:
    path = requested
    if path is None:
        stamp = datetime.now(UTC).strftime("drawer_phase_%Y%m%dT%H%M%SZ")
        path = RAW_ROOT / stamp
    path = path.resolve()
    raw_root = RAW_ROOT.resolve()
    if path != raw_root and raw_root not in path.parents:
        raise ValueError(f"Drawer phase scans must remain under {raw_root}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _suffix_trace(demo: DemoTrace, suffix_start: int) -> DemoTrace:
    if not 0 < suffix_start < len(demo.actions):
        raise ValueError(
            f"Invalid suffix boundary {suffix_start} for episode "
            f"{demo.episode_index} with {len(demo.actions)} actions"
        )
    actions = demo.actions[suffix_start:].copy()
    return DemoTrace(
        goal=demo.goal,
        episode_index=demo.episode_index,
        task_index=demo.task_index,
        frame_indices=demo.frame_indices[suffix_start:].copy(),
        states=demo.states[suffix_start:].copy(),
        actions=actions,
        action_sha256=_action_sha256(actions),
    )


def _extract_anchor(environment, *, spec, demo: DemoTrace, config: dict[str, Any]):
    controller = PolicyFreeController(environment)
    controller.reset_layout(
        spec.init_state_id, int(config["environment"]["reset_seed"])
    )
    initial_bowl = controller.bowl_position()
    gripper = np.asarray(demo.actions[:, 6], dtype=np.float64)
    close_candidates = np.flatnonzero(
        (gripper[1:] > 0.25) & (gripper[:-1] <= 0.25)
    ) + 1
    if not len(close_candidates):
        raise RuntimeError(
            f"Episode {demo.episode_index} has no gripper-close transition"
        )
    close_start = int(close_candidates[0])
    suffix_start = close_start - PREGRASP_LEAD_FRAMES
    if suffix_start <= 0:
        raise RuntimeError(
            f"Episode {demo.episode_index} has only {close_start} frames before "
            f"gripper close; {PREGRASP_LEAD_FRAMES} are required"
        )
    for index, action in enumerate(demo.actions[:suffix_start]):
        _, _, done, _ = controller.step(action)
        if done:
            raise RuntimeError(
                f"Episode {demo.episode_index} terminated before its phase anchor"
            )
    goals = evaluate_common_goals(environment)
    bowl_drift = float(np.linalg.norm(controller.bowl_position() - initial_bowl))
    if any(goals.values()) or controller.bowl_grasped():
        raise RuntimeError(
            f"Episode {demo.episode_index} has an invalid drawer-open anchor: "
            f"goals={goals}, grasped={controller.bowl_grasped()}"
        )
    if bowl_drift > float(config["oracle"]["normalized_bowl_position_tolerance_m"]):
        raise RuntimeError(
            f"Episode {demo.episode_index} moved the bowl before its anchor: "
            f"{bowl_drift:.6f} m"
        )
    suffix = _suffix_trace(demo, suffix_start)
    metadata = {
        "anchor_rule": ANCHOR_RULE,
        "pregrasp_lead_frames": PREGRASP_LEAD_FRAMES,
        "first_gripper_close_frame": int(demo.frame_indices[close_start]),
        "anchor_after_frame": int(demo.frame_indices[suffix_start - 1]),
        "suffix_start_frame": int(demo.frame_indices[suffix_start]),
        "prefix_action_count": suffix_start,
        "prefix_action_sha256": _action_sha256(demo.actions[:suffix_start]),
        "suffix_action_count": int(len(suffix.actions)),
        "suffix_action_sha256": suffix.action_sha256,
        "reference_drawer_joint": controller.top_drawer_position(),
        "reference_bowl_drift_m": bowl_drift,
        "anchor_eef_position": controller.eef_position().tolist(),
        "anchor_eef_orientation": controller.eef_orientation().tolist(),
        "reference_goals": goals,
        "reference_nonterminal": True,
        "reference_bowl_grasped": False,
    }
    return (
        np.asarray(metadata["anchor_eef_position"], dtype=np.float64),
        np.asarray(metadata["anchor_eef_orientation"], dtype=np.float64),
        suffix,
        metadata,
    )


def _phase_attempt(
    environment,
    *,
    prepared: PreparedOracleRoot,
    spec,
    demo: DemoTrace,
    config: dict[str, Any],
) -> dict[str, Any]:
    anchor_position, anchor_orientation, suffix, anchor = _extract_anchor(
        environment, spec=spec, demo=demo, config=config
    )
    restore_libero_state(environment, prepared.snapshot)
    controller = PolicyFreeController(environment)
    initial_bowl = controller.bowl_position()
    bridge_action_start = len(controller.actions)
    route = grasped_root_transit_plan(
        controller.eef_position(),
        anchor_position,
        clearance_margin_m=float(
            config["construction"]["grasped_root_clearance_margin_m"]
        ),
        workspace_bounds=config["construction"]["workspace_bounds_m"],
        phase_budgets=config["construction"]["grasped_root_transit_budgets"],
    )
    phase_records = []
    for phase in route:
        intermediate = phase["phase"] != "target_descent"
        result = controller.servo(
            target_position=phase["target_position"],
            target_orientation=anchor_orientation,
            gripper=-1.0,
            budget=int(phase["budget"]),
            max_translation_action=float(config["oracle"]["max_translation_action"]),
            position_tolerance_m=float(
                config["construction"]["grasped_root_waypoint_tolerance_m"]
                if intermediate
                else config["oracle"]["servo_tolerance_m"]
            ),
        )
        phase_records.append(
            {
                "phase": phase["phase"],
                "target_position": np.asarray(phase["target_position"]).tolist(),
                "result": result,
            }
        )
    bridge_action_count = len(controller.actions) - bridge_action_start
    bridge_goals = evaluate_common_goals(environment)
    bridge_drawer_joint = controller.top_drawer_position()
    bridge_bowl_drift = float(
        np.linalg.norm(controller.bowl_position() - initial_bowl)
    )
    bridge_grasp_ever = any(controller.grasp_values[:bridge_action_count])
    bridge_done_ever = any(controller.done_values[:bridge_action_count])
    bridge_goal_ever = any(
        any(values.values())
        for values in controller.goal_values[:bridge_action_count]
    )
    threshold = float(config["construction"]["open_drawer_threshold"])
    bridge_pass = bool(
        all(record["result"]["pass"] for record in phase_records)
        and not bridge_grasp_ever
        and not bridge_done_ever
        and not bridge_goal_ever
        and not any(bridge_goals.values())
        and bridge_drawer_joint <= threshold
        and bridge_bowl_drift
        <= float(config["oracle"]["normalized_bowl_position_tolerance_m"])
    )
    if bridge_pass:
        _, outcome = replay_goal_proposal(
            environment,
            goal="drawer",
            demo=suffix,
            controller=controller,
        )
    else:
        outcome = {
            "goal_ever_achieved": False,
            "first_goal_demo_frame": None,
            "wrong_goal_ever_achieved": False,
            "unexpected_done_before_goal": bridge_done_ever,
            "final_goals": bridge_goals,
            "pass": False,
        }
    executed_suffix_actions = len(controller.actions) - bridge_action_count
    bridge_actions = np.stack(controller.actions[:bridge_action_count])
    all_actions = np.stack(controller.actions)
    result = {
        "episode_index": demo.episode_index,
        "task_index": demo.task_index,
        "frame_count": int(len(demo.actions)),
        "action_sha256": demo.action_sha256,
        "anchor": anchor,
        "bridge": {
            "mode": BRIDGE_MODE,
            "pass": bridge_pass,
            "phases": phase_records,
            "action_count": bridge_action_count,
            "action_sha256": _action_sha256(bridge_actions),
            "active_action_steps": int(
                sum(
                    int(record["result"]["active_action_steps"])
                    for record in phase_records
                )
            ),
            "bowl_drift_m": bridge_bowl_drift,
            "bowl_grasp_ever": bridge_grasp_ever,
            "done_ever": bridge_done_ever,
            "goal_ever": bridge_goal_ever,
            "final_goals": bridge_goals,
            "final_drawer_joint": bridge_drawer_joint,
        },
        **outcome,
        "pass": bool(bridge_pass and outcome["pass"]),
        "executed_suffix_action_steps": int(executed_suffix_actions),
        "phase_execution_action_steps": int(len(controller.actions)),
        "phase_execution_action_sha256": _action_sha256(all_actions),
        "eef_path_length_m": controller.eef_path_length_m,
        "control_effort": controller.control_effort,
        "motion_control_effort": controller.motion_control_effort,
    }
    restore_libero_state(environment, prepared.snapshot)
    return result


def main() -> None:
    args = _parse_args()
    config_path = args.config.resolve()
    config = _load_config(config_path)
    spec = candidate_spec(args.candidate_id)
    if spec.drawer_aperture != "open":
        raise ValueError("Drawer phase scan requires an open-drawer candidate")
    proposals = _load_proposal_bank(config)["drawer"]
    scan_dir = _scan_directory(args.scan_dir)
    manifest_path = scan_dir / "manifest.json"
    attempts_path = scan_dir / "attempts.json"
    proposal_inventory = [
        {
            "proposal_index": index,
            "episode_index": demo.episode_index,
            "task_index": demo.task_index,
            "frame_count": int(len(demo.actions)),
            "action_sha256": demo.action_sha256,
        }
        for index, demo in enumerate(proposals)
    ]
    expected = {
        "schema_version": 1,
        "stage": "phase3b_stage_a_drawer_phase_scan",
        "scan_id": scan_dir.name,
        "candidate_id": spec.candidate_id,
        "config_path": config_path.relative_to(PROJECT).as_posix(),
        "config_sha256": _file_sha256(config_path),
        "construction_revision": config["construction_revision"],
        "proposal_inventory_sha256": canonical_sha256(proposal_inventory),
        "proposal_count": len(proposals),
        "anchor_rule": ANCHOR_RULE,
        "bridge_mode": BRIDGE_MODE,
        "pregrasp_lead_frames": PREGRASP_LEAD_FRAMES,
        "bridge_contract": {
            "clearance_margin_m": config["construction"][
                "grasped_root_clearance_margin_m"
            ],
            "phase_budgets": config["construction"][
                "grasped_root_transit_budgets"
            ],
            "intermediate_tolerance_m": config["construction"][
                "grasped_root_waypoint_tolerance_m"
            ],
            "final_tolerance_m": config["oracle"]["servo_tolerance_m"],
            "max_translation_action": config["oracle"]["max_translation_action"],
            "gripper": -1.0,
        },
        "policy_loaded": False,
        "source_sha256": {
            "scanner": _file_sha256(Path(__file__).resolve()),
            "runner": _file_sha256(PROJECT / "scripts/run_phase3b_stage_a.py"),
            "runtime": _file_sha256(
                PROJECT / "src/smolvla_analysis/phase3b_libero.py"
            ),
            "lattice": _file_sha256(
                PROJECT / "src/smolvla_analysis/phase3b_stage_a.py"
            ),
        },
    }
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"Drawer-phase scan resume mismatch for {key}")
    else:
        manifest = {
            **expected,
            "status": "in_progress",
            "attempt_count": 0,
            "success_count": 0,
            "created_at": datetime.now(UTC).isoformat(),
        }
        atomic_write_json(manifest_path, manifest)

    existing_attempts = (
        json.loads(attempts_path.read_text()) if attempts_path.is_file() else []
    )
    if not isinstance(existing_attempts, list):
        raise ValueError("Existing drawer-phase attempts are not a list")
    existing = {int(row["episode_index"]): row for row in existing_attempts}
    if len(existing) != len(existing_attempts):
        raise ValueError("Existing drawer-phase attempts contain duplicate episodes")
    expected_episode_ids = {demo.episode_index for demo in proposals}
    if not set(existing).issubset(expected_episode_ids):
        raise ValueError("Existing drawer-phase attempts are outside the inventory")

    demos = _load_demos(config)
    environment = make_stage_a_environment(PROJECT, scan_dir, config)
    try:
        constructed = construct_candidate(
            environment, spec, demos, config, support_reference_bank=None
        )
        root_certificate = certify_computational_state(
            environment,
            constructed.snapshot,
            possession=spec.possession,
            probe_actions=config["certificate"]["actions"],
        )
        if not root_certificate["pass"]:
            raise RuntimeError("Drawer-phase scan root certificate failed")
        reference = run_goal_oracle(
            environment,
            constructed.snapshot,
            spec=spec,
            goal="drawer",
            demo=proposals[0],
            initial_bowl_position=constructed.initial_bowl_position,
            initial_eef_position=constructed.initial_eef_position,
            initial_eef_orientation=constructed.initial_eef_orientation,
            initial_joint_positions=constructed.initial_joint_positions,
            recovery_waypoints=constructed.recovery_waypoints,
            config=config,
            raise_on_failure=False,
            return_prepared_root=True,
        )
        prepared = reference.pop("_prepared_oracle_root")
        if not isinstance(prepared, PreparedOracleRoot):
            raise TypeError("Oracle did not return a prepared normalized root")
        normalized_certificate = certify_computational_state(
            environment,
            prepared.snapshot,
            possession="on_table",
            probe_actions=config["certificate"]["actions"],
        )
        if not normalized_certificate["pass"]:
            raise RuntimeError("Drawer-phase normalized-root certificate failed")
        root_metadata = {
            "root_state_sha256": snapshot_sha256(constructed.snapshot),
            "normalized_state_sha256": snapshot_sha256(prepared.snapshot),
            "normalization_action_sha256": reference[
                "normalization_action_sha256"
            ],
            "normalization_action_steps": reference["normalization_action_steps"],
            "root_certificate": root_certificate,
            "normalized_certificate": normalized_certificate,
        }
        if manifest.get("root_metadata", root_metadata) != root_metadata:
            raise ValueError("Drawer-phase scan reconstructed a different root")
        manifest["root_metadata"] = root_metadata
        atomic_write_json(manifest_path, manifest)

        for proposal_index, demo in enumerate(proposals):
            if demo.episode_index in existing:
                locked = proposal_inventory[proposal_index]
                row = existing[demo.episode_index]
                if row["action_sha256"] != locked["action_sha256"]:
                    raise ValueError(
                        f"Episode {demo.episode_index} action hash changed"
                    )
                print(f"skip complete episode {demo.episode_index}", flush=True)
                continue
            attempt = _phase_attempt(
                environment,
                prepared=prepared,
                spec=spec,
                demo=demo,
                config=config,
            )
            attempt["proposal_index"] = proposal_index
            existing_attempts.append(attempt)
            existing[demo.episode_index] = attempt
            existing_attempts.sort(key=lambda row: int(row["proposal_index"]))
            atomic_write_json(attempts_path, existing_attempts)
            manifest.update(
                {
                    "status": "in_progress",
                    "attempt_count": len(existing_attempts),
                    "success_count": sum(
                        bool(row["pass"]) for row in existing_attempts
                    ),
                    "last_completed_episode": demo.episode_index,
                    "updated_at": datetime.now(UTC).isoformat(),
                }
            )
            atomic_write_json(manifest_path, manifest)
            print(
                f"episode {demo.episode_index}: bridge={attempt['bridge']['pass']} "
                f"pass={attempt['pass']} goal_frame={attempt['first_goal_demo_frame']}",
                flush=True,
            )

        if len(existing_attempts) != len(proposals):
            raise RuntimeError("Drawer-phase scan ended with an incomplete ledger")
        successful = [row for row in existing_attempts if row["pass"]]
        bridge_failures = [
            row for row in existing_attempts if not row["bridge"]["pass"]
        ]
        anchor_inventory = [
            {
                "proposal_index": row["proposal_index"],
                "episode_index": row["episode_index"],
                "anchor": row["anchor"],
            }
            for row in existing_attempts
        ]
        manifest.update(
            {
                "status": "complete",
                "attempt_count": len(existing_attempts),
                "success_count": len(successful),
                "bridge_failure_count": len(bridge_failures),
                "successful_episode_indices": [
                    int(row["episode_index"]) for row in successful
                ],
                "bridge_failure_episode_indices": [
                    int(row["episode_index"]) for row in bridge_failures
                ],
                "anchor_inventory_sha256": canonical_sha256(anchor_inventory),
                "completed_at": datetime.now(UTC).isoformat(),
            }
        )
        atomic_write_json(manifest_path, manifest)
        print(
            f"complete: {len(successful)}/{len(proposals)} proposals pass; "
            f"bridge_failures={len(bridge_failures)}; "
            f"episodes={[row['episode_index'] for row in successful]}",
            flush=True,
        )
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
                "updated_at": datetime.now(UTC).isoformat(),
            }
        )
        atomic_write_json(manifest_path, manifest)
        raise
    finally:
        environment.close()


if __name__ == "__main__":
    main()
