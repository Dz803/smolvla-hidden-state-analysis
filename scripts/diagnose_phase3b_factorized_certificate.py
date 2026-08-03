#!/usr/bin/env python
"""Test one factorized acquisition-to-placement certificate on the v37 failure."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from diagnose_phase3b_normalized_state import (
        DEFAULT_CONFIG,
        DEFAULT_SOURCE_RUN,
        LOW_CANDIDATE,
        PROPOSAL_EPISODE,
        PROPOSAL_INDEX,
        _checkpoint_evidence,
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
        PROPOSAL_EPISODE,
        PROPOSAL_INDEX,
        _checkpoint_evidence,
        reconstruct_normalized_candidate,
    )
    from scripts.run_phase3b_stage_a import (
        PROJECT,
        _file_sha256,
        _load_config,
        _load_demos,
        _load_proposal_bank,
    )
from smolvla_analysis.libero_state import capture_libero_state, restore_libero_state
from smolvla_analysis.phase3_crd import atomic_write_json, evaluate_common_goals
from smolvla_analysis.phase3b_libero import (
    CABINET_GOAL_SITE,
    PolicyFreeController,
    _action_sha256,
    _transport_validation_limits,
    _validate_grasped_transport_phase,
    build_landmark_registered_action_phase_proposal_bank,
    build_support_reference_bank,
    grasped_root_transit_plan,
    make_stage_a_environment,
    registered_root_execution_anchor,
)
from smolvla_analysis.phase3b_stage_a import (
    canonical_sha256,
    candidate_spec,
    snapshot_sha256,
)


RAW_ROOT = PROJECT / "local/phase3b_stage_a/factorized_certificates"
DEFAULT_REFERENCE = (
    PROJECT
    / "local/phase3b_stage_a/layout_alignment_diagnostics/"
    "layout_alignment_20260731T070614Z"
)
DIAGNOSTIC_REVISION = "phase3b-factorized-certificate-v4"
CONDITION = "early_stop_registered_acquisition_then_goal_registered_placement"
STABLE_GRASP_STREAK = 3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace the failed open-loop placement tail with a factorized "
            "goal-registered transport and release."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--reference-run", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _output_directory(requested: Path | None) -> Path:
    if requested is None:
        stamp = datetime.now(UTC).strftime(
            "factorized_certificate_%Y%m%dT%H%M%SZ"
        )
        requested = RAW_ROOT / stamp
    output = requested.resolve()
    raw_root = RAW_ROOT.resolve()
    if output == raw_root or raw_root not in output.parents:
        raise ValueError(f"Factorized certificate must remain under {raw_root}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite certificate: {output}")
    output.mkdir(parents=True)
    return output


def goal_registered_target(
    reference_bowl: Any,
    reference_goal: Any,
    target_goal: Any,
) -> np.ndarray:
    """Preserve a demonstrated bowl-to-goal offset at a target goal site."""

    reference_bowl_array = np.asarray(reference_bowl, dtype=np.float64)
    reference_goal_array = np.asarray(reference_goal, dtype=np.float64)
    target_goal_array = np.asarray(target_goal, dtype=np.float64)
    if any(
        value.shape != (3,) or not np.isfinite(value).all()
        for value in (
            reference_bowl_array,
            reference_goal_array,
            target_goal_array,
        )
    ):
        raise ValueError("Goal registration requires three finite 3-vectors")
    return reference_bowl_array + target_goal_array - reference_goal_array


def _acquire_until_stable(
    environment: Any,
    *,
    prepared: Any,
    spec: Any,
    proposal: Any,
    config: dict[str, Any],
) -> tuple[dict[str, Any], Any | None]:
    restore_libero_state(environment, prepared.snapshot)
    controller = PolicyFreeController(environment)
    phase_cfg = config["action_phase_oracle"]
    initial_bowl = controller.bowl_position()
    execution_anchor, registration = registered_root_execution_anchor(
        proposal, initial_bowl, config=config
    )
    route = grasped_root_transit_plan(
        controller.eef_position(),
        execution_anchor,
        clearance_margin_m=float(phase_cfg["clearance_margin_m"]),
        workspace_bounds=phase_cfg["workspace_bounds_m"],
        phase_budgets=phase_cfg["bridge_phase_budgets"],
    )
    bridge_phases = []
    for phase in route:
        intermediate = phase["phase"] != "target_descent"
        phase_result = controller.servo(
            target_position=phase["target_position"],
            target_orientation=proposal.anchor_orientation,
            gripper=float(phase_cfg["gripper_action"]),
            budget=int(phase["budget"]),
            max_translation_action=float(phase_cfg["max_translation_action"]),
            position_tolerance_m=float(
                phase_cfg[
                    "intermediate_tolerance_m"
                    if intermediate
                    else "final_tolerance_m"
                ]
            ),
            orientation_tolerance_rad=float(
                phase_cfg["orientation_tolerance_rad"]
            ),
        )
        bridge_phases.append(
            {
                "phase": phase["phase"],
                "target_position": np.asarray(phase["target_position"]).tolist(),
                "result": phase_result,
            }
        )
    bridge_action_count = len(controller.actions)
    bridge_goals = evaluate_common_goals(environment)
    bridge_pass = bool(
        all(item["result"]["pass"] for item in bridge_phases)
        and not any(controller.grasp_values)
        and not any(controller.done_values)
        and not any(any(value.values()) for value in controller.goal_values)
        and not any(bridge_goals.values())
        and controller.top_drawer_position()
        <= float(config["construction"]["open_drawer_threshold"])
        and np.linalg.norm(controller.bowl_position() - initial_bowl)
        <= float(phase_cfg["bowl_drift_tolerance_m"])
    )
    stable_streak = 0
    acquisition_snapshot = None
    trace = []
    if bridge_pass:
        for frame_index, action in zip(
            proposal.suffix.frame_indices,
            proposal.suffix.actions,
            strict=True,
        ):
            _, _, done, _ = controller.step(action)
            goals = evaluate_common_goals(environment)
            grasped = controller.bowl_grasped()
            stable_streak = stable_streak + 1 if grasped else 0
            trace.append(
                {
                    "source_frame": int(frame_index),
                    "bowl_grasped": grasped,
                    "goals": goals,
                    "done": bool(done),
                    "eef_position": controller.eef_position().tolist(),
                    "bowl_position": controller.bowl_position().tolist(),
                }
            )
            if any(goals.values()) or done:
                break
            if stable_streak >= STABLE_GRASP_STREAK:
                acquisition_snapshot = capture_libero_state(environment)
                break
    acquisition_actions = controller.actions[bridge_action_count:]
    success = bool(
        bridge_pass
        and acquisition_snapshot is not None
        and not any(
            point["goals"]["drawer"] or point["goals"]["cabinet"]
            for point in trace
        )
        and not any(point["done"] for point in trace)
    )
    return {
        "pass": success,
        "bridge_pass": bridge_pass,
        "root_landmark_registration": registration,
        "bridge_phases": bridge_phases,
        "bridge_action_count": bridge_action_count,
        "bridge_action_sha256": _action_sha256(
            controller.actions[:bridge_action_count]
        ),
        "source_actions_executed": len(acquisition_actions),
        "source_action_sha256": _action_sha256(acquisition_actions),
        "first_stable_grasp_source_frame": (
            trace[-1]["source_frame"] if acquisition_snapshot is not None else None
        ),
        "stable_grasp_streak": STABLE_GRASP_STREAK,
        "factorized_transport": {
            "phase_budget_policy": "ceiling_with_early_stop_on_tolerance",
            "pad_to_budget": False,
            "goal_or_terminal_stop": True,
            "physical_tolerances": "unchanged_from_config",
        },
        "trace": trace,
        "trace_sha256": canonical_sha256(trace),
        "acquisition_state_sha256": (
            snapshot_sha256(acquisition_snapshot)
            if acquisition_snapshot is not None
            else None
        ),
        "acquisition_timestep": (
            int(
                acquisition_snapshot.runtime_state.get("environment", {}).get(
                    "timestep", -1
                )
            )
            if acquisition_snapshot is not None
            else None
        ),
    }, acquisition_snapshot


def _factorized_placement(
    environment: Any,
    *,
    acquisition_snapshot: Any,
    reference_goal_state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    if acquisition_snapshot is None:
        return {
            "status": "not_run",
            "pass": False,
            "reason": "stable acquisition was not achieved",
        }
    restore_libero_state(environment, acquisition_snapshot)
    controller = PolicyFreeController(environment)
    if not controller.bowl_grasped():
        raise RuntimeError("Acquisition snapshot lost the bowl grasp")
    target_goal = controller.site_position(CABINET_GOAL_SITE)
    target_bowl = goal_registered_target(
        reference_goal_state["bowl_position"],
        reference_goal_state["cabinet_goal_position"],
        target_goal,
    )
    grasp_offset = controller.eef_position() - controller.bowl_position()
    target_eef = target_bowl + grasp_offset
    target_orientation = controller.eef_orientation()
    construction = config["construction"]
    route = grasped_root_transit_plan(
        controller.eef_position(),
        target_eef,
        clearance_margin_m=float(
            construction["grasped_root_clearance_margin_m"]
        ),
        workspace_bounds=construction["workspace_bounds_m"],
        phase_budgets=construction["grasped_root_transit_budgets"],
    )
    phases = []
    for phase in route:
        phase_result = controller.servo(
            target_position=phase["target_position"],
            target_orientation=target_orientation,
            gripper=1.0,
            budget=int(phase["budget"]),
            max_translation_action=float(
                construction["max_translation_action_grasped"]
            ),
            position_tolerance_m=float(
                construction[
                    "grasped_root_waypoint_tolerance_m"
                    if phase["phase"] != "target_descent"
                    else "root_tolerance_m"
                ]
            ),
            pad_to_budget=False,
            stop_on_goal_or_terminal=True,
        )
        goals_after_phase = evaluate_common_goals(environment)
        correct_goal_stop = bool(
            phase_result["stopped_on_goal"]
            and goals_after_phase["cabinet"]
            and not goals_after_phase["drawer"]
        )
        if not (
            correct_goal_stop
            or phase_result["stopped_on_goal"]
            or phase_result["stopped_on_terminal"]
        ):
            _validate_grasped_transport_phase(
                controller,
                phase_result,
                candidate_id=LOW_CANDIDATE,
                phase=f"factorized_{phase['phase']}",
                **_transport_validation_limits(config),
            )
        phases.append(
            {
                "phase": phase["phase"],
                "target_position": np.asarray(phase["target_position"]).tolist(),
                "result": phase_result,
                "bowl_grasped_after_phase": controller.bowl_grasped(),
                "goals_after_phase": goals_after_phase,
                "correct_goal_stop": correct_goal_stop,
            }
        )
        if (
            phase_result["stopped_on_goal"]
            or phase_result["stopped_on_terminal"]
            or not controller.bowl_grasped()
        ):
            break
    goals_before_release = evaluate_common_goals(environment)
    release = None
    correct_goal_stop = any(
        item["correct_goal_stop"] for item in phases
    )
    if (
        controller.bowl_grasped()
        and not correct_goal_stop
        and not any(controller.done_values)
    ):
        release = controller.servo(
            target_position=controller.eef_position(),
            target_orientation=target_orientation,
            gripper=-1.0,
            budget=int(config["oracle"]["setdown_release_budget"]),
            max_translation_action=float(config["oracle"]["max_translation_action"]),
            position_tolerance_m=float(config["oracle"]["servo_tolerance_m"]),
            stop_on_goal_or_terminal=True,
        )
    final_goals = evaluate_common_goals(environment)
    release_goal_stop = bool(
        release is not None
        and release["stopped_on_goal"]
        and final_goals["cabinet"]
        and not final_goals["drawer"]
    )
    correct_goal_stop = correct_goal_stop or release_goal_stop
    goal_ever = bool(
        goals_before_release["cabinet"]
        or final_goals["cabinet"]
        or any(values["cabinet"] for values in controller.goal_values)
    )
    wrong_goal = bool(
        final_goals["drawer"]
        or any(values["drawer"] for values in controller.goal_values)
    )
    seen_goal = False
    done_before_goal = False
    for goals, done in zip(
        controller.goal_values, controller.done_values, strict=True
    ):
        seen_goal = seen_goal or goals["cabinet"]
        done_before_goal = done_before_goal or (done and not seen_goal)
    transport_pass = bool(
        correct_goal_stop
        or (
            len(phases) == len(route)
            and all(item["result"]["pass"] for item in phases)
            and all(item["bowl_grasped_after_phase"] for item in phases)
        )
    )
    release_required = not any(
        item["correct_goal_stop"] for item in phases
    )
    release_pass = bool(
        not release_required
        or (
            release is not None
            and (release["pass"] or release_goal_stop)
        )
    )
    bowl_released = not controller.bowl_grasped()
    return {
        "status": "complete",
        "pass": bool(
            transport_pass
            and release_pass
            and (bowl_released or correct_goal_stop)
            and goal_ever
            and final_goals["cabinet"]
            and not wrong_goal
            and not done_before_goal
        ),
        "target_goal_position": target_goal.tolist(),
        "target_bowl_position": target_bowl.tolist(),
        "target_eef_position": target_eef.tolist(),
        "initial_grasp_offset": grasp_offset.tolist(),
        "transport_phases": phases,
        "transport_pass": transport_pass,
        "correct_goal_stop": correct_goal_stop,
        "goals_before_release": goals_before_release,
        "release": release,
        "release_pass": release_pass,
        "release_required": release_required,
        "release_goal_stop": release_goal_stop,
        "bowl_released": bowl_released,
        "goal_ever_achieved": goal_ever,
        "wrong_goal_ever_achieved": wrong_goal,
        "unexpected_done_before_goal": done_before_goal,
        "final_goals": final_goals,
        "action_count": len(controller.actions),
        "action_sha256": _action_sha256(controller.actions),
        "final_bowl_position": controller.bowl_position().tolist(),
        "final_eef_position": controller.eef_position().tolist(),
    }


def main() -> None:
    args = _parse_args()
    output_dir = _output_directory(args.output_dir)
    config_path = args.config.resolve()
    source_run = args.source_run.resolve()
    reference_run = args.reference_run.resolve()
    evidence = _checkpoint_evidence(source_run, LOW_CANDIDATE)
    source_contract_path = source_run / "contract.json"
    source_manifest_path = source_run / "manifest.json"
    source_contract_sha = canonical_sha256(
        json.loads(source_contract_path.read_text())
    )
    source_manifest = json.loads(source_manifest_path.read_text())
    if source_contract_sha != evidence["contract_sha256"]:
        raise ValueError("Source contract no longer matches low checkpoint")
    reference_result_path = reference_run / "result.json"
    reference_manifest_path = reference_run / "manifest.json"
    reference_result = json.loads(reference_result_path.read_text())
    reference_matches = [
        item
        for item in reference_result["conditions"]
        if item["condition"] == "layout_a_world_anchor_full_suffix"
    ]
    if len(reference_matches) != 1 or reference_matches[0].get(
        "goal_ever_achieved"
    ) is not True:
        raise ValueError("Reference cabinet goal state changed")
    reference_goal_state = reference_matches[0].get("first_goal_state")
    if not isinstance(reference_goal_state, dict):
        raise ValueError("Reference run has no first cabinet goal state")
    implementation_files = (
        Path(__file__).resolve(),
        PROJECT / "scripts/diagnose_phase3b_normalized_state.py",
        PROJECT / "src/smolvla_analysis/phase3b_libero.py",
        PROJECT / "src/smolvla_analysis/libero_state.py",
        PROJECT / "src/smolvla_analysis/phase3b_stage_a.py",
    )
    contract = {
        "schema_version": 1,
        "diagnostic_revision": DIAGNOSTIC_REVISION,
        "condition": CONDITION,
        "stable_grasp_streak": STABLE_GRASP_STREAK,
        "config_path": config_path.relative_to(PROJECT).as_posix(),
        "config_file_sha256": _file_sha256(config_path),
        "source_run": source_run.relative_to(PROJECT).as_posix(),
        "source_contract_sha256": source_contract_sha,
        "source_artifact_sha256": {
            "contract.json": _file_sha256(source_contract_path),
            "manifest.json": _file_sha256(source_manifest_path),
            evidence["checkpoint_path"].split("/")[-1]: evidence[
                "checkpoint_file_sha256"
            ],
        },
        "source_evidence": evidence,
        "reference_run": reference_run.relative_to(PROJECT).as_posix(),
        "reference_artifact_sha256": {
            "result.json": _file_sha256(reference_result_path),
            "manifest.json": _file_sha256(reference_manifest_path),
        },
        "reference_goal_state": reference_goal_state,
        "proposal_episode": PROPOSAL_EPISODE,
        "proposal_index": PROPOSAL_INDEX,
        "implementation_sha256": {
            path.relative_to(PROJECT).as_posix(): _file_sha256(path)
            for path in implementation_files
        },
        "execution_scope": {
            "completed_full_suffix_baselines_reexecuted": 0,
            "new_factorized_branches": 1,
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
        config = _load_config(config_path)
        demos = _load_demos(config)
        proposal_banks = _load_proposal_bank(config)
        proposal = proposal_banks["cabinet"][PROPOSAL_INDEX]
        if proposal.episode_index != PROPOSAL_EPISODE:
            raise ValueError("Locked proposal index changed")
        bank_environment = make_stage_a_environment(PROJECT, output_dir, config)
        try:
            support_bank = build_support_reference_bank(
                bank_environment,
                {goal: demos[goal] for goal in ("drawer", "cabinet")},
                config,
            )
        finally:
            bank_environment.close()
        expected_support_sha = source_manifest.get(
            "support_reference_bank_sha256"
        )
        if support_bank.sha256 != expected_support_sha:
            raise RuntimeError("Support-reference bank changed")
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
        acquisition_matches = [
            item
            for item in phase_bank
            if item.source.episode_index == PROPOSAL_EPISODE
        ]
        if len(acquisition_matches) != 1:
            raise ValueError("Registered acquisition identity changed")
        environment = make_stage_a_environment(PROJECT, output_dir, config)
        try:
            reconstruction, prepared = reconstruct_normalized_candidate(
                environment=environment,
                candidate_id=LOW_CANDIDATE,
                evidence=evidence,
                demos=demos,
                proposal=proposal,
                config=config,
                support_bank=support_bank,
                registered_acquisition=acquisition_matches[0],
            )
            acquisition, acquisition_snapshot = _acquire_until_stable(
                environment,
                prepared=prepared,
                spec=candidate_spec(LOW_CANDIDATE),
                proposal=phase_proposal,
                config=config,
            )
            atomic_write_json(output_dir / "acquisition.json", acquisition)
            placement = _factorized_placement(
                environment,
                acquisition_snapshot=acquisition_snapshot,
                reference_goal_state=reference_goal_state,
                config=config,
            )
        finally:
            environment.close()
        passed = bool(acquisition["pass"] and placement["pass"])
        result = {
            "schema_version": 1,
            "status": "complete",
            "diagnostic_revision": DIAGNOSTIC_REVISION,
            "contract_sha256": contract_sha,
            "condition": CONDITION,
            "pass": passed,
            "policy_loaded": False,
            "completed_full_suffix_baselines_reexecuted": 0,
            "source_proposal_full_tail_executed": False,
            "support_reference_bank_sha256": support_bank.sha256,
            "reconstruction": reconstruction,
            "acquisition": acquisition,
            "placement": placement,
            "scientific_boundary": (
                "A passing result certifies one factorized policy-free path "
                "from this normalized state. It does not make the failed "
                "open-loop proposal compatible or establish a VLA mechanism."
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
                "pass": passed,
                "policy_loaded": False,
                "completed_full_suffix_baselines_reexecuted": 0,
            },
        )
        print(
            json.dumps(
                {
                    "pass": passed,
                    "acquisition_pass": acquisition["pass"],
                    "placement_pass": placement["pass"],
                    "first_stable_grasp_source_frame": acquisition[
                        "first_stable_grasp_source_frame"
                    ],
                    "final_goals": placement.get("final_goals"),
                },
                indent=2,
                sort_keys=True,
            )
        )
        print(f"Factorized certificate complete: {output_dir}")
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
                "completed_full_suffix_baselines_reexecuted": 0,
            },
        )
        raise


if __name__ == "__main__":
    main()
