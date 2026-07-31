#!/usr/bin/env python
"""Factor a Stage A layout failure into acquisition and placement competence.

This bounded, policy-free smoke never edits a Stage A run.  It reproduces one
locked proposal on the exact layout-A and layout-B roots, then tests one
landmark registration and one factorized placement branch.  The acquisition
snapshot from the registered full-suffix condition is reused by the placement
branch, so no acquisition path is simulated twice.
"""

from __future__ import annotations

import argparse
import json
import traceback
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
from smolvla_analysis.libero_state import capture_libero_state, restore_libero_state
from smolvla_analysis.phase3_crd import atomic_write_json, evaluate_common_goals
from smolvla_analysis.phase3b_alignment import landmark_registered_point
from smolvla_analysis.phase3b_libero import (
    CABINET_GOAL_SITE,
    PolicyFreeController,
    _action_sha256,
    build_action_phase_proposal_bank,
    certify_computational_state,
    construct_candidate,
    grasped_root_transit_plan,
    make_stage_a_environment,
    run_goal_oracle,
)
from smolvla_analysis.phase3b_stage_a import (
    canonical_sha256,
    candidate_spec,
    snapshot_sha256,
)


RAW_ROOT = PROJECT / "local/phase3b_stage_a/layout_alignment_diagnostics"
DEFAULT_SOURCE_RUN = (
    PROJECT
    / "local/phase3b_stage_a/phase3b_stage_a_20260731T035449Z"
)
BASE_CANDIDATE = (
    "stagea__drawer-open__possession-on-table__locus-drawer-side__"
    "support-demonstration-near__layout-{}"
)
PROPOSAL_EPISODE = 474
STABLE_GRASP_STREAK = 3
CONDITIONS = (
    "layout_a_world_anchor_full_suffix",
    "layout_b_world_anchor_full_suffix",
    "layout_b_bowl_registered_full_suffix",
    "layout_b_registered_acquisition_then_goal_registered_placement",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose world-frame versus object-relative Stage A alignment."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _output_directory(requested: Path | None) -> Path:
    if requested is None:
        stamp = datetime.now(UTC).strftime("layout_alignment_%Y%m%dT%H%M%SZ")
        requested = RAW_ROOT / stamp
    output = requested.resolve()
    raw_root = RAW_ROOT.resolve()
    if output == raw_root or raw_root not in output.parents:
        raise ValueError(f"Alignment diagnostics must remain under {raw_root}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite alignment diagnostic: {output}")
    output.mkdir(parents=True)
    return output


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _expected_evidence(source_run: Path) -> dict[str, dict[str, str]]:
    source_run = source_run.resolve()
    expected: dict[str, dict[str, str]] = {}
    for layout in ("a", "b"):
        candidate_id = BASE_CANDIDATE.format(layout)
        if layout == "a":
            record_path = source_run / "candidates" / f"{candidate_id}.json"
            record = json.loads(record_path.read_text())
            oracle = record["oracles"]["cabinet"]
            expected[layout] = {
                "root_state_sha256": record["state_sha256"],
                "normalized_state_sha256": oracle[
                    "shared_normalized_state_sha256"
                ],
                "evidence_file": record_path.relative_to(PROJECT).as_posix(),
                "evidence_file_sha256": _file_sha256(record_path),
            }
        else:
            checkpoint_path = (
                source_run / "checkpoints" / f"{candidate_id}__cabinet.json"
            )
            checkpoint = json.loads(checkpoint_path.read_text())
            if checkpoint.get("result_count") != 46:
                raise ValueError("Layout-B cabinet checkpoint is not exhaustive")
            normalized = {
                item["result"]["normalized_state_sha256"]
                for item in checkpoint["results"]
            }
            if len(normalized) != 1:
                raise ValueError("Layout-B cabinet checkpoint changed normalized roots")
            expected[layout] = {
                "root_state_sha256": checkpoint["root_state_sha256"],
                "normalized_state_sha256": next(iter(normalized)),
                "evidence_file": checkpoint_path.relative_to(PROJECT).as_posix(),
                "evidence_file_sha256": _file_sha256(checkpoint_path),
            }
    return expected


def _prepare_layout(
    environment,
    *,
    layout: str,
    demos: dict[str, Any],
    proposal: Any,
    config: dict[str, Any],
    expected: dict[str, str],
) -> dict[str, Any]:
    spec = candidate_spec(BASE_CANDIDATE.format(layout.lower()))
    constructed = construct_candidate(
        environment,
        spec,
        demos,
        config,
        support_reference_bank=None,
    )
    root_sha = snapshot_sha256(constructed.snapshot)
    if root_sha != expected["root_state_sha256"]:
        raise RuntimeError(
            f"Reconstructed {layout} root changed: {root_sha} != "
            f"{expected['root_state_sha256']}"
        )
    certificate = certify_computational_state(
        environment,
        constructed.snapshot,
        possession=spec.possession,
        probe_actions=config["certificate"]["actions"],
    )
    if certificate["pass"] is not True:
        raise RuntimeError(f"Reconstructed {layout} certificate failed")
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
    if normalization["normalized_state_sha256"] != expected[
        "normalized_state_sha256"
    ]:
        raise RuntimeError(f"Reconstructed {layout} normalized root changed")
    restore_libero_state(environment, prepared.snapshot)
    controller = PolicyFreeController(environment)
    return {
        "spec": spec,
        "constructed": constructed,
        "prepared": prepared,
        "root_state_sha256": root_sha,
        "normalized_state_sha256": normalization["normalized_state_sha256"],
        "normalized_bowl_position": controller.bowl_position(),
        "cabinet_goal_position": controller.site_position(CABINET_GOAL_SITE),
        "certificate": certificate,
        "normalization": normalization,
    }


def _trace_point(
    controller: PolicyFreeController,
    *,
    frame_index: int,
    goals: dict[str, bool],
    done: bool,
) -> dict[str, Any]:
    eef = controller.eef_position()
    bowl = controller.bowl_position()
    return {
        "source_frame": int(frame_index),
        "eef_position": eef,
        "bowl_position": bowl,
        "eef_bowl_distance_m": float(np.linalg.norm(eef - bowl)),
        "bowl_grasped": controller.bowl_grasped(),
        "goals": goals,
        "done": bool(done),
    }


def _run_full_suffix(
    environment,
    *,
    prepared: Any,
    spec: Any,
    proposal: Any,
    anchor_position: np.ndarray,
    anchor_orientation: np.ndarray,
    config: dict[str, Any],
    condition: str,
) -> tuple[dict[str, Any], Any | None]:
    restore_libero_state(environment, prepared.snapshot)
    controller = PolicyFreeController(environment)
    phase_cfg = config["action_phase_oracle"]
    initial_bowl = controller.bowl_position()
    route = grasped_root_transit_plan(
        controller.eef_position(),
        anchor_position,
        clearance_margin_m=float(phase_cfg["clearance_margin_m"]),
        workspace_bounds=phase_cfg["workspace_bounds_m"],
        phase_budgets=phase_cfg["bridge_phase_budgets"],
    )
    bridge_phases = []
    for phase in route:
        intermediate = phase["phase"] != "target_descent"
        result = controller.servo(
            target_position=phase["target_position"],
            target_orientation=anchor_orientation,
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
                "target_position": phase["target_position"],
                "result": result,
            }
        )
    bridge_action_count = len(controller.actions)
    bridge_drawer_joint = controller.top_drawer_position()
    aperture_preserved = bool(
        bridge_drawer_joint
        <= float(config["construction"]["open_drawer_threshold"])
        if spec.drawer_aperture == "open"
        else abs(bridge_drawer_joint)
        <= float(config["construction"]["closed_drawer_tolerance"])
    )
    bridge_pass = bool(
        all(item["result"]["pass"] for item in bridge_phases)
        and not any(controller.grasp_values)
        and not any(controller.done_values)
        and not any(any(goals.values()) for goals in controller.goal_values)
        and np.linalg.norm(controller.bowl_position() - initial_bowl)
        <= float(phase_cfg["bowl_drift_tolerance_m"])
        and aperture_preserved
    )
    trace = []
    first_goal_state = None
    stable_grasp = 0
    acquisition_snapshot = None
    if bridge_pass:
        for frame_index, action in zip(
            proposal.suffix.frame_indices,
            proposal.suffix.actions,
            strict=True,
        ):
            _, _, done, _ = controller.step(action)
            goals = evaluate_common_goals(environment)
            point = _trace_point(
                controller,
                frame_index=int(frame_index),
                goals=goals,
                done=done,
            )
            trace.append(point)
            stable_grasp = stable_grasp + 1 if point["bowl_grasped"] else 0
            if stable_grasp >= STABLE_GRASP_STREAK and acquisition_snapshot is None:
                acquisition_snapshot = capture_libero_state(environment)
            if goals["cabinet"]:
                first_goal_state = {
                    **point,
                    "cabinet_goal_position": controller.site_position(
                        CABINET_GOAL_SITE
                    ),
                }
                break
            if done:
                break
    final_goals = evaluate_common_goals(environment)
    return (
        {
            "condition": condition,
            "bridge_pass": bridge_pass,
            "bridge_action_count": bridge_action_count,
            "bridge_action_sha256": _action_sha256(
                controller.actions[:bridge_action_count]
            ),
            "bridge_phases": bridge_phases,
            "bridge_final_drawer_joint": bridge_drawer_joint,
            "bridge_drawer_aperture_preserved": aperture_preserved,
            "anchor_position": anchor_position,
            "anchor_orientation": anchor_orientation,
            "trace": trace,
            "trace_sha256": canonical_sha256(trace),
            "stable_grasp_achieved": acquisition_snapshot is not None,
            "first_stable_grasp_source_frame": next(
                (
                    trace[index]["source_frame"]
                    for index in range(STABLE_GRASP_STREAK - 1, len(trace))
                    if all(
                        trace[offset]["bowl_grasped"]
                        for offset in range(
                            index - STABLE_GRASP_STREAK + 1, index + 1
                        )
                    )
                ),
                None,
            ),
            "first_goal_state": first_goal_state,
            "goal_ever_achieved": first_goal_state is not None,
            "wrong_goal_ever_achieved": any(
                point["goals"]["drawer"] for point in trace
            ),
            "unexpected_done_before_goal": any(
                point["done"] for point in trace
            )
            and first_goal_state is None,
            "final_goals": final_goals,
            "source_actions_executed": len(trace),
            "source_action_count": len(proposal.suffix.actions),
            "minimum_eef_bowl_distance_m": min(
                (point["eef_bowl_distance_m"] for point in trace),
                default=None,
            ),
            "final_bowl_position": controller.bowl_position(),
            "final_eef_position": controller.eef_position(),
        },
        acquisition_snapshot,
    )


def _run_registered_placement(
    environment,
    *,
    acquisition_snapshot: Any,
    reference_goal_state: dict[str, Any],
    target_cabinet_goal: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    condition = CONDITIONS[3]
    if acquisition_snapshot is None:
        return {
            "condition": condition,
            "status": "not_run",
            "reason": "registered suffix never acquired a stable grasp",
        }
    if reference_goal_state is None:
        return {
            "condition": condition,
            "status": "not_run",
            "reason": "layout-A reference never reached the cabinet goal",
        }
    restore_libero_state(environment, acquisition_snapshot)
    controller = PolicyFreeController(environment)
    if not controller.bowl_grasped():
        raise RuntimeError("Acquisition snapshot does not preserve the bowl grasp")
    target_bowl = landmark_registered_point(
        np.asarray(reference_goal_state["bowl_position"]),
        np.asarray(reference_goal_state["cabinet_goal_position"]),
        np.asarray(target_cabinet_goal),
    )
    grasp_offset = controller.eef_position() - controller.bowl_position()
    target_eef = target_bowl + grasp_offset
    orientation = controller.eef_orientation()
    construction_cfg = config["construction"]
    plan = grasped_root_transit_plan(
        controller.eef_position(),
        target_eef,
        clearance_margin_m=float(
            construction_cfg["grasped_root_clearance_margin_m"]
        ),
        workspace_bounds=construction_cfg["workspace_bounds_m"],
        phase_budgets=construction_cfg["grasped_root_transit_budgets"],
    )
    phases = []
    grasp_preserved = True
    for phase in plan:
        result = controller.servo(
            target_position=phase["target_position"],
            target_orientation=orientation,
            gripper=1.0,
            budget=int(phase["budget"]),
            max_translation_action=float(
                construction_cfg["max_translation_action_grasped"]
            ),
            position_tolerance_m=float(
                construction_cfg[
                    "grasped_root_waypoint_tolerance_m"
                    if phase["phase"] != "target_descent"
                    else "root_tolerance_m"
                ]
            ),
        )
        grasp_preserved = grasp_preserved and controller.bowl_grasped()
        phases.append(
            {
                "phase": phase["phase"],
                "target_position": phase["target_position"],
                "result": result,
                "grasped_after_phase": controller.bowl_grasped(),
                "goals_after_phase": evaluate_common_goals(environment),
            }
        )
        if any(controller.done_values) or not controller.bowl_grasped():
            break
    goals_before_release = evaluate_common_goals(environment)
    release = None
    if controller.bowl_grasped() and not any(controller.done_values):
        release = controller.servo(
            target_position=controller.eef_position(),
            target_orientation=orientation,
            gripper=-1.0,
            budget=int(config["oracle"]["setdown_release_budget"]),
            max_translation_action=float(config["oracle"]["max_translation_action"]),
            position_tolerance_m=float(config["oracle"]["servo_tolerance_m"]),
        )
    final_goals = evaluate_common_goals(environment)
    goal_ever = bool(
        goals_before_release["cabinet"]
        or final_goals["cabinet"]
        or any(values["cabinet"] for values in controller.goal_values)
    )
    wrong_goal = bool(
        any(values["drawer"] for values in controller.goal_values)
        or final_goals["drawer"]
    )
    done_before_goal = False
    seen_goal = False
    for goals, done in zip(
        controller.goal_values, controller.done_values, strict=True
    ):
        seen_goal = seen_goal or goals["cabinet"]
        done_before_goal = done_before_goal or (done and not seen_goal)
    return {
        "condition": condition,
        "status": "complete",
        "target_bowl_position": target_bowl,
        "target_eef_position": target_eef,
        "initial_grasp_offset": grasp_offset,
        "transport_phases": phases,
        "grasp_preserved_through_transport": grasp_preserved,
        "goals_before_release": goals_before_release,
        "release": release,
        "goal_ever_achieved": goal_ever,
        "wrong_goal_ever_achieved": wrong_goal,
        "unexpected_done_before_goal": done_before_goal,
        "final_goals": final_goals,
        "pass": bool(goal_ever and not wrong_goal and not done_before_goal),
        "action_count": len(controller.actions),
        "action_sha256": _action_sha256(controller.actions),
        "final_bowl_position": controller.bowl_position(),
        "final_eef_position": controller.eef_position(),
    }


def main() -> None:
    args = _parse_args()
    config_path = args.config.resolve()
    source_run = args.source_run.resolve()
    output_dir = _output_directory(args.output_dir)
    expected = _expected_evidence(source_run)
    contract = {
        "schema_version": 1,
        "diagnostic_revision": "phase3b-layout-alignment-v1",
        "conditions": list(CONDITIONS),
        "stable_grasp_streak": STABLE_GRASP_STREAK,
        "proposal_episode_index": PROPOSAL_EPISODE,
        "proposal_selection_reason": (
            "pre-existing cabinet support and grasp-construction trace"
        ),
        "registration": (
            "preserve layout-A EEF-to-bowl anchor offset in layout B; preserve "
            "layout-A achieved bowl-to-cabinet offset for factorized placement"
        ),
        "policy_loaded": False,
        "canonical_rollout_reused": False,
        "source_run": source_run.relative_to(PROJECT).as_posix(),
        "source_run_contract_file_sha256": _file_sha256(
            source_run / "contract.json"
        ),
        "source_evidence": expected,
        "config_path": config_path.relative_to(PROJECT).as_posix(),
        "config_sha256": _file_sha256(config_path),
        "source_sha256": {
            "script": _file_sha256(Path(__file__).resolve()),
            "alignment": _file_sha256(
                PROJECT / "src/smolvla_analysis/phase3b_alignment.py"
            ),
            "runtime": _file_sha256(
                PROJECT / "src/smolvla_analysis/phase3b_libero.py"
            ),
            "lattice": _file_sha256(
                PROJECT / "src/smolvla_analysis/phase3b_stage_a.py"
            ),
        },
    }
    contract_sha = canonical_sha256(contract)
    atomic_write_json(output_dir / "contract.json", contract)
    try:
        config = _load_config(config_path)
        demos = _load_demos(config)
        proposals = _load_proposal_bank(config)["cabinet"]
        proposal = next(
            item for item in proposals if item.episode_index == PROPOSAL_EPISODE
        )
        environment = make_stage_a_environment(PROJECT, output_dir, config)
        try:
            phase_proposals = {
                layout: build_action_phase_proposal_bank(
                    environment,
                    layout=layout,
                    proposals=(proposal,),
                    config=config,
                )[0]
                for layout in ("A", "B")
            }
            prepared = {
                layout: _prepare_layout(
                    environment,
                    layout=layout,
                    demos=demos,
                    proposal=proposal,
                    config=config,
                    expected=expected[layout.lower()],
                )
                for layout in ("A", "B")
            }
            world_a, _ = _run_full_suffix(
                environment,
                prepared=prepared["A"]["prepared"],
                spec=prepared["A"]["spec"],
                proposal=phase_proposals["A"],
                anchor_position=phase_proposals["A"].anchor_position,
                anchor_orientation=phase_proposals["A"].anchor_orientation,
                config=config,
                condition=CONDITIONS[0],
            )
            world_b, _ = _run_full_suffix(
                environment,
                prepared=prepared["B"]["prepared"],
                spec=prepared["B"]["spec"],
                proposal=phase_proposals["B"],
                anchor_position=phase_proposals["B"].anchor_position,
                anchor_orientation=phase_proposals["B"].anchor_orientation,
                config=config,
                condition=CONDITIONS[1],
            )
            registered_anchor_b = landmark_registered_point(
                phase_proposals["A"].anchor_position,
                prepared["A"]["normalized_bowl_position"],
                prepared["B"]["normalized_bowl_position"],
            )
            registered_b, acquisition_snapshot = _run_full_suffix(
                environment,
                prepared=prepared["B"]["prepared"],
                spec=prepared["B"]["spec"],
                proposal=phase_proposals["B"],
                anchor_position=registered_anchor_b,
                anchor_orientation=phase_proposals["A"].anchor_orientation,
                config=config,
                condition=CONDITIONS[2],
            )
            placement = _run_registered_placement(
                environment,
                acquisition_snapshot=acquisition_snapshot,
                reference_goal_state=world_a["first_goal_state"],
                target_cabinet_goal=prepared["B"]["cabinet_goal_position"],
                config=config,
            )
            result = {
                "schema_version": 1,
                "status": "complete",
                "contract_sha256": contract_sha,
                "proposal": {
                    "episode_index": proposal.episode_index,
                    "task_index": proposal.task_index,
                    "action_sha256": proposal.action_sha256,
                },
                "roots": {
                    layout: {
                        key: value
                        for key, value in prepared[layout].items()
                        if key
                        in {
                            "root_state_sha256",
                            "normalized_state_sha256",
                            "normalized_bowl_position",
                            "cabinet_goal_position",
                            "certificate",
                            "normalization",
                        }
                    }
                    for layout in ("A", "B")
                },
                "anchor_comparison": {
                    "layout_a_world_anchor": phase_proposals[
                        "A"
                    ].anchor_position,
                    "layout_b_world_anchor": phase_proposals[
                        "B"
                    ].anchor_position,
                    "layout_b_bowl_registered_anchor": registered_anchor_b,
                    "registration_delta_from_layout_b_world_anchor": (
                        registered_anchor_b
                        - phase_proposals["B"].anchor_position
                    ),
                },
                "conditions": [world_a, world_b, registered_b, placement],
            }
            result = _jsonable(result)
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
                    "condition_pass": {
                        item["condition"]: bool(
                            item.get("pass", item.get("goal_ever_achieved", False))
                        )
                        for item in result["conditions"]
                    },
                },
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            print(f"Layout-alignment diagnostic complete: {output_dir}")
        finally:
            environment.close()
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
            },
        )
        raise


if __name__ == "__main__":
    main()
