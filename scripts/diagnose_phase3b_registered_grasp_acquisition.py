#!/usr/bin/env python
"""Test phase-separated drawer opening and registered bowl acquisition."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path

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
from smolvla_analysis.phase3_crd import atomic_write_json
from smolvla_analysis.phase3b_libero import (
    PolicyFreeController,
    _transport_validation_limits,
    _validate_grasped_transport_phase,
    _validate_servo_phase,
    build_landmark_registered_action_phase_proposal_bank,
    evaluate_common_goals,
    grasped_root_transit_plan,
    make_stage_a_environment,
)
from smolvla_analysis.phase3b_stage_a import canonical_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test registered acquisition after independently opening the drawer."
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "configs/phase3b_stage_a_v35.yaml"
    )
    parser.add_argument("--episode-index", type=int, default=474)
    parser.add_argument("--stable-grasp-steps", type=int, default=3)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _layout_id(layout: str) -> int:
    return {"A": 0, "B": 1}[layout]


def _run_layout(
    *,
    layout: str,
    config: dict,
    demos: dict,
    proposal,
    output_dir: Path,
    stable_grasp_steps: int,
) -> dict:
    environment = make_stage_a_environment(PROJECT, output_dir, config)
    try:
        controller = PolicyFreeController(environment)
        controller.reset_layout(
            _layout_id(layout), int(config["environment"]["reset_seed"])
        )
        prefix = controller.replay_until(
            demos["drawer_construction"],
            condition=lambda: controller.top_drawer_position()
            <= float(config["construction"]["open_drawer_threshold"]),
        )
        if (
            controller.top_drawer_position()
            > float(config["construction"]["open_drawer_threshold"])
            or controller.bowl_grasped()
            or any(controller.done_values)
            or any(any(item.values()) for item in controller.goal_values)
        ):
            raise RuntimeError("Independent drawer-opening prefix is invalid")

        phase_cfg = config["action_phase_oracle"]
        initial_bowl_position = controller.bowl_position()
        bridge_start = len(controller.actions)
        route = grasped_root_transit_plan(
            controller.eef_position(),
            proposal.anchor_position,
            clearance_margin_m=float(phase_cfg["clearance_margin_m"]),
            workspace_bounds=phase_cfg["workspace_bounds_m"],
            phase_budgets=phase_cfg["bridge_phase_budgets"],
        )
        bridge_phases = []
        for phase in route:
            intermediate = phase["phase"] != "target_descent"
            result = controller.servo(
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
            _validate_servo_phase(
                result,
                candidate_id=f"registered-acquisition-{layout}",
                phase=f"bridge_{phase['phase']}",
            )
            bridge_phases.append(
                {
                    "phase": phase["phase"],
                    "target_position": np.asarray(
                        phase["target_position"]
                    ).tolist(),
                    "result": result,
                }
            )
        bridge_bowl_drift = float(
            np.linalg.norm(controller.bowl_position() - initial_bowl_position)
        )
        if (
            any(controller.grasp_values[bridge_start:])
            or any(controller.done_values[bridge_start:])
            or any(
                any(item.values())
                for item in controller.goal_values[bridge_start:]
            )
            or bridge_bowl_drift
            > float(phase_cfg["bowl_drift_tolerance_m"])
            or controller.top_drawer_position()
            > float(config["construction"]["open_drawer_threshold"])
        ):
            raise RuntimeError("Registered acquisition bridge changed the task state")

        acquisition_start = len(controller.actions)
        grasp_streak = 0
        acquired_at_frame = None
        for frame_index, action in zip(
            proposal.suffix.frame_indices,
            proposal.suffix.actions,
            strict=True,
        ):
            _, _, done, _ = controller.step(action)
            if done or any(evaluate_common_goals(environment).values()):
                raise RuntimeError("Registered acquisition crossed a terminal or goal")
            if controller.bowl_grasped():
                grasp_streak += 1
            else:
                grasp_streak = 0
            if grasp_streak >= stable_grasp_steps:
                acquired_at_frame = int(frame_index)
                break
        if acquired_at_frame is None:
            raise RuntimeError("Registered continuation never acquired a stable grasp")
        acquired_relative = controller.eef_position() - controller.bowl_position()

        safe_lift_start = len(controller.actions)
        safe_lift = controller.servo(
            target_position=controller.eef_position()
            + np.asarray(
                [0.0, 0.0, float(config["construction"]["safe_lift_m"])]
            ),
            target_orientation=controller.eef_orientation(),
            gripper=1.0,
            budget=int(config["construction"]["safe_lift_budget"]),
            max_translation_action=float(
                config["construction"]["max_translation_action_grasped"]
            ),
            position_tolerance_m=float(
                config["construction"]["root_tolerance_m"]
            ),
        )
        _validate_grasped_transport_phase(
            controller,
            safe_lift,
            candidate_id=f"registered-acquisition-{layout}",
            phase="safe_lift",
            **_transport_validation_limits(config),
        )
        final_relative = controller.eef_position() - controller.bowl_position()
        return {
            "layout": layout,
            "pass": True,
            "drawer_prefix": prefix,
            "bridge": {
                "phases": bridge_phases,
                "action_count": len(controller.actions[bridge_start:acquisition_start]),
                "bowl_drift_m": bridge_bowl_drift,
            },
            "acquisition": {
                "source_episode_index": proposal.source.episode_index,
                "source_action_sha256": proposal.source.action_sha256,
                "phase_proposal": proposal.metadata,
                "executed_suffix_action_count": (
                    safe_lift_start - acquisition_start
                ),
                "acquired_at_source_frame": acquired_at_frame,
                "required_stable_grasp_steps": stable_grasp_steps,
                "relative_position_m": acquired_relative.tolist(),
            },
            "safe_lift": safe_lift,
            "post_lift_relative_position_m": final_relative.tolist(),
            "post_lift_relative_delta_m": float(
                np.linalg.norm(final_relative - acquired_relative)
            ),
            "final_drawer_joint": controller.top_drawer_position(),
            "final_goals": evaluate_common_goals(environment),
            "final_grasped": controller.bowl_grasped(),
        }
    finally:
        environment.close()


def main() -> None:
    args = _parse_args()
    if args.stable_grasp_steps < 1:
        raise ValueError("--stable-grasp-steps must be positive")
    config_path = args.config.resolve()
    config = _load_config(config_path)
    demos = _load_demos(config)
    proposal_bank = _load_proposal_bank(config)["cabinet"]
    matches = [
        item for item in proposal_bank if item.episode_index == args.episode_index
    ]
    if len(matches) != 1:
        raise ValueError("Registered acquisition episode identity is not unique")
    source = matches[0]
    run_id = datetime.now(UTC).strftime(
        "registered_grasp_acquisition_%Y%m%dT%H%M%SZ"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else PROJECT
        / "local/phase3b_stage_a/construction_diagnostics"
        / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    contract = {
        "schema_version": 1,
        "diagnostic_revision": "phase3b-registered-grasp-acquisition-v1",
        "base_config": config_path.relative_to(PROJECT).as_posix(),
        "base_config_sha256": _file_sha256(config_path),
        "source_episode_index": source.episode_index,
        "source_action_sha256": source.action_sha256,
        "stable_grasp_steps": args.stable_grasp_steps,
        "layouts": ["A", "B"],
        "runner_sha256": _file_sha256(Path(__file__).resolve()),
        "libero_source_sha256": _file_sha256(
            PROJECT / "src/smolvla_analysis/phase3b_libero.py"
        ),
        "policy_loaded": False,
        "proposal_oracles_executed": 0,
    }
    contract_sha = canonical_sha256(contract)
    atomic_write_json(output_dir / "contract.json", contract)

    phase_proposals = {}
    for layout in ("A", "B"):
        phase_environment = make_stage_a_environment(PROJECT, output_dir, config)
        try:
            (phase_proposal,) = build_landmark_registered_action_phase_proposal_bank(
                phase_environment,
                target_layout=layout,
                proposals=(source,),
                config=config,
            )
        finally:
            phase_environment.close()
        phase_proposals[layout] = phase_proposal

    results = []
    for layout in ("A", "B"):
        try:
            results.append(
                _run_layout(
                    layout=layout,
                    config=config,
                    demos=demos,
                    proposal=phase_proposals[layout],
                    output_dir=output_dir,
                    stable_grasp_steps=args.stable_grasp_steps,
                )
            )
        except Exception as exc:
            results.append(
                {
                    "layout": layout,
                    "pass": False,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    result = {
        "schema_version": 1,
        "run_id": run_id,
        "contract_sha256": contract_sha,
        "all_pass": all(item["pass"] for item in results),
        "conditions": results,
        "policy_loaded": False,
        "proposal_oracles_executed": 0,
    }
    atomic_write_json(output_dir / "result.json", result)
    artifact_sha256 = {
        name: _file_sha256(output_dir / name)
        for name in ("contract.json", "result.json")
    }
    atomic_write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "status": "complete",
            "all_pass": result["all_pass"],
            "contract_sha256": contract_sha,
            "artifact_sha256": artifact_sha256,
            "policy_loaded": False,
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Registered grasp diagnostic complete: {output_dir}")


if __name__ == "__main__":
    main()
