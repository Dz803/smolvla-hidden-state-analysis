#!/usr/bin/env python
"""Observe grasp continuity after bypassing only its diagnostic threshold."""

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
    )
except ModuleNotFoundError:
    from scripts.run_phase3b_stage_a import (
        PROJECT,
        _file_sha256,
        _load_config,
        _load_demos,
    )
from smolvla_analysis.phase3_crd import atomic_write_json
import smolvla_analysis.phase3b_libero as phase3b_libero
from smolvla_analysis.phase3b_stage_a import (
    GOALS,
    candidate_spec,
    canonical_sha256,
    snapshot_sha256,
)


DEFAULT_CANDIDATE = (
    "stagea__drawer-open__possession-grasped__locus-drawer-side__"
    "support-demonstration-near__layout-a"
)


class TracingController(phase3b_libero.PolicyFreeController):
    instances: list["TracingController"] = []

    def __init__(self, environment):
        super().__init__(environment)
        self.eef_positions: list[np.ndarray] = []
        self.bowl_positions: list[np.ndarray] = []
        self.servo_traces: list[dict] = []
        self.__class__.instances.append(self)

    def step(self, action):
        result = super().step(action)
        self.eef_positions.append(self.eef_position())
        self.bowl_positions.append(self.bowl_position())
        return result

    def servo(self, **kwargs):
        start = len(self.actions)
        start_relative = self.eef_position() - self.bowl_position()
        result = super().servo(**kwargs)
        stop = len(self.actions)
        relative = self.grasp_relative_positions[start:stop]
        deviations = [
            float(np.linalg.norm(position - start_relative))
            for position in relative
        ]
        self.servo_traces.append(
            {
                "validation_phase": None,
                "action_start": start,
                "action_stop": stop,
                "target_position": np.asarray(
                    kwargs["target_position"], dtype=np.float64
                ).tolist(),
                "budget": int(kwargs["budget"]),
                "max_translation_action": float(
                    kwargs["max_translation_action"]
                ),
                "position_tolerance_m": float(
                    kwargs["position_tolerance_m"]
                ),
                "result": result,
                "start_relative_position_m": start_relative.tolist(),
                "relative_positions_m": [item.tolist() for item in relative],
                "relative_deviations_m": deviations,
                "grasp_values": self.grasp_values[start:stop],
                "eef_positions_m": [
                    item.tolist() for item in self.eef_positions[start:stop]
                ],
                "bowl_positions_m": [
                    item.tolist() for item in self.bowl_positions[start:stop]
                ],
            }
        )
        return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace a failed grasp route without changing production gates."
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "configs/phase3b_stage_a_v35.yaml"
    )
    parser.add_argument("--candidate-id", default=DEFAULT_CANDIDATE)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _dropout_runs(values: list[bool]) -> list[list[int]]:
    runs = []
    active = []
    for index, value in enumerate(values):
        if value:
            if active:
                runs.append(active)
                active = []
        else:
            active.append(index)
    if active:
        runs.append(active)
    return runs


def _trace_summary(trace: dict) -> dict:
    deviations = trace["relative_deviations_m"]
    grasps = trace["grasp_values"]
    maximum_index = int(np.argmax(deviations)) if deviations else None
    final_delta = (
        float(
            np.linalg.norm(
                np.asarray(trace["relative_positions_m"][-1])
                - np.asarray(trace["start_relative_position_m"])
            )
        )
        if deviations
        else 0.0
    )
    return {
        "validation_phase": trace["validation_phase"],
        "budget": trace["budget"],
        "active_action_steps": trace["result"]["active_action_steps"],
        "grasp_false_indices": [
            index for index, value in enumerate(grasps) if not value
        ],
        "grasp_dropout_runs": _dropout_runs(grasps),
        "max_relative_deviation_m": max(deviations, default=0.0),
        "max_relative_deviation_index": maximum_index,
        "final_relative_delta_m": final_delta,
        "final_grasped": bool(grasps[-1]) if grasps else False,
        "final_position_error_m": trace["result"]["final_position_error_m"],
    }


def main() -> None:
    args = _parse_args()
    config_path = args.config.resolve()
    config = _load_config(config_path)
    spec = candidate_spec(args.candidate_id)
    if spec.possession != "grasped":
        raise ValueError("Continuity diagnostics require a grasped candidate")
    demos = _load_demos(config)
    run_id = datetime.now(UTC).strftime(
        "grasp_continuity_%Y%m%dT%H%M%SZ"
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
        "diagnostic_revision": "phase3b-grasp-continuity-observer-v1",
        "candidate_id": spec.candidate_id,
        "base_config": config_path.relative_to(PROJECT).as_posix(),
        "base_config_sha256": _file_sha256(config_path),
        "runner_sha256": _file_sha256(Path(__file__).resolve()),
        "libero_source_sha256": _file_sha256(
            PROJECT / "src/smolvla_analysis/phase3b_libero.py"
        ),
        "production_continuity_gate_changed": False,
        "diagnostic_observer": (
            "Require servo/goal/terminal/final-grasp gates, record but do not "
            "raise on intermediate contact or relative-pose continuity."
        ),
        "policy_loaded": False,
        "proposal_oracles_executed": 0,
    }
    contract_sha = canonical_sha256(contract)
    atomic_write_json(output_dir / "contract.json", contract)

    bank_environment = phase3b_libero.make_stage_a_environment(
        PROJECT, output_dir, config
    )
    try:
        support_bank = phase3b_libero.build_support_reference_bank(
            bank_environment,
            {goal: demos[goal] for goal in GOALS},
            config,
        )
    finally:
        bank_environment.close()

    original_controller = phase3b_libero.PolicyFreeController
    original_validator = phase3b_libero._validate_grasped_transport_phase

    def observe_validation(controller, result, *, candidate_id, phase, **limits):
        del limits
        phase3b_libero._validate_servo_phase(
            result, candidate_id=candidate_id, phase=phase
        )
        if not controller.bowl_grasped():
            raise RuntimeError(
                f"Diagnostic route ended without the bowl at {phase}: {result}"
            )
        controller.servo_traces[-1]["validation_phase"] = phase

    phase3b_libero.PolicyFreeController = TracingController
    phase3b_libero._validate_grasped_transport_phase = observe_validation
    environment = phase3b_libero.make_stage_a_environment(
        PROJECT, output_dir, config
    )
    try:
        try:
            constructed = phase3b_libero.construct_candidate(
                environment,
                spec,
                demos,
                config,
                support_reference_bank=support_bank,
            )
            certificate = phase3b_libero.certify_computational_state(
                environment,
                constructed.snapshot,
                possession=spec.possession,
                probe_actions=config["certificate"]["actions"],
            )
            status = "observed_complete"
            exception = None
            state_sha = snapshot_sha256(constructed.snapshot)
        except Exception as exc:
            status = "observed_failed"
            exception = {
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            certificate = None
            state_sha = None
        controllers = [
            item
            for item in TracingController.instances
            if item.environment is environment and item.servo_traces
        ]
        if len(controllers) != 1:
            raise RuntimeError("Continuity observer found an invalid controller count")
        traces = controllers[0].servo_traces
    finally:
        environment.close()
        phase3b_libero.PolicyFreeController = original_controller
        phase3b_libero._validate_grasped_transport_phase = original_validator

    result = {
        "schema_version": 1,
        "run_id": run_id,
        "status": status,
        "contract_sha256": contract_sha,
        "support_reference_bank_sha256": support_bank.sha256,
        "state_sha256": state_sha,
        "certificate": certificate,
        "exception": exception,
        "trace_summaries": [_trace_summary(trace) for trace in traces],
        "traces": traces,
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
            "observed_status": status,
            "contract_sha256": contract_sha,
            "artifact_sha256": artifact_sha256,
            "policy_loaded": False,
        },
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "run_id",
                    "status",
                    "state_sha256",
                    "exception",
                    "trace_summaries",
                    "policy_loaded",
                    "proposal_oracles_executed",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"Grasp-continuity diagnostic complete: {output_dir}")


if __name__ == "__main__":
    main()
