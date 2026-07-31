#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from run_phase3b_stage_a import (
    DEFAULT_CONFIG,
    PROJECT,
    _dataset_root,
    _file_sha256,
    _load_config,
    _load_demos,
)
from smolvla_analysis.libero_state import restore_libero_state
from smolvla_analysis.phase3_crd import atomic_write_json
from smolvla_analysis.phase3b_libero import (
    PreparedOracleRoot,
    certify_computational_state,
    construct_candidate,
    list_demo_trace_inventory,
    load_demo_trace,
    make_stage_a_environment,
    replay_goal_proposal,
    run_goal_oracle,
)
from smolvla_analysis.phase3b_stage_a import candidate_spec, snapshot_sha256


RAW_ROOT = PROJECT / "local/phase3b_stage_a/proposal_scans"
GOAL_TASKS = {"drawer": 12, "cabinet": 18}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan cached human demonstrations from one certified, normalized "
            "Stage A root without loading an evaluated policy."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--goal", choices=tuple(GOAL_TASKS), required=True)
    parser.add_argument("--scan-dir", type=Path)
    return parser.parse_args()


def _scan_directory(args: argparse.Namespace) -> Path:
    if args.scan_dir is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = RAW_ROOT / f"{args.goal}_{stamp}"
    else:
        path = args.scan_dir.resolve()
    raw_root = RAW_ROOT.resolve()
    path = path.resolve()
    if path != raw_root and raw_root not in path.parents:
        raise ValueError(f"Proposal scans must remain under {raw_root}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _episode_inventory(dataset_root: Path, task_index: int) -> list[dict[str, int]]:
    return [
        dict(row)
        for row in list_demo_trace_inventory(
            dataset_root, task_index=task_index
        )
    ]


def _inventory_sha256(inventory: list[dict[str, int]]) -> str:
    payload = json.dumps(
        inventory, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _load_existing_attempts(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    attempts = json.loads(path.read_text())
    if not isinstance(attempts, list):
        raise ValueError("Existing proposal attempts are not a list")
    episode_ids = [int(row["episode_index"]) for row in attempts]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("Existing proposal attempts contain duplicate episodes")
    return attempts


def _validate_resume_manifest(
    manifest: dict[str, Any], expected: dict[str, Any]
) -> None:
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"Proposal-scan resume mismatch for {key}: "
                f"{manifest.get(key)!r} != {value!r}"
            )


def _attempt_record(
    environment,
    *,
    prepared: PreparedOracleRoot,
    goal: str,
    demo,
) -> dict[str, Any]:
    restore_libero_state(environment, prepared.snapshot)
    try:
        controller, outcome = replay_goal_proposal(
            environment,
            goal=goal,
            demo=demo,
        )
    finally:
        restore_libero_state(environment, prepared.snapshot)
    return {
        "episode_index": demo.episode_index,
        "task_index": demo.task_index,
        "frame_count": int(len(demo.actions)),
        "action_sha256": demo.action_sha256,
        **outcome,
        "executed_demonstration_action_steps": int(len(controller.actions)),
        "eef_path_length_m": controller.eef_path_length_m,
        "control_effort": controller.control_effort,
        "motion_control_effort": controller.motion_control_effort,
    }


def main() -> None:
    args = _parse_args()
    config_path = args.config.resolve()
    config = _load_config(config_path)
    spec = candidate_spec(args.candidate_id)
    goal = str(args.goal)
    task_index = GOAL_TASKS[goal]
    dataset_root = _dataset_root(config)
    inventory = _episode_inventory(dataset_root, task_index)
    inventory_hash = _inventory_sha256(inventory)
    scan_dir = _scan_directory(args)
    manifest_path = scan_dir / "manifest.json"
    attempts_path = scan_dir / "attempts.json"
    expected_manifest = {
        "schema_version": 1,
        "scan_id": scan_dir.name,
        "stage": "phase3b_stage_a_proposal_scan",
        "candidate_id": spec.candidate_id,
        "goal": goal,
        "task_index": task_index,
        "config_path": config_path.relative_to(PROJECT).as_posix(),
        "config_sha256": _file_sha256(config_path),
        "construction_revision": config["construction_revision"],
        "proposal_inventory_sha256": inventory_hash,
        "proposal_count": len(inventory),
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
            "libero_state": _file_sha256(
                PROJECT / "src/smolvla_analysis/libero_state.py"
            ),
        },
    }
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        _validate_resume_manifest(manifest, expected_manifest)
    else:
        manifest = {
            **expected_manifest,
            "status": "in_progress",
            "attempt_count": 0,
            "success_count": 0,
            "created_at": datetime.now(UTC).isoformat(),
        }
        atomic_write_json(manifest_path, manifest)

    demos = _load_demos(config)
    reference_config = inventory[0]
    reference_demo = load_demo_trace(
        dataset_root,
        goal=goal,
        episode_index=int(reference_config["episode_index"]),
        task_index=task_index,
    )
    environment = make_stage_a_environment(PROJECT, scan_dir, config)
    try:
        constructed = construct_candidate(
            environment,
            spec,
            demos,
            config,
            support_reference_bank=None,
        )
        root_certificate = certify_computational_state(
            environment,
            constructed.snapshot,
            possession=spec.possession,
            probe_actions=config["certificate"]["actions"],
        )
        if not root_certificate["pass"]:
            raise RuntimeError("Proposal-scan root certificate failed")
        reference_result = run_goal_oracle(
            environment,
            constructed.snapshot,
            spec=spec,
            goal=goal,
            demo=reference_demo,
            initial_bowl_position=constructed.initial_bowl_position,
            initial_eef_position=constructed.initial_eef_position,
            initial_eef_orientation=constructed.initial_eef_orientation,
            initial_joint_positions=constructed.initial_joint_positions,
            recovery_waypoints=constructed.recovery_waypoints,
            config=config,
            raise_on_failure=False,
            return_prepared_root=True,
        )
        prepared = reference_result.pop("_prepared_oracle_root")
        if not isinstance(prepared, PreparedOracleRoot):
            raise TypeError("Oracle did not return a prepared normalized root")
        normalized_state_hash = snapshot_sha256(prepared.snapshot)
        if normalized_state_hash != reference_result["normalized_state_sha256"]:
            raise RuntimeError("Prepared normalized-root hash changed before scanning")
        normalized_certificate = certify_computational_state(
            environment,
            prepared.snapshot,
            possession="on_table",
            probe_actions=config["certificate"]["actions"],
        )
        if not normalized_certificate["pass"]:
            raise RuntimeError("Proposal-scan normalized-root certificate failed")

        root_metadata = {
            "root_state_sha256": snapshot_sha256(constructed.snapshot),
            "normalized_state_sha256": normalized_state_hash,
            "normalization_action_sha256": reference_result[
                "normalization_action_sha256"
            ],
            "normalization_action_steps": reference_result[
                "normalization_action_steps"
            ],
            "normalized_bowl_position_error_m": reference_result[
                "normalized_bowl_position_error_m"
            ],
            "root_certificate": root_certificate,
            "normalized_certificate": normalized_certificate,
        }
        if "root_metadata" in manifest and manifest["root_metadata"] != root_metadata:
            raise ValueError("Proposal-scan reconstructed root does not match its manifest")
        manifest["root_metadata"] = root_metadata
        atomic_write_json(manifest_path, manifest)

        attempts = _load_existing_attempts(attempts_path)
        existing = {int(row["episode_index"]): row for row in attempts}
        expected_ids = {row["episode_index"] for row in inventory}
        if not set(existing).issubset(expected_ids):
            raise ValueError("Existing proposal attempts are outside the locked inventory")
        for item in inventory:
            episode_index = item["episode_index"]
            demo = load_demo_trace(
                dataset_root,
                goal=goal,
                episode_index=episode_index,
                task_index=task_index,
            )
            if len(demo.actions) != item["frame_count"]:
                raise RuntimeError(f"Episode {episode_index} length changed")
            if episode_index in existing:
                if existing[episode_index]["action_sha256"] != demo.action_sha256:
                    raise ValueError(f"Episode {episode_index} action hash changed")
                print(f"skip complete episode {episode_index}", flush=True)
                continue
            attempt = _attempt_record(
                environment,
                prepared=prepared,
                goal=goal,
                demo=demo,
            )
            attempts.append(attempt)
            existing[episode_index] = attempt
            attempts.sort(
                key=lambda row: (
                    next(
                        index
                        for index, locked in enumerate(inventory)
                        if locked["episode_index"] == row["episode_index"]
                    )
                )
            )
            atomic_write_json(attempts_path, attempts)
            manifest.update(
                {
                    "status": "in_progress",
                    "attempt_count": len(attempts),
                    "success_count": sum(bool(row["pass"]) for row in attempts),
                    "last_completed_episode": episode_index,
                    "updated_at": datetime.now(UTC).isoformat(),
                }
            )
            atomic_write_json(manifest_path, manifest)
            print(
                f"episode {episode_index}: pass={attempt['pass']} "
                f"goal_frame={attempt['first_goal_demo_frame']}",
                flush=True,
            )

        attempts = _load_existing_attempts(attempts_path)
        if {int(row["episode_index"]) for row in attempts} != expected_ids:
            raise RuntimeError("Proposal scan ended with an incomplete episode ledger")
        successful = [row for row in attempts if bool(row["pass"])]
        manifest.update(
            {
                "status": "complete",
                "attempt_count": len(attempts),
                "success_count": len(successful),
                "successful_episode_indices": [
                    int(row["episode_index"]) for row in successful
                ],
                "completed_at": datetime.now(UTC).isoformat(),
            }
        )
        atomic_write_json(manifest_path, manifest)
        print(
            f"complete: {len(successful)}/{len(attempts)} proposals pass; "
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
