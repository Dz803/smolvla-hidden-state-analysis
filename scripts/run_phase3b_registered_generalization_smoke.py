#!/usr/bin/env python
"""Prospective two-root smoke for the frozen v35 bowl-registration rule."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from run_phase3b_stage_a import (
    PROJECT,
    _contract,
    _file_sha256,
    _load_config,
    _load_demos,
    _load_proposal_bank,
)
from smolvla_analysis.phase3_crd import atomic_write_json
from smolvla_analysis.phase3b_completion import proposal_inventory
from smolvla_analysis.phase3b_libero import (
    build_landmark_registered_action_phase_proposal_bank,
    build_support_reference_bank,
    certify_computational_state,
    compact_snapshot_metadata,
    construct_candidate,
    make_stage_a_environment,
    run_action_phase_oracle_from_prepared_root,
    run_goal_oracle,
)
from smolvla_analysis.phase3b_stage_a import (
    canonical_sha256,
    candidate_spec,
    snapshot_sha256,
)


DEFAULT_CONFIG = PROJECT / "configs/phase3b_stage_a_v35.yaml"
RAW_ROOT = PROJECT / "local/phase3b_stage_a/registered_generalization_smokes"
PROPOSAL_EPISODE = 474
CANDIDATE_IDS = (
    "stagea__drawer-open__possession-on-table__locus-drawer-side__"
    "support-transverse-low-support__layout-a",
    "stagea__drawer-open__possession-on-table__locus-drawer-side__"
    "support-transverse-low-support__layout-b",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen registered cabinet proposal on two untouched roots."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _output_dir(requested: Path | None) -> Path:
    if requested is None:
        stamp = datetime.now(UTC).strftime("registered_smoke_%Y%m%dT%H%M%SZ")
        requested = RAW_ROOT / stamp
    output = requested.resolve()
    raw_root = RAW_ROOT.resolve()
    if output == raw_root or raw_root not in output.parents:
        raise ValueError(f"Registered smoke must remain under {raw_root}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite registered smoke: {output}")
    output.mkdir(parents=True)
    return output


def _condition(
    environment,
    *,
    candidate_id: str,
    demos: dict[str, Any],
    support_bank: Any,
    proposal: Any,
    phase_proposal: Any,
    proposal_index: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    spec = candidate_spec(candidate_id)
    constructed = construct_candidate(
        environment,
        spec,
        demos,
        config,
        support_reference_bank=support_bank,
    )
    state_sha = snapshot_sha256(constructed.snapshot)
    certificate = certify_computational_state(
        environment,
        constructed.snapshot,
        possession=spec.possession,
        probe_actions=config["certificate"]["actions"],
    )
    if certificate["pass"] is not True:
        raise RuntimeError(f"Registered smoke certificate failed for {candidate_id}")
    preparation = run_goal_oracle(
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
        raise_on_failure=False,
        return_prepared_root=True,
        normalization_only=True,
    )
    prepared = preparation.pop("_prepared_oracle_root")
    attempt = run_action_phase_oracle_from_prepared_root(
        environment,
        constructed.snapshot,
        prepared,
        spec=spec,
        proposal=phase_proposal,
        config=config,
    )
    if phase_proposal.metadata["proposal_index"] != proposal_index:
        raise RuntimeError("Registered smoke proposal index changed")
    return {
        "candidate_id": candidate_id,
        "factors": spec.as_dict(),
        "root_state_sha256": state_sha,
        "snapshot_metadata": compact_snapshot_metadata(constructed.snapshot),
        "root_validation": constructed.root_validation,
        "root_geometry": constructed.root_geometry,
        "support_measurement": constructed.support_measurement,
        "certificate": certificate,
        "normalization": preparation,
        "proposal_index": proposal_index,
        "attempt": attempt,
        "pass": bool(attempt["pass"]),
    }


def main() -> None:
    args = _parse_args()
    config_path = args.config.resolve()
    output_dir = _output_dir(args.output_dir)
    config = _load_config(config_path)
    if config["construction_revision"] != "phase3b-stage-a-v35":
        raise ValueError("Registered smoke requires the locked v35 config")
    demos = _load_demos(config)
    proposal_banks = _load_proposal_bank(config)
    cabinet_bank = proposal_banks["cabinet"]
    proposal_index = next(
        index
        for index, proposal in enumerate(cabinet_bank)
        if proposal.episode_index == PROPOSAL_EPISODE
    )
    proposal = cabinet_bank[proposal_index]
    base_contract = _contract(config_path, config, demos, proposal_banks)
    contract = {
        "schema_version": 1,
        "diagnostic_revision": "phase3b-v35-registered-held-roots-v1",
        "selection_status": "prospective_after_freezing_on_prior_layout_pair",
        "candidate_ids": list(CANDIDATE_IDS),
        "proposal_index": proposal_index,
        "proposal_episode_index": PROPOSAL_EPISODE,
        "proposal_action_sha256": proposal.action_sha256,
        "full_cabinet_proposal_inventory_sha256": canonical_sha256(
            proposal_inventory(cabinet_bank)
        ),
        "v35_stage_a_contract_sha256": canonical_sha256(base_contract),
        "v35_stage_a_contract": base_contract,
        "policy_loaded": False,
        "canonical_rollout_reused": False,
        "future_reuse": (
            "Each full attempt may seed the same candidate/proposal index in the "
            "v35 completion shard after root and execution-contract validation."
        ),
        "source_sha256": {
            "script": _file_sha256(Path(__file__).resolve()),
            "runtime": _file_sha256(
                PROJECT / "src/smolvla_analysis/phase3b_libero.py"
            ),
            "lattice": _file_sha256(
                PROJECT / "src/smolvla_analysis/phase3b_stage_a.py"
            ),
            "completion": _file_sha256(
                PROJECT / "src/smolvla_analysis/phase3b_completion.py"
            ),
            "alignment": _file_sha256(
                PROJECT / "src/smolvla_analysis/phase3b_alignment.py"
            ),
        },
    }
    contract_sha = canonical_sha256(contract)
    atomic_write_json(output_dir / "contract.json", contract)
    try:
        environment = make_stage_a_environment(PROJECT, output_dir, config)
        try:
            support_bank = build_support_reference_bank(
                environment,
                {goal: demos[goal] for goal in ("drawer", "cabinet")},
                config,
            )
            phase_banks = {
                layout: build_landmark_registered_action_phase_proposal_bank(
                    environment,
                    target_layout=layout,
                    proposals=cabinet_bank,
                    config=config,
                )
                for layout in ("A", "B")
            }
            phase_contract_hashes = {
                layout: canonical_sha256(
                    [item.metadata for item in phase_banks[layout]]
                )
                for layout in phase_banks
            }
            conditions = [
                _condition(
                    environment,
                    candidate_id=candidate_id,
                    demos=demos,
                    support_bank=support_bank,
                    proposal=proposal,
                    phase_proposal=phase_banks[
                        candidate_spec(candidate_id).layout
                    ][proposal_index],
                    proposal_index=proposal_index,
                    config=config,
                )
                for candidate_id in CANDIDATE_IDS
            ]
            result = {
                "schema_version": 1,
                "status": "complete",
                "contract_sha256": contract_sha,
                "support_reference_bank_sha256": support_bank.sha256,
                "phase_proposal_contract_sha256_by_layout": (
                    phase_contract_hashes
                ),
                "condition_count": len(conditions),
                "pass_count": sum(condition["pass"] for condition in conditions),
                "all_pass": all(condition["pass"] for condition in conditions),
                "conditions": conditions,
            }
            atomic_write_json(output_dir / "result.json", result)
            artifacts = {
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
                    "all_pass": result["all_pass"],
                    "artifact_sha256": artifacts,
                    "policy_loaded": False,
                },
            )
            print(json.dumps(
                {
                    "output_dir": output_dir.as_posix(),
                    "all_pass": result["all_pass"],
                    "conditions": [
                        {
                            "candidate_id": item["candidate_id"],
                            "root_state_sha256": item["root_state_sha256"],
                            "normalized_state_sha256": item["attempt"][
                                "normalized_state_sha256"
                            ],
                            "pass": item["pass"],
                            "bridge_pass": item["attempt"]["phases"][
                                "action_phase_bridge"
                            ]["pass"],
                            "goal_ever_achieved": item["attempt"][
                                "goal_ever_achieved"
                            ],
                            "first_goal_demo_frame": item["attempt"][
                                "first_goal_demo_frame"
                            ],
                        }
                        for item in conditions
                    ],
                },
                indent=2,
                sort_keys=True,
            ))
            print(f"Registered generalization smoke complete: {output_dir}")
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
                "policy_loaded": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
