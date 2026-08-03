#!/usr/bin/env python
"""Consolidate non-duplicative Stage A shards into one verified compact lattice."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
import zarr

from smolvla_analysis.phase2_storage import read_libero_snapshot
from smolvla_analysis.phase3_crd import atomic_write_json
from smolvla_analysis.phase3b_consolidation import (
    CONSOLIDATION_MIGRATION,
    migrate_legacy_full_replay_record,
    validate_consolidated_records,
    validate_source_assignment,
)
from smolvla_analysis.phase3b_stage_a import (
    GOALS,
    canonical_sha256,
    iter_candidate_specs,
    snapshot_sha256,
    validate_selection_lock,
)


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT / "configs/phase3b_stage_a_consolidation.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and consolidate the non-duplicative 32-state Stage A lattice."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else PROJECT / path
    return path.resolve()


def _selection_candidate_ids(selection: dict[str, Any]) -> list[str]:
    if set(selection) == {"candidate_ids"}:
        candidate_ids = selection["candidate_ids"]
        if not isinstance(candidate_ids, list) or not candidate_ids:
            raise ValueError("Explicit source selection must be a nonempty list")
        return [str(candidate_id) for candidate_id in candidate_ids]
    if set(selection) == {"factor_equals"}:
        factors = selection["factor_equals"]
        if not isinstance(factors, dict) or not factors:
            raise ValueError("Factor source selection must be a nonempty mapping")
        selected = []
        for spec in iter_candidate_specs():
            values = spec.as_dict()
            if all(values.get(key) == value for key, value in factors.items()):
                selected.append(spec.candidate_id)
        if not selected:
            raise ValueError(
                f"Factor source selection matched no candidates: {factors}"
            )
        return selected
    raise ValueError(f"Unsupported source selection: {selection}")


def _load_source(
    source: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    source_id = str(source["source_id"])
    run_dir = _resolve_project_path(str(source["run_dir"]))
    raw_root = (PROJECT / "local/phase3b_stage_a").resolve()
    if run_dir == raw_root or raw_root not in run_dir.parents:
        raise ValueError(f"Source {source_id} is outside the Stage A raw root")
    required = ("manifest.json", "contract.json", "selection_lock.json")
    if any(not (run_dir / name).is_file() for name in required):
        raise FileNotFoundError(f"Source run is incomplete: {run_dir}")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    contract = json.loads((run_dir / "contract.json").read_text())
    selection_lock = json.loads((run_dir / "selection_lock.json").read_text())
    contract_sha = canonical_sha256(contract)
    if manifest.get("contract_sha256") != contract_sha:
        raise ValueError(f"Source contract hash mismatch for {source_id}")
    if (
        manifest.get("policy_loaded") is not False
        or contract.get("policy_loaded") is not False
    ):
        raise ValueError(f"Source {source_id} crossed the policy-free boundary")
    validate_selection_lock(
        selection_lock,
        contract_sha256=contract_sha,
        construction_revision=contract["construction_revision"],
    )
    if selection_lock["selection_lock_sha256"] != manifest["selection_lock_sha256"]:
        raise ValueError(f"Source selection-lock mismatch for {source_id}")
    candidate_ids = _selection_candidate_ids(source["selection"])
    return manifest, contract, selection_lock, candidate_ids


def _load_candidate_entry(
    *,
    source_id: str,
    run_dir: Path,
    manifest: dict[str, Any],
    candidate_id: str,
) -> dict[str, Any]:
    record_path = run_dir / "candidates" / f"{candidate_id}.json"
    state_path = run_dir / "states.zarr" / candidate_id
    if not record_path.is_file() or not state_path.is_dir():
        raise FileNotFoundError(
            f"Missing selected candidate artifact: {source_id}/{candidate_id}"
        )
    raw_record = json.loads(record_path.read_text())
    if raw_record.get("candidate_id") != candidate_id:
        raise ValueError(f"Candidate identity mismatch in {record_path}")
    if raw_record.get("contract_sha256") != manifest["contract_sha256"]:
        raise ValueError(f"Candidate contract mismatch for {candidate_id}")
    if raw_record.get("selection_lock_sha256") != manifest["selection_lock_sha256"]:
        raise ValueError(f"Candidate selection-lock mismatch for {candidate_id}")
    state_root = zarr.open_group(str(run_dir / "states.zarr"), mode="r")
    attrs = state_root[candidate_id].attrs
    if not attrs.get("complete") or attrs.get("state_sha256") != raw_record.get(
        "state_sha256"
    ):
        raise ValueError(f"Candidate state attributes changed for {candidate_id}")
    snapshot = read_libero_snapshot(state_root, candidate_id)
    observed_state_sha = snapshot_sha256(snapshot)
    if observed_state_sha != raw_record.get("state_sha256"):
        raise ValueError(f"Candidate state payload changed for {candidate_id}")
    migrated_record, migrated_goals = migrate_legacy_full_replay_record(raw_record)
    return {
        "source_id": source_id,
        "candidate_id": candidate_id,
        "record": migrated_record,
        "raw_record_sha256": _file_sha256(record_path),
        "raw_state_sha256": observed_state_sha,
        "schema_migration": (
            CONSOLIDATION_MIGRATION if migrated_goals else None
        ),
        "schema_migrated_goals": migrated_goals,
    }


def _candidate_row(
    entry: dict[str, Any], source_meta: dict[str, Any]
) -> dict[str, Any]:
    record = entry["record"]
    row = {
        **record["factors"],
        "source_id": entry["source_id"],
        "source_run_id": source_meta["run_id"],
        "source_construction_revision": record["construction_revision"],
        "root_final_timestep": int(record["construction"]["final_timestep"]),
        "source_record_sha256": entry["raw_record_sha256"],
        "state_sha256": record["state_sha256"],
        "schema_migration": entry["schema_migration"] or "none",
        "root_validation_pass": bool(record["root_validation"]["pass"]),
        "certificate_pass": bool(record["certificate"]["pass"]),
        "joint_support_distance": float(
            record["support_measurement"]["nearest"]["distance"]
        ),
        "exact_event_reference_count": int(
            record["support_measurement"]["event_matching_reference_count"]
        ),
    }
    for goal in GOALS:
        oracle = record["oracles"][goal]
        row.update(
            {
                f"{goal}_execution_mode": oracle["proposal_execution_mode"],
                f"{goal}_success_count": int(oracle["proposal_success_count"]),
                f"{goal}_proposal_count": int(oracle["proposal_attempt_count"]),
                f"{goal}_success_fraction": float(
                    oracle["proposal_success_fraction"]
                ),
                f"{goal}_selected_proposal_index": int(
                    oracle["selected_proposal_index"]
                ),
                f"{goal}_selected_episode_index": int(
                    oracle["demo_episode_index"]
                ),
                f"{goal}_selected_budgeted_steps": int(
                    oracle["cost"]["budgeted_action_steps"]
                ),
                f"{goal}_selected_executed_steps": int(
                    oracle["cost"]["executed_action_steps"]
                ),
                f"{goal}_selected_eef_path_m": float(
                    oracle["cost"]["eef_path_length_m"]
                ),
            }
        )
    return row


def _proposal_rows(entry: dict[str, Any]) -> list[dict[str, Any]]:
    record = entry["record"]
    rows = []
    for goal in GOALS:
        oracle = record["oracles"][goal]
        selected = int(oracle["selected_proposal_index"])
        for proposal, attempt in zip(
            oracle["proposal_bank"], oracle["proposal_attempts"], strict=True
        ):
            rows.append(
                {
                    "candidate_id": record["candidate_id"],
                    "source_id": entry["source_id"],
                    "goal": goal,
                    "execution_mode": oracle["proposal_execution_mode"],
                    "proposal_index": int(proposal["proposal_index"]),
                    "episode_index": int(proposal["episode_index"]),
                    "task_index": int(proposal["task_index"]),
                    "action_sha256": proposal["action_sha256"],
                    "pass": bool(attempt["pass"]),
                    "selected": int(proposal["proposal_index"]) == selected,
                    "goal_ever_achieved": bool(attempt["goal_ever_achieved"]),
                    "wrong_goal_ever_achieved": bool(
                        attempt["wrong_goal_ever_achieved"]
                    ),
                    "unexpected_done_before_goal": bool(
                        attempt["unexpected_done_before_goal"]
                    ),
                    "executed_action_steps": int(
                        attempt["cost"]["executed_action_steps"]
                    ),
                    "eef_path_length_m": float(
                        attempt["cost"]["eef_path_length_m"]
                    ),
                }
            )
    return rows


def _oracle_counterfactual_rows(entry: dict[str, Any]) -> list[dict[str, Any]]:
    record = entry["record"]
    rows = []
    for goal, prior in record.get(
        "prior_negative_oracle_evidence", {}
    ).items():
        current = record["oracles"][goal]
        rows.append(
            {
                "candidate_id": record["candidate_id"],
                "source_id": entry["source_id"],
                "goal": goal,
                "identical_normalized_state": bool(
                    prior["normalized_state_sha256"]
                    == current["shared_normalized_state_sha256"]
                ),
                "normalized_state_sha256": current[
                    "shared_normalized_state_sha256"
                ],
                "prior_execution_mode": prior["proposal_execution_mode"],
                "prior_proposal_count": int(prior["proposal_count"]),
                "prior_success_count": int(
                    prior["successful_proposal_count"]
                ),
                "current_execution_mode": current[
                    "proposal_execution_mode"
                ],
                "current_proposal_count": int(
                    current["proposal_attempt_count"]
                ),
                "current_success_count": int(
                    current["proposal_success_count"]
                ),
                "source_checkpoint_file_sha256": prior[
                    "source_checkpoint_file_sha256"
                ],
            }
        )
    return rows


def main() -> None:
    args = _parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("Consolidation config must have schema_version=1")
    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Consolidation config has no sources")

    loaded_sources = {}
    assignments = {}
    for source in sources:
        source_id = str(source["source_id"])
        if source_id in loaded_sources:
            raise ValueError(f"Duplicate source ID: {source_id}")
        manifest, contract, selection_lock, candidate_ids = _load_source(source)
        run_dir = _resolve_project_path(str(source["run_dir"]))
        loaded_sources[source_id] = {
            "run_dir": run_dir,
            "manifest": manifest,
            "contract": contract,
            "selection_lock": selection_lock,
            "candidate_ids": candidate_ids,
            "expected_root_final_timestep": int(
                source["expected_root_final_timestep"]
            ),
        }
        assignments[source_id] = candidate_ids
    selected_sources = validate_source_assignment(assignments)

    entries = []
    for candidate_id in sorted(selected_sources):
        source_id = selected_sources[candidate_id]
        source = loaded_sources[source_id]
        entries.append(
            _load_candidate_entry(
                source_id=source_id,
                run_dir=source["run_dir"],
                manifest=source["manifest"],
                candidate_id=candidate_id,
            )
        )
    validation = validate_consolidated_records(
        entries,
        contracts={
            source_id: source["contract"]
            for source_id, source in loaded_sources.items()
        },
        expected_source_root_timesteps={
            source_id: source["expected_root_final_timestep"]
            for source_id, source in loaded_sources.items()
        },
    )

    revision = str(config["consolidation_revision"])
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else PROJECT / "reports/phase3b_stage_a" / revision
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite consolidation: {output_dir}")
    staging = output_dir.with_name(f".{output_dir.name}__tmp__pid{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"Stale consolidation staging directory: {staging}")
    staging.mkdir(parents=True)

    source_inventory = {}
    for source_id, source in loaded_sources.items():
        run_dir = source["run_dir"]
        source_inventory[source_id] = {
            "run_id": source["manifest"]["run_id"],
            "run_dir": run_dir.relative_to(PROJECT).as_posix(),
            "run_status": source["manifest"]["status"],
            "contract_sha256": source["manifest"]["contract_sha256"],
            "contract_file_sha256": _file_sha256(run_dir / "contract.json"),
            "source_config_sha256": source["contract"]["config_sha256"],
            "construction_parameters_embedded_in_contract": (
                "construction" in source["contract"]
            ),
            "manifest_file_sha256": _file_sha256(run_dir / "manifest.json"),
            "selection_lock_sha256": source["manifest"][
                "selection_lock_sha256"
            ],
            "source_implementation_sha256": source["contract"]["source_sha256"],
            "candidate_ids": source["candidate_ids"],
            "expected_root_final_timestep": source[
                "expected_root_final_timestep"
            ],
        }
    atomic_write_json(staging / "source_inventory.json", source_inventory)

    candidate_frame = pd.DataFrame(
        [
            _candidate_row(entry, source_inventory[entry["source_id"]])
            for entry in entries
        ]
    ).sort_values("candidate_id")
    candidate_frame.to_csv(staging / "candidate_inventory.csv", index=False)
    proposal_frame = pd.DataFrame(
        [row for entry in entries for row in _proposal_rows(entry)]
    ).sort_values(["candidate_id", "goal", "proposal_index"])
    proposal_frame.to_csv(staging / "proposal_coverage.csv", index=False)
    counterfactual_frame = pd.DataFrame(
        [row for entry in entries for row in _oracle_counterfactual_rows(entry)]
    )
    counterfactual_frame.to_csv(
        staging / "oracle_execution_counterfactuals.csv", index=False
    )

    pair_frame = pd.DataFrame(validation["pair_metrics"]).sort_values(
        "support_pair_id"
    )
    for column in pair_frame.columns:
        if len(pair_frame) and isinstance(pair_frame.iloc[0][column], list):
            pair_frame[column] = pair_frame[column].map(json.dumps)
    pair_frame.to_csv(staging / "support_pairs.csv", index=False)

    coverage_summary = []
    for (goal, mode), group in proposal_frame.groupby(
        ["goal", "execution_mode"], sort=True
    ):
        coverage_summary.append(
            {
                "goal": goal,
                "execution_mode": mode,
                "candidate_count": int(group["candidate_id"].nunique()),
                "attempt_count": int(len(group)),
                "success_count": int(group["pass"].sum()),
                "mean_candidate_success_fraction": float(
                    group.groupby("candidate_id")["pass"].mean().mean()
                ),
            }
        )
    compact_validation = {
        key: value for key, value in validation.items() if key != "pair_metrics"
    }
    summary = {
        "schema_version": 1,
        "consolidation_revision": revision,
        "status": "complete",
        "policy_loaded": False,
        "candidate_count": 32,
        "support_pair_count": 16,
        "source_count": len(loaded_sources),
        "validation": compact_validation,
        "coverage_summary_stratified_by_execution_mode": coverage_summary,
        "oracle_execution_counterfactual_count": int(len(counterfactual_frame)),
        "scientific_boundary": config["scientific_boundary"],
        "interpretation": (
            "The physical candidate lattice and all 16 support-pair geometry gates "
            "are complete as exact observed state records under matching shared root "
            "context. Historical contracts did not embed their construction YAML "
            "block; source config, revision, and implementation hashes remain batch "
            "provenance. Oracle balance is evaluated separately by goal only where "
            "proposal and execution contracts match. Legacy full replay, v34 "
            "world-anchor phases, and v35 bowl-registered phases are not pooled. "
            "Cross-mode effects and drawer-aperture effects aliased with source "
            "revision are intentionally not estimated."
        ),
    }
    atomic_write_json(staging / "summary.json", summary)
    (staging / "README.md").write_text(
        "# Phase 3b Stage A: consolidated physical lattice\n\n"
        "This compact report verifies and indexes the complete 32-state, 16-pair "
        "policy-independent physical lattice without rerunning previously completed "
        "candidates. The selected source runs share identical candidate IDs, "
        "demonstrations, environment context apart from horizon, support metric, "
        "certificate, goal, and validation payloads. Historical contracts did not "
        "embed the construction-parameter YAML block, and their config, revision, "
        "and source-file hashes differ. This gap remains explicit rather than being "
        "treated as implementation identity. Every persisted raw state payload and "
        "candidate record is hash-checked before this report is written.\n\n"
        "Legacy v31/v32 records receive an additive in-memory schema migration that "
        "labels their already-recorded proposal attempts as untransformed full "
        "trajectory replay and makes existing environment-step accounting explicit. "
        "Raw records are never edited, and their file hashes remain in the compact "
        "inventory.\n\n"
        "The exact observed lattice is a root bank for within-state Stage B language "
        "contrasts. Every physical support-pair geometry gate is checked, but oracle "
        "cost/overlap balance is reported per goal only where proposal and execution "
        "contracts match. Proposal-basin statistics are stratified across full "
        "replay, world-anchor phase, and bowl-registered phase modes. Closed versus "
        "open roots are aliased with source revision, so cross-aperture causal claims "
        "remain non-estimable. `oracle_execution_counterfactuals.csv` preserves the "
        "exact-root legacy-negative versus registered-positive bank intervention "
        "without replacing either ledger. A Stage B runner must revalidate every "
        "root under one common hydration/certificate path. No VLA was loaded, and "
        "this report is not evidence of a hidden-state mechanism.\n"
    )
    artifact_sha256 = {
        path.name: _file_sha256(path)
        for path in sorted(staging.iterdir())
        if path.is_file()
    }
    atomic_write_json(
        staging / "manifest.json",
        {
            "schema_version": 1,
            "consolidation_revision": revision,
            "status": "complete",
            "created_at": datetime.now(UTC).isoformat(),
            "config_path": config_path.relative_to(PROJECT).as_posix(),
            "config_sha256": _file_sha256(config_path),
            "consolidator_sha256": _file_sha256(Path(__file__).resolve()),
            "candidate_inventory_sha256": canonical_sha256(
                candidate_frame.to_dict("records")
            ),
            "proposal_coverage_sha256": canonical_sha256(
                proposal_frame.to_dict("records")
            ),
            "oracle_execution_counterfactuals_sha256": canonical_sha256(
                counterfactual_frame.to_dict("records")
            ),
            "artifact_sha256": artifact_sha256,
        },
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Stage A consolidation complete: {output_dir}")


if __name__ == "__main__":
    main()
