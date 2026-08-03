from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

import numpy as np

from .phase3b_completion import oracle_pair_comparability
from .phase3b_registered_validation import (
    validate_oracle_proposal_ledger_compatible,
    validate_support_pair_records_compatible,
)
from .phase3b_stage_a import (
    FACTOR_LEVELS,
    GOALS,
    candidate_spec,
    canonical_sha256,
    iter_candidate_specs,
    symmetric_relative_difference,
    validate_support_pair_geometry_records,
)


LEGACY_EXECUTION_MODE = "full_trajectory_replay"
LEGACY_EXECUTION_CONTRACT = {
    "execution_mode": LEGACY_EXECUTION_MODE,
    "transformation": "none",
}
CONSOLIDATION_MIGRATION = "additive_full_replay_provenance_v1"


def construction_contract_payload(contract: dict[str, Any]) -> dict[str, Any]:
    """Return shared declarative root context, excluding oracle horizon.

    Historical contracts did not embed the construction-parameter block from the
    YAML config. This payload therefore proves equality only for the fields that
    were actually bound into every contract. Revision, source-file, and config
    hashes remain separate provenance rather than implementation-identity claims.
    """

    environment = dict(contract["environment"])
    environment.pop("episode_length", None)
    payload = {
        "candidate_ids": contract["candidate_ids"],
        "certificate": contract["certificate"],
        "dataset_root": contract["dataset_root"],
        "demonstrations": contract["demonstrations"],
        "environment_without_horizon": environment,
        "goals": contract["goals"],
        "support_metric": contract["support_metric"],
        "validation": contract["validation"],
    }
    if "construction" in contract:
        payload["construction"] = contract["construction"]
    return payload


def validate_construction_contracts(
    contracts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not contracts:
        raise ValueError("Consolidation requires at least one source contract")
    payloads = {
        source_id: construction_contract_payload(contract)
        for source_id, contract in contracts.items()
    }
    payload_hashes = {
        source_id: canonical_sha256(payload)
        for source_id, payload in payloads.items()
    }
    if len(set(payload_hashes.values())) != 1:
        raise ValueError("Source runs use different shared root-context contracts")
    horizons = {
        source_id: int(contract["environment"]["episode_length"])
        for source_id, contract in contracts.items()
    }
    if any(horizon < 1 for horizon in horizons.values()):
        raise ValueError("A source contract has a nonpositive episode horizon")
    revisions = {
        source_id: str(contract["construction_revision"])
        for source_id, contract in contracts.items()
    }
    implementation_hashes = {
        source_id: contract["source_sha256"]
        for source_id, contract in contracts.items()
    }
    implementation_identity_hashes = {
        source_id: canonical_sha256(source_hashes)
        for source_id, source_hashes in implementation_hashes.items()
    }
    config_hashes = {
        source_id: str(contract["config_sha256"])
        for source_id, contract in contracts.items()
    }
    construction_parameters_bound = all(
        isinstance(contract.get("construction"), dict)
        for contract in contracts.values()
    )
    return {
        "pass": True,
        "shared_root_context_contract_match": True,
        "shared_root_context_contract_sha256": next(
            iter(payload_hashes.values())
        ),
        "source_shared_root_context_contract_sha256": payload_hashes,
        "construction_parameters_cryptographically_bound": (
            construction_parameters_bound
        ),
        "source_construction_revisions": revisions,
        "construction_revision_identical": len(set(revisions.values())) == 1,
        "source_implementation_sha256": implementation_hashes,
        "source_implementation_identical": len(
            set(implementation_identity_hashes.values())
        )
        == 1,
        "source_config_sha256": config_hashes,
        "config_bytes_available_in_historical_contract": False,
        "source_episode_horizons": horizons,
        "horizon_excluded_reason": (
            "The observed candidate records, rather than the historical contract, "
            "bind the common root timestep. Later oracle horizons are reported "
            "separately."
        ),
        "compatibility_boundary": (
            "Shared contract context matches, but historical contracts omitted the "
            "construction-parameter block. Differing config, revision, and source "
            "hashes remain explicit generation-batch provenance; cross-revision "
            "construction equivalence is not cryptographically established."
        ),
    }


def validate_source_assignment(
    assignments: dict[str, Iterable[str]],
) -> dict[str, str]:
    expected = {spec.candidate_id for spec in iter_candidate_specs()}
    selected: dict[str, str] = {}
    for source_id, candidate_ids in assignments.items():
        for candidate_id in candidate_ids:
            if candidate_id not in expected:
                raise ValueError(
                    f"Unknown consolidation candidate {candidate_id!r} in {source_id}"
                )
            if candidate_id in selected:
                raise ValueError(
                    f"Candidate {candidate_id} is assigned to both "
                    f"{selected[candidate_id]} and {source_id}"
                )
            selected[candidate_id] = source_id
    missing = expected - set(selected)
    if missing:
        raise ValueError(
            f"Consolidation assignment omits candidates: {sorted(missing)}"
        )
    return selected


def validate_source_root_timesteps(
    observed: dict[str, set[int]],
    *,
    contracts: dict[str, dict[str, Any]],
    expected: dict[str, int],
) -> dict[str, Any]:
    """Bind each source to one declared root timestep without conflating sources."""

    source_ids = set(contracts)
    if set(observed) != source_ids or set(expected) != source_ids:
        raise ValueError("Root-timestep provenance does not cover every source")
    normalized_expected = {
        source_id: int(expected[source_id]) for source_id in source_ids
    }
    for source_id in sorted(source_ids):
        timestep = normalized_expected[source_id]
        if timestep < 1 or observed[source_id] != {timestep}:
            raise ValueError(
                f"Source {source_id} does not use its declared root timestep"
            )
        if int(contracts[source_id]["environment"]["episode_length"]) <= timestep:
            raise ValueError(
                f"Source {source_id} oracle horizon ends before its roots"
            )
    unique = sorted(set(normalized_expected.values()))
    return {
        "source_expected_root_final_timesteps": dict(
            sorted(normalized_expected.items())
        ),
        "source_observed_root_final_timesteps": {
            source_id: sorted(observed[source_id])
            for source_id in sorted(source_ids)
        },
        "observed_root_final_timesteps": unique,
        "root_final_timestep_identical": len(unique) == 1,
        "observed_roots_precede_oracle_horizons": True,
    }


def factor_source_overlap(
    assignments: dict[str, Iterable[str]],
) -> dict[str, dict[str, Any]]:
    """Report whether each factor has a within-source level contrast."""

    selected = validate_source_assignment(assignments)
    report: dict[str, dict[str, Any]] = {}
    for factor, levels in FACTOR_LEVELS.items():
        sources_by_level = {
            level: sorted(
                {
                    source_id
                    for candidate_id, source_id in selected.items()
                    if getattr(candidate_spec(candidate_id), factor) == level
                }
            )
            for level in levels
        }
        common_sources = set(sources_by_level[levels[0]])
        for level in levels[1:]:
            common_sources &= set(sources_by_level[level])
        report[factor] = {
            "sources_by_level": sources_by_level,
            "common_sources_across_levels": sorted(common_sources),
            "source_blocked_contrast_available": bool(common_sources),
        }
    return report


def migrate_legacy_full_replay_record(
    record: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Add fields introduced in v33 without changing legacy evidence values."""

    migrated = deepcopy(record)
    migrated_goals: list[str] = []
    for goal in GOALS:
        oracle = migrated["oracles"][goal]
        fields = (
            oracle.get("proposal_execution_mode"),
            oracle.get("proposal_execution_contract"),
            oracle.get("proposal_execution_contract_sha256"),
            oracle.get("total_environment_action_steps"),
        )
        if all(value is None for value in fields):
            oracle["proposal_execution_mode"] = LEGACY_EXECUTION_MODE
            oracle["proposal_execution_contract"] = deepcopy(
                LEGACY_EXECUTION_CONTRACT
            )
            oracle["proposal_execution_contract_sha256"] = canonical_sha256(
                LEGACY_EXECUTION_CONTRACT
            )
            oracle["total_environment_action_steps"] = int(
                oracle["total_attempted_action_steps"]
            )
            for attempt in oracle["proposal_attempts"]:
                attempt["proposal_execution_mode"] = LEGACY_EXECUTION_MODE
            migrated_goals.append(goal)
        elif any(value is None for value in fields):
            raise ValueError(
                f"Partially migrated proposal provenance for "
                f"{record.get('candidate_id')}/{goal}"
            )
    return migrated, migrated_goals


def _validate_candidate_record(record: dict[str, Any]) -> dict[str, Any]:
    candidate_id = str(record.get("candidate_id"))
    spec = candidate_spec(candidate_id)
    if record.get("factors") != spec.as_dict():
        raise ValueError(f"Factor provenance mismatch for {candidate_id}")
    if record.get("policy_loaded") is not False:
        raise ValueError(f"Policy boundary violated for {candidate_id}")
    if record.get("root_validation", {}).get("pass") is not True:
        raise ValueError(f"Root validation failed for {candidate_id}")
    if record["root_validation"].get("goals") != {
        "drawer": False,
        "cabinet": False,
    }:
        raise ValueError(f"Candidate root already satisfies a goal: {candidate_id}")
    if record.get("certificate", {}).get("pass") is not True:
        raise ValueError(f"Computational-state certificate failed for {candidate_id}")
    if len(str(record.get("state_sha256", ""))) != 64:
        raise ValueError(f"Invalid state identity for {candidate_id}")
    construction = record.get("construction", {})
    root_final_timestep = int(construction.get("final_timestep", -1))
    if (
        construction.get("mode")
        != "current_process_policy_independent_script"
        or root_final_timestep < 1
        or len(str(construction.get("action_sha256", ""))) != 64
    ):
        raise ValueError(f"Invalid root construction record for {candidate_id}")
    modes = {}
    normalized_hashes = set()
    normalization_action_hashes = set()
    for goal in GOALS:
        oracle = record.get("oracles", {}).get(goal, {})
        if (
            oracle.get("pass") is not True
            or oracle.get("goal_ever_achieved") is not True
        ):
            raise ValueError(f"{goal} feasibility failed for {candidate_id}")
        ledger = validate_oracle_proposal_ledger_compatible(
            oracle, candidate_id=candidate_id, goal=goal
        )
        modes[goal] = ledger["execution_mode"]
        normalized_hashes.add(oracle["shared_normalized_state_sha256"])
        normalization_action_hashes.add(
            oracle["shared_normalization_action_sha256"]
        )
    if len(normalized_hashes) != 1 or len(normalization_action_hashes) != 1:
        raise ValueError(f"Goal-specific normalization differs for {candidate_id}")
    support = record.get("support_measurement", {})
    if (
        support.get("pass") is not True
        or len(str(support.get("reference_bank_sha256", ""))) != 64
        or int(support.get("reference_count", 0)) < 1
        or not np.isfinite(float(support.get("nearest", {}).get("distance", np.nan)))
    ):
        raise ValueError(f"Support measurement failed for {candidate_id}")
    category_matches = support.get("factor_category_matches", {})
    if (
        category_matches.get("drawer_aperture") is not True
        or category_matches.get("possession") is not True
    ):
        raise ValueError(f"Physical factor realization mismatch for {candidate_id}")
    return {
        "candidate_id": candidate_id,
        "execution_mode_by_goal": modes,
        "normalized_state_sha256": next(iter(normalized_hashes)),
        "normalization_action_sha256": next(iter(normalization_action_hashes)),
        "root_final_timestep": root_final_timestep,
    }


def _goal_pair_metrics(
    near: dict[str, Any],
    low: dict[str, Any],
    *,
    goal: str,
    comparable: dict[str, Any],
    limits: dict[str, Any],
) -> dict[str, Any]:
    prefix = f"{goal}_"
    base = {
        f"{prefix}oracle_balance_estimable": bool(comparable["estimable"]),
        f"{prefix}same_proposal_bank": bool(comparable["same_proposal_bank"]),
        f"{prefix}same_execution_contract": bool(
            comparable["same_execution_contract"]
        ),
        f"{prefix}near_execution_mode": comparable["near_execution_mode"],
        f"{prefix}low_execution_mode": comparable["low_execution_mode"],
    }
    if not comparable["estimable"]:
        return {
            **base,
            f"{prefix}selected_proposal_match": None,
            f"{prefix}shared_success_count": None,
            f"{prefix}success_set_jaccard": None,
            f"{prefix}matched_cost_proposal_index": None,
            f"{prefix}budgeted_cost_mismatch": None,
            f"{prefix}executed_step_mismatch": None,
            f"{prefix}active_step_mismatch": None,
            f"{prefix}eef_path_mismatch": None,
            f"{prefix}motion_effort_mismatch": None,
        }
    near_ledger = validate_oracle_proposal_ledger_compatible(
        near["oracles"][goal], candidate_id=near["candidate_id"], goal=goal
    )
    low_ledger = validate_oracle_proposal_ledger_compatible(
        low["oracles"][goal], candidate_id=low["candidate_id"], goal=goal
    )
    near_success = set(near_ledger["successful_indices"])
    low_success = set(low_ledger["successful_indices"])
    shared = sorted(near_success & low_success)
    union = near_success | low_success
    common_index = (
        min(
            shared,
            key=lambda index: (
                max(
                    int(
                        near_ledger["attempts"][index]["cost"][
                            "executed_action_steps"
                        ]
                    ),
                    int(
                        low_ledger["attempts"][index]["cost"][
                            "executed_action_steps"
                        ]
                    ),
                ),
                sum(
                    float(
                        ledger["attempts"][index]["cost"]["eef_path_length_m"]
                    )
                    for ledger in (near_ledger, low_ledger)
                ),
                index,
            ),
        )
        if shared
        else None
    )
    near_cost = near["oracles"][goal]["cost"]
    low_cost = low["oracles"][goal]["cost"]
    mismatch_specs = (
        (
            "budgeted_cost",
            "budgeted_action_steps",
            float(limits["oracle_cost_mismatch_limit"]),
        ),
        (
            "executed_step",
            "executed_action_steps",
            float(limits["executed_step_mismatch_limit"]),
        ),
        (
            "active_step",
            "active_servo_steps",
            float(limits["active_step_mismatch_limit"]),
        ),
        (
            "eef_path",
            "eef_path_length_m",
            float(limits["eef_path_mismatch_limit"]),
        ),
        (
            "motion_effort",
            "motion_control_effort",
            float(limits["motion_control_effort_mismatch_limit"]),
        ),
    )
    mismatches = {}
    for label, field, limit in mismatch_specs:
        mismatch = symmetric_relative_difference(
            near_cost[field], low_cost[field]
        )
        if mismatch > limit:
            raise ValueError(
                f"Support pair {near['factors']['support_pair_id']} exceeds the "
                f"{goal} {field} limit"
            )
        mismatches[f"{prefix}{label}_mismatch"] = mismatch
    return {
        **base,
        f"{prefix}selected_proposal_match": bool(
            near_ledger["selected_index"] == low_ledger["selected_index"]
        ),
        f"{prefix}shared_success_count": len(shared),
        f"{prefix}success_set_jaccard": len(shared) / len(union),
        f"{prefix}matched_cost_proposal_index": common_index,
        **mismatches,
    }


def validate_consolidated_records(
    entries: Iterable[dict[str, Any]],
    *,
    contracts: dict[str, dict[str, Any]],
    expected_source_root_timesteps: dict[str, int],
) -> dict[str, Any]:
    entries = tuple(entries)
    expected = {spec.candidate_id for spec in iter_candidate_specs()}
    by_id = {entry["record"]["candidate_id"]: entry for entry in entries}
    if len(entries) != 32 or set(by_id) != expected:
        raise ValueError(
            "Consolidated records do not cover the locked 32-state lattice"
        )
    entry_sources = {entry["source_id"] for entry in entries}
    if entry_sources != set(contracts):
        raise ValueError(
            "Consolidated record sources do not match the source contracts"
        )
    construction = validate_construction_contracts(contracts)
    assignments: dict[str, list[str]] = {}
    for entry in entries:
        assignments.setdefault(entry["source_id"], []).append(
            entry["record"]["candidate_id"]
        )
    source_overlap = factor_source_overlap(assignments)
    candidate_summaries = {
        candidate_id: _validate_candidate_record(entry["record"])
        for candidate_id, entry in by_id.items()
    }
    state_hashes = {
        entry["record"]["state_sha256"] for entry in entries
    }
    if len(state_hashes) != len(entries):
        raise ValueError("Consolidated lattice contains duplicate physical states")
    source_root_timesteps: dict[str, set[int]] = {}
    for candidate_id, summary in candidate_summaries.items():
        source_id = by_id[candidate_id]["source_id"]
        source_root_timesteps.setdefault(source_id, set()).add(
            summary["root_final_timestep"]
        )
    construction.update(
        validate_source_root_timesteps(
            source_root_timesteps,
            contracts=contracts,
            expected=expected_source_root_timesteps,
        )
    )
    support_bank_hashes = {
        entry["record"]["support_measurement"]["reference_bank_sha256"]
        for entry in entries
    }
    if len(support_bank_hashes) != 1:
        raise ValueError("Consolidated records use different support reference banks")

    limits = next(iter(contracts.values()))["validation"]
    pair_rows = []
    for pair_id in sorted({spec.support_pair_id for spec in iter_candidate_specs()}):
        pair_specs = [
            spec for spec in iter_candidate_specs() if spec.support_pair_id == pair_id
        ]
        near_spec = next(
            spec
            for spec in pair_specs
            if spec.support_stratum == "demonstration_near"
        )
        low_spec = next(
            spec
            for spec in pair_specs
            if spec.support_stratum == "transverse_low_support"
        )
        near_entry = by_id[near_spec.candidate_id]
        low_entry = by_id[low_spec.candidate_id]
        geometry = validate_support_pair_geometry_records(
            near_entry["record"],
            low_entry["record"],
            max_realized_goal_distance_mismatch=float(
                limits["realized_goal_distance_mismatch_limit"]
            ),
            max_planned_recovery_distance_mismatch=float(
                limits["planned_recovery_distance_mismatch_limit"]
            ),
        )
        comparability = oracle_pair_comparability(
            near_entry["record"], low_entry["record"]
        )
        goal_metrics = {
            key: value
            for goal in GOALS
            for key, value in _goal_pair_metrics(
                near_entry["record"],
                low_entry["record"],
                goal=goal,
                comparable=comparability["by_goal"][goal],
                limits=limits,
            ).items()
        }
        strict_pass = False
        if comparability["all_goals_estimable"]:
            validate_support_pair_records_compatible(
                near_entry["record"],
                low_entry["record"],
                max_oracle_cost_mismatch=float(
                    limits["oracle_cost_mismatch_limit"]
                ),
                max_realized_goal_distance_mismatch=float(
                    limits["realized_goal_distance_mismatch_limit"]
                ),
                max_planned_recovery_distance_mismatch=float(
                    limits["planned_recovery_distance_mismatch_limit"]
                ),
                max_executed_step_mismatch=float(
                    limits["executed_step_mismatch_limit"]
                ),
                max_active_step_mismatch=float(
                    limits["active_step_mismatch_limit"]
                ),
                max_eef_path_mismatch=float(
                    limits["eef_path_mismatch_limit"]
                ),
                max_motion_control_effort_mismatch=float(
                    limits["motion_control_effort_mismatch_limit"]
                ),
            )
            strict_pass = True
        pair_rows.append(
            {
                "near_source_id": near_entry["source_id"],
                "low_source_id": low_entry["source_id"],
                "same_source": bool(
                    near_entry["source_id"] == low_entry["source_id"]
                ),
                "same_root_final_timestep": bool(
                    near_entry["record"]["construction"]["final_timestep"]
                    == low_entry["record"]["construction"]["final_timestep"]
                ),
                "oracle_balance_all_goals_estimable": comparability[
                    "all_goals_estimable"
                ],
                "strict_oracle_balance_pass": strict_pass,
                **geometry,
                **goal_metrics,
            }
        )

    execution_modes = {
        goal: sorted(
            {
                summary["execution_mode_by_goal"][goal]
                for summary in candidate_summaries.values()
            }
        )
        for goal in GOALS
    }
    return {
        "pass": True,
        "candidate_count": len(entries),
        "support_pair_count": len(pair_rows),
        "physical_geometry_pair_count": len(pair_rows),
        "oracle_balance_all_goals_estimable_pair_count": sum(
            row["oracle_balance_all_goals_estimable"] for row in pair_rows
        ),
        "oracle_balance_estimable_pair_count_by_goal": {
            goal: sum(row[f"{goal}_oracle_balance_estimable"] for row in pair_rows)
            for goal in GOALS
        },
        "source_count": len({entry["source_id"] for entry in entries}),
        "support_reference_bank_sha256": next(iter(support_bank_hashes)),
        "execution_modes_by_goal": execution_modes,
        "cross_mode_factor_effects_estimable": all(
            len(modes) == 1 for modes in execution_modes.values()
        ),
        "construction_compatibility": construction,
        "factor_source_overlap": source_overlap,
        "source_blocked_factor_contrast_available": {
            factor: details["source_blocked_contrast_available"]
            for factor, details in source_overlap.items()
        },
        "pair_metrics": pair_rows,
    }
