#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from smolvla_analysis.phase3_crd import atomic_write_json
from smolvla_analysis.phase3b_proposal_analysis import (
    factorial_scalar_decomposition,
    factorial_proposal_decomposition,
    support_set_transitions,
    validate_proposal_coverage,
)
from smolvla_analysis.phase3b_stage_a import (
    GOALS,
    canonical_sha256,
    candidate_spec,
    iter_candidate_specs,
    validate_selection_lock,
    validate_stage_a_records,
)


PROJECT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and decompose a complete Phase 3b Stage A compact report."
        )
    )
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validation_limits(contract: dict[str, Any]) -> dict[str, float]:
    values = contract["validation"]
    return {
        "max_oracle_cost_mismatch": float(
            values["oracle_cost_mismatch_limit"]
        ),
        "max_realized_goal_distance_mismatch": float(
            values["realized_goal_distance_mismatch_limit"]
        ),
        "max_planned_recovery_distance_mismatch": float(
            values["planned_recovery_distance_mismatch_limit"]
        ),
        "max_executed_step_mismatch": float(
            values["executed_step_mismatch_limit"]
        ),
        "max_active_step_mismatch": float(
            values["active_step_mismatch_limit"]
        ),
        "max_eef_path_mismatch": float(values["eef_path_mismatch_limit"]),
        "max_motion_control_effort_mismatch": float(
            values["motion_control_effort_mismatch_limit"]
        ),
    }


def _verify_coverage_against_records(
    coverage: pd.DataFrame, records: list[dict[str, Any]]
) -> None:
    observed = {}
    for row in coverage.to_dict("records"):
        key = (
            row["candidate_id"],
            row["goal"],
            int(row["proposal_index"]),
        )
        observed[key] = {
            "episode_index": int(row["episode_index"]),
            "action_sha256": row["action_sha256"],
            "pass": bool(row["pass"]),
            "selected": bool(row["selected"]),
        }
    expected = {}
    for record in records:
        for goal in GOALS:
            oracle = record["oracles"][goal]
            selected = int(oracle["selected_proposal_index"])
            for proposal, attempt in zip(
                oracle["proposal_bank"],
                oracle["proposal_attempts"],
                strict=True,
            ):
                index = int(proposal["proposal_index"])
                key = (record["candidate_id"], goal, index)
                expected[key] = {
                    "episode_index": int(proposal["episode_index"]),
                    "action_sha256": proposal["action_sha256"],
                    "pass": bool(attempt["pass"]),
                    "selected": index == selected,
                }
    if observed != expected:
        raise ValueError("Proposal coverage CSV differs from candidate ledgers")


def _verify_primary_report(
    report_dir: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    pd.DataFrame,
    dict[str, Any],
]:
    manifest = json.loads((report_dir / "manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("candidate_count") != 32:
        raise ValueError("Stage A compact report is not complete")
    expected_artifacts = manifest.get("artifact_sha256")
    if not isinstance(expected_artifacts, dict) or not expected_artifacts:
        raise ValueError("Stage A compact report has no artifact hash inventory")
    observed_artifacts = {
        path.name for path in report_dir.iterdir() if path.is_file()
    } - {"manifest.json"}
    if observed_artifacts != set(expected_artifacts):
        raise ValueError(
            "Stage A compact artifact inventory changed: "
            f"missing={sorted(set(expected_artifacts) - observed_artifacts)}, "
            f"extra={sorted(observed_artifacts - set(expected_artifacts))}"
        )
    for name, expected_hash in expected_artifacts.items():
        actual_hash = _file_sha256(report_dir / name)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Stage A compact artifact hash mismatch for {name}: "
                f"{actual_hash} != {expected_hash}"
            )

    contract = json.loads((report_dir / "contract.json").read_text())
    if canonical_sha256(contract) != manifest["contract_sha256"]:
        raise ValueError("Stage A compact contract hash mismatch")
    selection_lock = json.loads((report_dir / "selection_lock.json").read_text())
    validate_selection_lock(
        selection_lock,
        contract_sha256=manifest["contract_sha256"],
        construction_revision=contract["construction_revision"],
    )
    if (
        selection_lock["selection_lock_sha256"]
        != manifest["selection_lock_sha256"]
    ):
        raise ValueError("Stage A compact selection-lock hash mismatch")

    records = json.loads((report_dir / "candidates.json").read_text())
    ordered_records = sorted(records, key=lambda record: record["candidate_id"])
    if canonical_sha256(ordered_records) != manifest["candidate_records_sha256"]:
        raise ValueError("Stage A compact candidate-record hash mismatch")
    state_hash_inventory = [
        {
            "candidate_id": record["candidate_id"],
            "state_sha256": record["state_sha256"],
        }
        for record in ordered_records
    ]
    if (
        canonical_sha256(state_hash_inventory)
        != manifest["state_hash_inventory_sha256"]
    ):
        raise ValueError("Stage A compact state-hash inventory mismatch")
    validation = validate_stage_a_records(
        ordered_records, **_validation_limits(contract)
    )

    summary = json.loads((report_dir / "summary.json").read_text())
    for field in ("candidate_records_sha256", "state_hash_inventory_sha256"):
        if summary.get(field) != manifest[field]:
            raise ValueError(f"Stage A compact summary mismatch for {field}")
    coverage = validate_proposal_coverage(
        pd.read_csv(report_dir / "proposal_coverage.csv")
    )
    _verify_coverage_against_records(coverage, ordered_records)
    source_validation = {
        "pass": True,
        "artifact_count": len(expected_artifacts),
        "candidate_count": len(ordered_records),
        "coverage_row_count": int(len(coverage)),
        "contract_sha256": manifest["contract_sha256"],
        "selection_lock_sha256": manifest["selection_lock_sha256"],
        "candidate_records_sha256": manifest["candidate_records_sha256"],
        "state_hash_inventory_sha256": manifest[
            "state_hash_inventory_sha256"
        ],
        "aggregate_gate": validation,
    }
    return manifest, contract, ordered_records, coverage, source_validation


def _normalization_diagnostics(
    records: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows = []
    for record in records:
        spec = candidate_spec(record["candidate_id"])
        drawer = record["oracles"]["drawer"]
        cabinet = record["oracles"]["cabinet"]
        if drawer["phases"] != cabinet["phases"]:
            raise ValueError(
                f"Goal-specific normalization phases differ for {spec.candidate_id}"
            )
        if (
            drawer["shared_normalized_state_sha256"]
            != cabinet["shared_normalized_state_sha256"]
            or drawer["shared_normalization_action_sha256"]
            != cabinet["shared_normalization_action_sha256"]
        ):
            raise ValueError(
                f"Goal-specific normalized roots differ for {spec.candidate_id}"
            )
        home = drawer["phases"]["home"]
        rows.append(
            {
                **spec.as_dict(),
                "normalized_state_sha256": drawer[
                    "shared_normalized_state_sha256"
                ],
                "normalization_action_sha256": drawer[
                    "shared_normalization_action_sha256"
                ],
                "normalization_action_steps": int(
                    drawer["shared_normalization_action_steps"]
                ),
                "normalized_bowl_position_error_m": float(
                    drawer["normalized_bowl_position_error_m"]
                ),
                "home_final_position_error_m": float(
                    home["final_position_error_m"]
                ),
                "home_final_orientation_error_rad": float(
                    home["final_orientation_error_rad"]
                ),
                "home_joint_max_abs_error_rad": float(
                    home["joint_max_abs_error_from_layout_reset"]
                ),
                "home_active_action_steps": int(home["active_action_steps"]),
            }
        )
    candidates = pd.DataFrame(rows).sort_values("candidate_id")

    pair_rows = []
    for pair_id, pair in candidates.groupby("support_pair_id", sort=True):
        near = pair[pair["support_stratum"] == "demonstration_near"].iloc[0]
        low = pair[
            pair["support_stratum"] == "transverse_low_support"
        ].iloc[0]
        pair_rows.append(
            {
                "support_pair_id": pair_id,
                "drawer_aperture": near["drawer_aperture"],
                "possession": near["possession"],
                "transit_locus": near["transit_locus"],
                "layout": near["layout"],
                "normalized_state_hash_equal": (
                    near["normalized_state_sha256"]
                    == low["normalized_state_sha256"]
                ),
                "normalization_action_hash_equal": (
                    near["normalization_action_sha256"]
                    == low["normalization_action_sha256"]
                ),
                "normalization_action_step_difference": int(
                    low["normalization_action_steps"]
                    - near["normalization_action_steps"]
                ),
                "normalized_bowl_error_abs_difference_m": float(
                    abs(
                        low["normalized_bowl_position_error_m"]
                        - near["normalized_bowl_position_error_m"]
                    )
                ),
                "home_position_error_abs_difference_m": float(
                    abs(
                        low["home_final_position_error_m"]
                        - near["home_final_position_error_m"]
                    )
                ),
                "home_joint_error_abs_difference_rad": float(
                    abs(
                        low["home_joint_max_abs_error_rad"]
                        - near["home_joint_max_abs_error_rad"]
                    )
                ),
                "home_active_action_step_difference": int(
                    low["home_active_action_steps"]
                    - near["home_active_action_steps"]
                ),
            }
        )
    pairs = pd.DataFrame(pair_rows).sort_values("support_pair_id")
    summary = {
        "candidate_count": int(len(candidates)),
        "unique_normalized_full_state_count": int(
            candidates["normalized_state_sha256"].nunique()
        ),
        "support_pair_count": int(len(pairs)),
        "support_pairs_with_identical_normalized_full_state": int(
            pairs["normalized_state_hash_equal"].sum()
        ),
        "support_pairs_with_identical_normalization_actions": int(
            pairs["normalization_action_hash_equal"].sum()
        ),
        "max_normalized_bowl_position_error_m": float(
            candidates["normalized_bowl_position_error_m"].max()
        ),
        "max_home_position_error_m": float(
            candidates["home_final_position_error_m"].max()
        ),
        "max_home_joint_error_rad": float(
            candidates["home_joint_max_abs_error_rad"].max()
        ),
        "max_pair_normalized_bowl_error_abs_difference_m": float(
            pairs["normalized_bowl_error_abs_difference_m"].max()
        ),
        "max_pair_home_position_error_abs_difference_m": float(
            pairs["home_position_error_abs_difference_m"].max()
        ),
        "max_pair_home_joint_error_abs_difference_rad": float(
            pairs["home_joint_error_abs_difference_rad"].max()
        ),
        "estimand": "Y(N(s), g, k)",
        "interpretation": (
            "The demonstration proposal is replayed after deterministic recovery/"
            "homing N(s). Proposal-set differences therefore include sensitivity "
            "to residual normalized controller/simulator state and are not direct "
            "evidence of trajectory memory at the original candidate root."
        ),
    }
    return candidates, pairs, summary


def _correlation(
    first: pd.Series, second: pd.Series, *, method: str
) -> float | None:
    values = pd.DataFrame({"first": first, "second": second}).dropna()
    if (
        len(values) < 3
        or float(values["first"].std(ddof=0)) < 1e-12
        or float(values["second"].std(ddof=0)) < 1e-12
    ):
        return None
    return float(values["first"].corr(values["second"], method=method))


def _mean_or_none(values: pd.Series) -> float | None:
    return float(values.mean()) if len(values) else None


def _support_coverage_association(
    records: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    candidate_rows = []
    records_by_id = {record["candidate_id"]: record for record in records}
    for record in records:
        spec = candidate_spec(record["candidate_id"])
        support = record["support_measurement"]
        for goal in GOALS:
            candidate_rows.append(
                {
                    **spec.as_dict(),
                    "goal": goal,
                    "joint_support_distance": float(
                        support["nearest"]["distance"]
                    ),
                    "exact_event_reference_count": int(
                        support["event_matching_reference_count"]
                    ),
                    "proposal_success_count": int(
                        record["oracles"][goal]["proposal_success_count"]
                    ),
                    "proposal_attempt_count": int(
                        record["oracles"][goal]["proposal_attempt_count"]
                    ),
                    "proposal_success_fraction": float(
                        record["oracles"][goal]["proposal_success_fraction"]
                    ),
                }
            )
    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["goal", "candidate_id"]
    )

    pair_rows = []
    pair_ids = sorted(
        {spec.support_pair_id for spec in iter_candidate_specs()}
    )
    for pair_id in pair_ids:
        specs = [
            spec
            for spec in iter_candidate_specs()
            if spec.support_pair_id == pair_id
        ]
        near_spec = next(
            spec for spec in specs if spec.support_stratum == "demonstration_near"
        )
        low_spec = next(
            spec
            for spec in specs
            if spec.support_stratum == "transverse_low_support"
        )
        near = records_by_id[near_spec.candidate_id]
        low = records_by_id[low_spec.candidate_id]
        support_delta = float(
            low["support_measurement"]["nearest"]["distance"]
            - near["support_measurement"]["nearest"]["distance"]
        )
        for goal in GOALS:
            basin_delta = float(
                low["oracles"][goal]["proposal_success_fraction"]
                - near["oracles"][goal]["proposal_success_fraction"]
            )
            pair_rows.append(
                {
                    "support_pair_id": pair_id,
                    "drawer_aperture": near_spec.drawer_aperture,
                    "possession": near_spec.possession,
                    "transit_locus": near_spec.transit_locus,
                    "layout": near_spec.layout,
                    "goal": goal,
                    "joint_support_distance_difference": support_delta,
                    "proposal_success_fraction_difference": basin_delta,
                    "same_nonzero_direction": bool(
                        support_delta != 0.0
                        and basin_delta != 0.0
                        and np.sign(support_delta) == np.sign(basin_delta)
                    ),
                }
            )
    pairs = pd.DataFrame(pair_rows).sort_values(["goal", "support_pair_id"])

    summary = {}
    for goal in GOALS:
        goal_candidates = candidates[candidates["goal"] == goal]
        goal_pairs = pairs[pairs["goal"] == goal]
        supported = goal_candidates[
            goal_candidates["exact_event_reference_count"] > 0
        ]["proposal_success_fraction"]
        unsupported = goal_candidates[
            goal_candidates["exact_event_reference_count"] == 0
        ]["proposal_success_fraction"]
        summary[goal] = {
            "candidate_support_basin_pearson": _correlation(
                goal_candidates["joint_support_distance"],
                goal_candidates["proposal_success_fraction"],
                method="pearson",
            ),
            "candidate_support_basin_spearman": _correlation(
                goal_candidates["joint_support_distance"],
                goal_candidates["proposal_success_fraction"],
                method="spearman",
            ),
            "matched_delta_pearson": _correlation(
                goal_pairs["joint_support_distance_difference"],
                goal_pairs["proposal_success_fraction_difference"],
                method="pearson",
            ),
            "matched_delta_spearman": _correlation(
                goal_pairs["joint_support_distance_difference"],
                goal_pairs["proposal_success_fraction_difference"],
                method="spearman",
            ),
            "positive_measured_support_direction_pair_count": int(
                (goal_pairs["joint_support_distance_difference"] > 0.0).sum()
            ),
            "same_nonzero_support_and_basin_direction_pair_count": int(
                goal_pairs["same_nonzero_direction"].sum()
            ),
            "exact_event_supported_candidate_count": int(len(supported)),
            "mean_basin_width_exact_event_supported": _mean_or_none(supported),
            "mean_basin_width_exact_event_unsupported": _mean_or_none(
                unsupported
            ),
        }
    return candidates, pairs, summary


def main() -> None:
    args = _parse_args()
    report_dir = args.report_dir.resolve()
    if not report_dir.is_dir():
        raise FileNotFoundError(f"Missing Stage A compact report: {report_dir}")
    manifest, contract, records, coverage, source_validation = (
        _verify_primary_report(report_dir)
    )
    components, decomposition_summary = factorial_proposal_decomposition(
        coverage
    )
    transitions, transition_summary = support_set_transitions(coverage)
    normalization_candidates, normalization_pairs, normalization_summary = (
        _normalization_diagnostics(records)
    )
    support_candidates, support_pairs, support_summary = (
        _support_coverage_association(records)
    )
    support_distance_candidates = support_candidates.drop_duplicates(
        "candidate_id"
    )[["candidate_id", "joint_support_distance"]]
    support_factorial, support_factorial_summary = (
        factorial_scalar_decomposition(
            support_distance_candidates,
            value_column="joint_support_distance",
        )
    )

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else PROJECT / "reports/phase3b_stage_a_analysis" / report_dir.name
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Stage A analysis: {output_dir}")
    staging = output_dir.with_name(f".{output_dir.name}__tmp__pid{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"Stale Stage A analysis staging path: {staging}")
    staging.mkdir(parents=True)

    summary = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "construction_revision": contract["construction_revision"],
        "source_validation": source_validation,
        "proposal_surface_decomposition_by_goal": decomposition_summary,
        "support_set_transition_summary_by_goal": transition_summary,
        "support_coverage_association_by_goal": support_summary,
        "support_distance_factorial_decomposition": (
            support_factorial_summary
        ),
        "normalization_diagnostics": normalization_summary,
        "scientific_boundary": (
            "Exact descriptive decomposition of deterministic human-demonstration "
            "replays after policy-independent normalization; no VLA was evaluated, "
            "and the components are not evidence of a hidden-state mechanism."
        ),
    }
    atomic_write_json(staging / "summary.json", summary)
    components.to_csv(staging / "factorial_proposal_decomposition.csv", index=False)
    transition_csv = transitions.copy()
    for column in ("gained_episode_indices", "lost_episode_indices"):
        transition_csv[column] = transition_csv[column].map(json.dumps)
    transition_csv.to_csv(staging / "support_set_transitions.csv", index=False)
    normalization_candidates.to_csv(
        staging / "normalization_candidates.csv", index=False
    )
    normalization_pairs.to_csv(staging / "normalization_pairs.csv", index=False)
    support_candidates.to_csv(
        staging / "support_coverage_candidates.csv", index=False
    )
    support_pairs.to_csv(staging / "support_coverage_pairs.csv", index=False)
    support_factorial.to_csv(
        staging / "support_distance_factorial_decomposition.csv", index=False
    )
    (staging / "README.md").write_text(
        "# Phase 3b Stage A proposal-surface analysis\n\n"
        "This report verifies every hash in the primary compact report, then "
        "decomposes the complete deterministic proposal-outcome matrix. For each "
        "goal, total binary-outcome variance is split exactly into proposal-wide "
        "generality, state effects shared across proposals, and state-by-proposal "
        "compatibility. The 31 orthogonal factorial terms further localize common "
        "and proposal-specific factor interactions.\n\n"
        "Matched and marginal associations between joint support distance and "
        "proposal-basin width are reported separately, so a geometric support "
        "label cannot substitute for measured occupancy.\n\n"
        "The proposal table estimates `Y(N(s), g, k)`, not direct policy behaviour "
        "from candidate root `s`: the oracle first applies deterministic recovery/"
        "homing `N`. Distinct normalized full-state hashes show that residual "
        "controller and solver state can mediate proposal-set changes. These "
        "results diagnose open-loop template brittleness and experimental nuisance "
        "structure; they do not establish VLA trajectory memorisation.\n"
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
            "run_id": manifest["run_id"],
            "status": "complete",
            "source_report": report_dir.relative_to(PROJECT).as_posix(),
            "source_report_manifest_sha256": _file_sha256(
                report_dir / "manifest.json"
            ),
            "source_contract_sha256": manifest["contract_sha256"],
            "analysis_source_sha256": _file_sha256(Path(__file__).resolve()),
            "artifact_sha256": artifact_sha256,
        },
    )
    os.replace(staging, output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Stage A proposal analysis complete: {output_dir}")


if __name__ == "__main__":
    main()
