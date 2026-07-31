from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .phase3b_stage_a import FACTOR_LEVELS, GOALS, candidate_spec, iter_candidate_specs


REQUIRED_COVERAGE_COLUMNS = {
    "candidate_id",
    "goal",
    "proposal_index",
    "episode_index",
    "proposal_execution_mode",
    "pass",
    *FACTOR_LEVELS,
}


def validate_proposal_coverage(coverage: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COVERAGE_COLUMNS - set(coverage.columns)
    if missing:
        raise ValueError(f"Proposal coverage is missing columns: {sorted(missing)}")
    frame = coverage.copy()
    if frame.empty:
        raise ValueError("Proposal coverage is empty")
    boolean_values = set(frame["pass"].dropna().unique().tolist())
    if not boolean_values <= {True, False, 0, 1}:
        raise ValueError(f"Proposal outcomes are not boolean: {boolean_values}")
    frame["pass"] = frame["pass"].astype(bool)
    frame["proposal_index"] = frame["proposal_index"].astype(int)
    frame["episode_index"] = frame["episode_index"].astype(int)

    expected_candidates = {
        spec.candidate_id: spec for spec in iter_candidate_specs()
    }
    if set(frame["candidate_id"]) != set(expected_candidates):
        raise ValueError("Proposal coverage does not contain the exact Stage A lattice")
    if set(frame["goal"]) != set(GOALS):
        raise ValueError("Proposal coverage does not contain both locked goals")
    for candidate_id, group in frame.groupby("candidate_id", sort=False):
        spec = expected_candidates[candidate_id]
        for factor in FACTOR_LEVELS:
            if set(group[factor]) != {getattr(spec, factor)}:
                raise ValueError(
                    f"Coverage factor mismatch for {candidate_id}/{factor}"
                )

    if frame.duplicated(["candidate_id", "goal", "proposal_index"]).any():
        raise ValueError("Proposal coverage contains duplicate outcome cells")
    for goal, goal_frame in frame.groupby("goal", sort=False):
        execution_modes = set(goal_frame["proposal_execution_mode"])
        if len(execution_modes) != 1:
            raise ValueError(
                f"{goal} proposal coverage mixes execution modes: "
                f"{sorted(execution_modes)}"
            )
        proposal_ids = sorted(goal_frame["proposal_index"].unique())
        if proposal_ids != list(range(len(proposal_ids))):
            raise ValueError(f"{goal} proposal indices are not contiguous from zero")
        counts = goal_frame.groupby("candidate_id")["proposal_index"].nunique()
        if set(counts.index) != set(expected_candidates) or set(counts.values) != {
            len(proposal_ids)
        }:
            raise ValueError(f"{goal} proposal coverage is not rectangular")
        identities = goal_frame.groupby("proposal_index")["episode_index"].nunique()
        if set(identities.values) != {1}:
            raise ValueError(f"{goal} proposal identities vary across candidates")
    return frame


def _factorial_terms() -> tuple[tuple[str, ...], ...]:
    factors = tuple(FACTOR_LEVELS)
    return tuple(
        term
        for order in range(1, len(factors) + 1)
        for term in combinations(factors, order)
    )


def _factorial_design() -> tuple[
    tuple[Any, ...], tuple[tuple[str, ...], ...], np.ndarray
]:
    specs = iter_candidate_specs()
    terms = _factorial_terms()
    design = np.column_stack(
        [
            np.prod(
                [
                    np.asarray(
                        [
                            -1.0
                            if getattr(spec, factor) == FACTOR_LEVELS[factor][0]
                            else 1.0
                            for spec in specs
                        ]
                    )
                    for factor in term
                ],
                axis=0,
            )
            for term in terms
        ]
    )
    gram = design.T @ design / len(specs)
    if not np.allclose(gram, np.eye(len(terms)), rtol=0.0, atol=1e-12):
        raise RuntimeError("Stage A factorial basis is not orthonormal")
    return specs, terms, design


def factorial_scalar_decomposition(
    frame: pd.DataFrame, *, value_column: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "candidate_id" not in frame or value_column not in frame:
        raise ValueError(
            f"Scalar decomposition requires candidate_id and {value_column}"
        )
    if frame.duplicated("candidate_id").any():
        raise ValueError("Scalar decomposition requires one row per candidate")
    specs, terms, design = _factorial_design()
    expected_ids = [spec.candidate_id for spec in specs]
    indexed = frame.set_index("candidate_id")
    if set(indexed.index) != set(expected_ids):
        raise ValueError("Scalar decomposition does not cover the exact Stage A lattice")
    values = indexed.reindex(expected_ids)[value_column].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Scalar decomposition field {value_column} is non-finite")
    mean = float(values.mean())
    coefficients = design.T @ values / len(values)
    component_variances = np.square(coefficients)
    total_variance = float(np.mean(np.square(values - mean)))
    reconstructed_variance = float(component_variances.sum())
    if not np.isclose(
        reconstructed_variance, total_variance, rtol=0.0, atol=1e-12
    ):
        raise RuntimeError(
            f"{value_column} factorial decomposition does not reconstruct variance"
        )
    rows = []
    for index, term in enumerate(terms):
        rows.append(
            {
                "outcome": value_column,
                "term": "*".join(term),
                "order": len(term),
                "coefficient": float(coefficients[index]),
                "factorial_contrast": float(
                    (2 ** len(term)) * coefficients[index]
                ),
                "component_variance": float(component_variances[index]),
                "variance_fraction": (
                    float(component_variances[index] / total_variance)
                    if total_variance > 0.0
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows), {
        "outcome": value_column,
        "candidate_count": len(values),
        "mean": mean,
        "total_variance": total_variance,
        "variance_reconstruction_error": float(
            reconstructed_variance - total_variance
        ),
    }


def factorial_proposal_decomposition(
    coverage: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Exactly decompose deterministic proposal outcomes on the 2^5 lattice.

    For each goal, the binary matrix Y(state, proposal) is expanded in the
    orthogonal +/-1 factorial basis over states. Proposal-mean variation is the
    proposal-generality component. The mean coefficient for each state term is
    common difficulty; coefficient variation across proposals is state-by-
    proposal compatibility. These components reconstruct total variance exactly.
    """

    frame = validate_proposal_coverage(coverage)
    specs, terms, design = _factorial_design()
    candidate_ids = [spec.candidate_id for spec in specs]

    rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for goal in GOALS:
        goal_frame = frame[frame["goal"] == goal]
        matrix = (
            goal_frame.pivot(
                index="candidate_id", columns="proposal_index", values="pass"
            )
            .reindex(candidate_ids)
            .sort_index(axis=1)
            .to_numpy(dtype=np.float64)
        )
        if matrix.shape[0] != 32 or np.isnan(matrix).any():
            raise ValueError(f"{goal} proposal matrix is incomplete")
        proposal_means = matrix.mean(axis=0)
        grand_mean = float(matrix.mean())
        coefficients = design.T @ matrix / len(specs)
        mean_coefficients = coefficients.mean(axis=1)
        common_variance = np.square(mean_coefficients)
        interaction_variance = np.mean(
            np.square(coefficients - mean_coefficients[:, None]), axis=1
        )
        proposal_variance = float(np.mean(np.square(proposal_means - grand_mean)))
        total_variance = float(np.mean(np.square(matrix - grand_mean)))
        common_state_variance = float(common_variance.sum())
        state_proposal_variance = float(interaction_variance.sum())
        reconstructed_variance = (
            proposal_variance
            + common_state_variance
            + state_proposal_variance
        )
        if not np.isclose(
            reconstructed_variance, total_variance, rtol=0.0, atol=1e-12
        ):
            raise RuntimeError(
                f"{goal} factorial decomposition does not reconstruct variance: "
                f"{reconstructed_variance} != {total_variance}"
            )

        for index, term in enumerate(terms):
            scale = float(2 ** len(term))
            rows.append(
                {
                    "goal": goal,
                    "term": "*".join(term),
                    "order": len(term),
                    "common_coefficient": float(mean_coefficients[index]),
                    "common_factorial_contrast": float(
                        scale * mean_coefficients[index]
                    ),
                    "proposal_specific_contrast_sd": float(
                        scale * np.std(coefficients[index])
                    ),
                    "common_state_variance": float(common_variance[index]),
                    "state_proposal_interaction_variance": float(
                        interaction_variance[index]
                    ),
                    "common_state_variance_fraction": (
                        float(common_variance[index] / common_state_variance)
                        if common_state_variance > 0.0
                        else 0.0
                    ),
                    "state_proposal_interaction_variance_fraction": (
                        float(
                            interaction_variance[index]
                            / state_proposal_variance
                        )
                        if state_proposal_variance > 0.0
                        else 0.0
                    ),
                    "total_outcome_variance_fraction_common": (
                        float(common_variance[index] / total_variance)
                        if total_variance > 0.0
                        else 0.0
                    ),
                    "total_outcome_variance_fraction_interaction": (
                        float(interaction_variance[index] / total_variance)
                        if total_variance > 0.0
                        else 0.0
                    ),
                }
            )

        successful_cells_by_proposal = matrix.sum(axis=0)
        total_success_cells = float(successful_cells_by_proposal.sum())
        effective_proposal_count = (
            float(
                total_success_cells**2
                / np.square(successful_cells_by_proposal).sum()
            )
            if total_success_cells > 0.0
            else 0.0
        )
        sorted_successes = np.sort(successful_cells_by_proposal)[::-1]
        summaries[goal] = {
            "candidate_count": int(matrix.shape[0]),
            "proposal_count": int(matrix.shape[1]),
            "outcome_cell_count": int(matrix.size),
            "success_cell_count": int(matrix.sum()),
            "grand_success_fraction": grand_mean,
            "total_outcome_variance": total_variance,
            "proposal_generality_variance": proposal_variance,
            "common_state_variance": common_state_variance,
            "state_proposal_interaction_variance": state_proposal_variance,
            "proposal_generality_variance_fraction": (
                proposal_variance / total_variance if total_variance > 0.0 else 0.0
            ),
            "common_state_variance_fraction": (
                common_state_variance / total_variance
                if total_variance > 0.0
                else 0.0
            ),
            "state_proposal_interaction_variance_fraction": (
                state_proposal_variance / total_variance
                if total_variance > 0.0
                else 0.0
            ),
            "variance_reconstruction_error": float(
                reconstructed_variance - total_variance
            ),
            "never_successful_proposal_count": int(
                np.sum(successful_cells_by_proposal == 0.0)
            ),
            "always_successful_proposal_count": int(
                np.sum(successful_cells_by_proposal == matrix.shape[0])
            ),
            "effective_success_carrying_proposal_count": effective_proposal_count,
            "top_1_proposal_success_share": (
                float(sorted_successes[:1].sum() / total_success_cells)
                if total_success_cells > 0.0
                else 0.0
            ),
            "top_5_proposal_success_share": (
                float(sorted_successes[:5].sum() / total_success_cells)
                if total_success_cells > 0.0
                else 0.0
            ),
        }
    return pd.DataFrame(rows), summaries


def support_set_transitions(
    coverage: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    frame = validate_proposal_coverage(coverage)
    candidate_ids = [spec.candidate_id for spec in iter_candidate_specs()]
    rows: list[dict[str, Any]] = []
    for goal in GOALS:
        goal_frame = frame[frame["goal"] == goal]
        matrix = (
            goal_frame.pivot(
                index="candidate_id", columns="proposal_index", values="pass"
            )
            .reindex(candidate_ids)
            .sort_index(axis=1)
        )
        episode_by_proposal = (
            goal_frame[["proposal_index", "episode_index"]]
            .drop_duplicates()
            .set_index("proposal_index")["episode_index"]
            .to_dict()
        )
        records = {candidate_id: matrix.loc[candidate_id] for candidate_id in matrix.index}
        for pair_id in sorted(
            {spec.support_pair_id for spec in iter_candidate_specs()}
        ):
            pair = [
                spec
                for spec in iter_candidate_specs()
                if spec.support_pair_id == pair_id
            ]
            near = next(
                spec for spec in pair if spec.support_stratum == "demonstration_near"
            )
            low = next(
                spec
                for spec in pair
                if spec.support_stratum == "transverse_low_support"
            )
            near_success = set(
                records[near.candidate_id][records[near.candidate_id]].index.astype(int)
            )
            low_success = set(
                records[low.candidate_id][records[low.candidate_id]].index.astype(int)
            )
            shared = near_success & low_success
            union = near_success | low_success
            gained = low_success - near_success
            lost = near_success - low_success
            rows.append(
                {
                    "goal": goal,
                    "support_pair_id": pair_id,
                    "near_success_count": len(near_success),
                    "low_success_count": len(low_success),
                    "shared_success_count": len(shared),
                    "gained_success_count": len(gained),
                    "lost_success_count": len(lost),
                    "symmetric_difference_count": len(gained | lost),
                    "success_set_jaccard": len(shared) / len(union),
                    "identical_success_set": near_success == low_success,
                    "disjoint_success_set": not shared,
                    "gained_episode_indices": sorted(
                        episode_by_proposal[index] for index in gained
                    ),
                    "lost_episode_indices": sorted(
                        episode_by_proposal[index] for index in lost
                    ),
                }
            )
    transitions = pd.DataFrame(rows).sort_values(["goal", "support_pair_id"])
    summaries = {}
    for goal in GOALS:
        group = transitions[transitions["goal"] == goal]
        summaries[goal] = {
            "support_pair_count": int(len(group)),
            "disjoint_success_set_count": int(group["disjoint_success_set"].sum()),
            "identical_success_set_count": int(group["identical_success_set"].sum()),
            "median_success_set_jaccard": float(group["success_set_jaccard"].median()),
            "mean_success_set_jaccard": float(group["success_set_jaccard"].mean()),
            "mean_symmetric_difference_count": float(
                group["symmetric_difference_count"].mean()
            ),
        }
    return transitions, summaries
