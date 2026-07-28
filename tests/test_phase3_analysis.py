import numpy as np
import pytest

from smolvla_analysis.phase3_analysis import (
    beta_posterior_mean,
    empirical_beta_prior,
    grouped_bootstrap_mean_interval,
    grouped_ridge_oof,
    hierarchical_variance_components,
    wilson_interval,
)


def test_wilson_and_empirical_beta_shrink_extreme_small_cells():
    low, high = wilson_interval(0, 2)
    assert low == pytest.approx(0.0)
    assert 0.0 < high < 1.0
    successes = np.asarray([0, 1, 2, 2], dtype=float)
    trials = np.full(4, 2.0)
    prior = empirical_beta_prior(successes, trials)
    posterior = beta_posterior_mean(successes, trials, prior)
    assert 0.0 < posterior[0] < 0.5
    assert 0.5 < posterior[-1] < 1.0


def test_grouped_ridge_has_complete_held_group_coverage():
    groups = np.repeat(np.arange(4), 3)
    x = np.stack([groups, np.tile(np.arange(3), 4)], axis=1).astype(float)
    y = 0.2 * x[:, 0] + 0.1 * x[:, 1]
    result = grouped_ridge_oof(x, y, groups, alpha=1.0)
    assert np.isfinite(result.predictions).all()
    assert len(result.folds) == 4
    assert sum(fold["test_rows"] for fold in result.folds) == len(y)


def test_grouped_bootstrap_is_deterministic_and_group_aware():
    values = np.asarray([0.0, 0.0, 1.0, 1.0])
    groups = np.asarray(["a", "a", "b", "b"])
    left = grouped_bootstrap_mean_interval(values, groups, repetitions=100, seed=7)
    right = grouped_bootstrap_mean_interval(values, groups, repetitions=100, seed=7)
    assert left == right
    assert left["estimate"] == pytest.approx(0.5)


def test_hierarchical_variance_decomposition_reconstructs_total():
    outcomes = np.asarray([0, 0, 1, 1, 0, 1, 1, 0], dtype=float)
    cells = np.asarray(["a"] * 4 + ["b"] * 4)
    proposals = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
    result = hierarchical_variance_components(outcomes, cells, proposals)
    assert result["reconstruction_error"] < 1e-12
    assert sum(
        result[key]
        for key in (
            "state_goal_fraction",
            "proposal_within_state_goal_fraction",
            "continuation_within_proposal_fraction",
        )
    ) == pytest.approx(1.0)
