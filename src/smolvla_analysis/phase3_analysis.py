from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("Invalid binomial counts")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * np.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return float(center - half_width), float(center + half_width)


def empirical_beta_prior(successes: np.ndarray, trials: np.ndarray) -> dict[str, float]:
    successes = np.asarray(successes, dtype=np.float64)
    trials = np.asarray(trials, dtype=np.float64)
    if successes.shape != trials.shape or successes.ndim != 1 or len(successes) < 2:
        raise ValueError("Empirical beta prior requires aligned one-dimensional cell counts")
    if np.any(trials <= 0) or np.any(successes < 0) or np.any(successes > trials):
        raise ValueError("Invalid binomial cell counts")
    rates = successes / trials
    mean = float(successes.sum() / trials.sum())
    mean = float(np.clip(mean, 1e-6, 1.0 - 1e-6))
    observed_variance = float(np.var(rates, ddof=1))
    sampling_variance = float(np.mean(mean * (1.0 - mean) / trials))
    floor = mean * (1.0 - mean) / 1000.0
    between_variance = max(observed_variance - sampling_variance, floor)
    concentration = mean * (1.0 - mean) / between_variance - 1.0
    concentration = float(np.clip(concentration, 2.0, 1000.0))
    return {
        "alpha": mean * concentration,
        "beta": (1.0 - mean) * concentration,
        "mean": mean,
        "concentration": concentration,
        "observed_rate_variance": observed_variance,
        "estimated_sampling_variance": sampling_variance,
    }


def beta_posterior_mean(successes: np.ndarray, trials: np.ndarray, prior: dict[str, float]) -> np.ndarray:
    successes = np.asarray(successes, dtype=np.float64)
    trials = np.asarray(trials, dtype=np.float64)
    return (successes + prior["alpha"]) / (trials + prior["alpha"] + prior["beta"])


@dataclass(frozen=True)
class GroupedPrediction:
    predictions: np.ndarray
    folds: tuple[dict[str, Any], ...]
    rmse: float
    mae: float


def grouped_ridge_oof(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    *,
    alpha: float = 100.0,
) -> GroupedPrediction:
    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    groups = np.asarray(groups)
    if features.ndim != 2 or target.ndim != 1 or groups.ndim != 1:
        raise ValueError("Grouped prediction expects X[rows,features], y[rows], and groups[rows]")
    if len(features) != len(target) or len(target) != len(groups):
        raise ValueError("Grouped prediction arrays have different row counts")
    if len(np.unique(groups)) < 2 or not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError("Grouped prediction requires finite data and at least two groups")
    predictions = np.full(len(target), np.nan, dtype=np.float64)
    folds = []
    for held_out in np.unique(groups):
        test = groups == held_out
        train = ~test
        mean = features[train].mean(axis=0)
        scale = features[train].std(axis=0)
        scale[scale < 1e-12] = 1.0
        train_x = (features[train] - mean) / scale
        test_x = (features[test] - mean) / scale
        model = Ridge(alpha=alpha)
        model.fit(train_x, target[train])
        predictions[test] = model.predict(test_x)
        error = predictions[test] - target[test]
        folds.append(
            {
                "held_out_group": str(held_out),
                "train_rows": int(train.sum()),
                "test_rows": int(test.sum()),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "mae": float(np.mean(np.abs(error))),
            }
        )
    error = predictions - target
    if not np.isfinite(predictions).all():
        raise RuntimeError("Grouped predictions did not cover every row")
    return GroupedPrediction(
        predictions=predictions,
        folds=tuple(folds),
        rmse=float(np.sqrt(np.mean(np.square(error)))),
        mae=float(np.mean(np.abs(error))),
    )


def grouped_bootstrap_mean_interval(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    repetitions: int = 5000,
    seed: int = 20260728,
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    if values.ndim != 1 or groups.ndim != 1 or len(values) != len(groups):
        raise ValueError("Grouped bootstrap requires aligned one-dimensional arrays")
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("Grouped bootstrap requires at least two groups")
    by_group = {group: values[groups == group] for group in unique}
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        estimates[index] = np.concatenate([by_group[group] for group in sampled]).mean()
    return {
        "estimate": float(values.mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "bootstrap_repetitions": int(repetitions),
        "group_count": int(len(unique)),
    }


def hierarchical_variance_components(outcomes: np.ndarray, cell_ids: np.ndarray, proposal_ids: np.ndarray) -> dict[str, float]:
    outcomes = np.asarray(outcomes, dtype=np.float64)
    cell_ids = np.asarray(cell_ids)
    proposal_ids = np.asarray(proposal_ids)
    if not (len(outcomes) == len(cell_ids) == len(proposal_ids)):
        raise ValueError("Variance-component inputs have different row counts")
    global_mean = float(outcomes.mean())
    cell_mean = np.empty_like(outcomes)
    proposal_mean = np.empty_like(outcomes)
    combined = np.asarray([f"{cell}:{proposal}" for cell, proposal in zip(cell_ids, proposal_ids)])
    for cell in np.unique(cell_ids):
        mask = cell_ids == cell
        cell_mean[mask] = outcomes[mask].mean()
    for proposal in np.unique(combined):
        mask = combined == proposal
        proposal_mean[mask] = outcomes[mask].mean()
    components = {
        "state_goal": float(np.mean(np.square(cell_mean - global_mean))),
        "proposal_within_state_goal": float(np.mean(np.square(proposal_mean - cell_mean))),
        "continuation_within_proposal": float(np.mean(np.square(outcomes - proposal_mean))),
    }
    total = float(np.mean(np.square(outcomes - global_mean)))
    reconstructed = sum(components.values())
    return {
        "global_mean": global_mean,
        "total_variance": total,
        **components,
        "reconstruction_error": abs(total - reconstructed),
        **{
            f"{name}_fraction": (value / total if total > 0 else 0.0)
            for name, value in components.items()
        },
    }
