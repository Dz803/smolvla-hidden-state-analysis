from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold


def bootstrap_rate_interval(
    values, *, samples: int = 2000, confidence: float = 0.95, seed: int = 0
) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or data.size == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if not np.isfinite(data).all():
        raise ValueError("values must be finite")
    rng = np.random.default_rng(seed)
    draws = rng.choice(data, size=(samples, data.size), replace=True).mean(axis=1)
    alpha = (1 - confidence) / 2
    return float(data.mean()), float(np.quantile(draws, alpha)), float(np.quantile(draws, 1 - alpha))


def grouped_episode_splits(groups, *, n_splits: int | None = None):
    groups = np.asarray(groups)
    if groups.ndim != 1 or groups.size == 0:
        raise ValueError("groups must be a non-empty one-dimensional sequence")
    unique = np.unique(groups)
    if unique.size < 2:
        raise ValueError("at least two task groups are required")
    indices = np.arange(groups.size)
    if n_splits is not None and n_splits < unique.size:
        if n_splits < 2:
            raise ValueError("n_splits must be at least two")
        yield from GroupKFold(n_splits=n_splits).split(indices, groups=groups)
        return
    for held_out in unique:
        test = indices[groups == held_out]
        train = indices[groups != held_out]
        yield train, test
