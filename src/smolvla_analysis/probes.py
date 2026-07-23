from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import grouped_episode_splits


def grouped_probe_predictions(
    features, labels, task_groups, episode_ids, *, max_components: int | None = None,
    group_folds: int | None = None,
):
    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int)
    groups = np.asarray(task_groups)
    episodes = np.asarray(episode_ids)
    if x.shape[0] != y.size or y.size != groups.size or groups.size != episodes.size:
        raise ValueError("features, labels, groups, and episode_ids must have equal row counts")
    predictions = np.full(y.size, np.nan)
    for train, test in grouped_episode_splits(groups, n_splits=group_folds):
        if set(episodes[train]) & set(episodes[test]):
            raise ValueError("episode leakage detected across train/test split")
        if np.unique(y[train]).size < 2:
            continue
        steps = [StandardScaler()]
        if max_components and x.shape[1] > max_components:
            components = min(max_components, len(train) - 1, x.shape[1])
            steps.append(PCA(n_components=components, svd_solver="randomized", random_state=0))
        steps.append(LogisticRegression(class_weight="balanced", max_iter=1000))
        model = make_pipeline(*steps)
        episode_counts = dict(zip(*np.unique(episodes[train], return_counts=True), strict=False))
        weights = np.asarray([1 / episode_counts[episode] for episode in episodes[train]], dtype=float)
        weights *= len(weights) / weights.sum()
        model.fit(x[train], y[train], logisticregression__sample_weight=weights)
        predictions[test] = model.predict_proba(x[test])[:, 1]
    return predictions


def grouped_majority_predictions(labels, task_groups, episode_ids, *, group_folds: int | None = None):
    y = np.asarray(labels, dtype=int)
    groups = np.asarray(task_groups)
    episodes = np.asarray(episode_ids)
    predictions = np.full(y.size, np.nan)
    for train, test in grouped_episode_splits(groups, n_splits=group_folds):
        unique_episodes, first = np.unique(episodes[train], return_index=True)
        del unique_episodes
        predictions[test] = float(y[train][first].mean())
    return predictions
