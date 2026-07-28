from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def recent_state_action_history_features(
    metadata: pd.DataFrame,
    state_features: np.ndarray,
    policy_features: np.ndarray,
    *,
    window: int = 5,
) -> np.ndarray:
    """Summarize the current and recent sampled state/action history per episode."""
    if window < 1:
        raise ValueError("window must be at least one")
    if len(metadata) != len(state_features) or len(metadata) != len(policy_features):
        raise ValueError("metadata, state_features, and policy_features must have equal row counts")
    required = {"episode_id", "env_step"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"metadata must contain {sorted(required)}")

    current = np.concatenate(
        [np.asarray(state_features, dtype=float), np.asarray(policy_features, dtype=float)], axis=1,
    )
    history = np.empty((len(metadata), current.shape[1] * 4), dtype=float)
    for _, indices in metadata.groupby("episode_id", sort=False).groups.items():
        ordered = np.asarray(list(indices), dtype=int)
        ordered = ordered[np.argsort(metadata.iloc[ordered]["env_step"].to_numpy())]
        for position, index in enumerate(ordered):
            start = max(0, position - window + 1)
            values = current[ordered[start : position + 1]]
            history[index] = np.concatenate(
                [current[index], values.mean(axis=0), values.std(axis=0), current[index] - values[0]],
            )
    if not np.isfinite(history).all():
        raise ValueError("history features must be finite")
    return history


def _fit_predict_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_episodes: np.ndarray,
    test_features: np.ndarray,
    *,
    max_components: int | None,
) -> np.ndarray:
    if np.unique(train_labels).size < 2:
        return np.full(len(test_features), np.nan)
    steps = [StandardScaler()]
    if max_components and train_features.shape[1] > max_components:
        components = min(max_components, len(train_features) - 1, train_features.shape[1])
        steps.append(PCA(n_components=components, svd_solver="randomized", random_state=0))
    steps.append(LogisticRegression(class_weight="balanced", max_iter=1000))
    model = make_pipeline(*steps)
    episode_counts = dict(zip(*np.unique(train_episodes, return_counts=True), strict=False))
    weights = np.asarray([1 / episode_counts[episode] for episode in train_episodes], dtype=float)
    weights *= len(weights) / weights.sum()
    model.fit(train_features, train_labels, logisticregression__sample_weight=weights)
    return model.predict_proba(test_features)[:, 1]


def leave_one_group_out_probe_predictions(
    features,
    labels,
    split_groups,
    episode_ids,
    *,
    max_components: int | None = None,
) -> np.ndarray:
    """Fit fold-local probes and predict each held-out split group."""
    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int)
    groups = np.asarray(split_groups)
    episodes = np.asarray(episode_ids)
    if x.shape[0] != y.size or y.size != groups.size or groups.size != episodes.size:
        raise ValueError("features, labels, split_groups, and episode_ids must have equal row counts")
    predictions = np.full(y.size, np.nan)
    for held_out in np.unique(groups):
        train = groups != held_out
        test = groups == held_out
        if set(episodes[train]) & set(episodes[test]):
            raise ValueError("episode leakage detected across train/test split")
        predictions[test] = _fit_predict_probe(
            x[train], y[train], episodes[train], x[test], max_components=max_components,
        )
    return predictions


def task_grouped_condition_transfer_predictions(
    source_features,
    source_labels,
    source_task_groups,
    source_episode_ids,
    target_features,
    target_task_groups,
    target_episode_ids,
    *,
    max_components: int | None = None,
) -> np.ndarray:
    """Train on source-condition tasks and predict the matching held-out target tasks."""
    source_x = np.asarray(source_features, dtype=float)
    source_y = np.asarray(source_labels, dtype=int)
    source_groups = np.asarray(source_task_groups)
    source_episodes = np.asarray(source_episode_ids)
    target_x = np.asarray(target_features, dtype=float)
    target_groups = np.asarray(target_task_groups)
    target_episodes = np.asarray(target_episode_ids)
    if source_x.shape[0] != source_y.size or source_y.size != source_groups.size or source_groups.size != source_episodes.size:
        raise ValueError("source arrays must have equal row counts")
    if target_x.shape[0] != target_groups.size or target_groups.size != target_episodes.size:
        raise ValueError("target arrays must have equal row counts")
    if source_x.shape[1] != target_x.shape[1]:
        raise ValueError("source and target features must have the same width")
    if set(source_episodes) & set(target_episodes):
        raise ValueError("source and target episode IDs must be disjoint")

    predictions = np.full(len(target_x), np.nan)
    for held_out in np.unique(target_groups):
        train = source_groups != held_out
        test = target_groups == held_out
        if not test.any():
            continue
        if held_out not in set(source_groups):
            continue
        predictions[test] = _fit_predict_probe(
            source_x[train], source_y[train], source_episodes[train], target_x[test],
            max_components=max_components,
        )
    return predictions
