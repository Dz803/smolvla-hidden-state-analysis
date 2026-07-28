import numpy as np
import pandas as pd

from scripts.audit_warning_confounds import bootstrap_metric_intervals
from smolvla_analysis.robustness import (
    leave_one_group_out_probe_predictions,
    recent_state_action_history_features,
    task_grouped_condition_transfer_predictions,
)


def test_clustered_bootstrap_is_deterministic():
    frame = pd.DataFrame(
        {
            "task_group": np.repeat(["a", "b", "c"], 4),
            "failure": np.tile([0, 0, 1, 1], 3),
            "score": np.tile([0.1, 0.2, 0.8, 0.9], 3),
        }
    )
    first = bootstrap_metric_intervals(frame, samples=20, confidence=0.95, seed=7)
    second = bootstrap_metric_intervals(frame, samples=20, confidence=0.95, seed=7)
    assert first == second
    assert first["auroc_episode_ci_low"] == 1.0
    assert first["within_task_pairwise_auc_task_cluster_ci_high"] == 1.0


def test_recent_history_does_not_cross_episode_boundaries():
    metadata = pd.DataFrame(
        {"episode_id": ["a", "b", "a", "b"], "env_step": [0, 0, 5, 5]}
    )
    state = np.asarray([[1.0], [10.0], [3.0], [14.0]])
    policy = np.asarray([[2.0], [20.0], [6.0], [28.0]])
    history = recent_state_action_history_features(metadata, state, policy, window=2)
    np.testing.assert_allclose(history[2], [3, 6, 2, 4, 1, 2, 2, 4])
    np.testing.assert_allclose(history[3], [14, 28, 12, 24, 2, 4, 4, 8])


def test_leave_one_group_out_probe_predictions_are_finite():
    rng = np.random.default_rng(0)
    labels = np.tile([0, 1], 6)
    groups = np.repeat(["a", "b", "c"], 4)
    features = labels[:, None] + rng.normal(scale=0.05, size=(12, 3))
    episodes = np.asarray([f"episode_{index}" for index in range(12)])
    predictions = leave_one_group_out_probe_predictions(
        features, labels, groups, episodes, max_components=2,
    )
    assert np.isfinite(predictions).all()
    assert predictions[labels == 1].mean() > predictions[labels == 0].mean()


def test_condition_transfer_holds_out_target_task_from_source_fit():
    rng = np.random.default_rng(1)
    source_labels = np.tile([0, 1], 3)
    source_groups = np.repeat(["task_a", "task_b", "task_c"], 2)
    target_groups = source_groups.copy()
    source = source_labels[:, None] + rng.normal(scale=0.02, size=(6, 2))
    target = source_labels[:, None] + rng.normal(scale=0.02, size=(6, 2))
    predictions = task_grouped_condition_transfer_predictions(
        source,
        source_labels,
        source_groups,
        np.asarray([f"source_{index}" for index in range(6)]),
        target,
        target_groups,
        np.asarray([f"target_{index}" for index in range(6)]),
        max_components=1,
    )
    assert np.isfinite(predictions).all()
    assert predictions[source_labels == 1].mean() > predictions[source_labels == 0].mean()
