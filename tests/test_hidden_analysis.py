import numpy as np
import pandas as pd

from smolvla_analysis.hidden_state_analysis import evaluate_predictions
from smolvla_analysis.probes import grouped_probe_predictions


def test_hidden_probe_is_finite_with_grouped_pca():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(12, 40))
    labels = np.tile([0, 1], 6)
    groups = np.repeat([0, 1, 2], 4)
    episodes = np.asarray([f"episode_{index}" for index in range(12)])
    predictions = grouped_probe_predictions(
        features, labels, groups, episodes, max_components=4
    )
    assert np.isfinite(predictions).all()


def test_probe_metrics_are_aggregated_by_episode():
    metadata = pd.DataFrame(
        {
            "episode_id": ["a", "a", "b", "b"],
            "env_step": [0, 5, 0, 5],
            "normalized_progress": [0.0, 0.5, 0.0, 0.5],
            "task_id": [0, 0, 1, 1],
            "task_group": ["s:0", "s:0", "s:1", "s:1"],
            "failure": [0, 0, 1, 1],
        }
    )
    predictions, metrics = evaluate_predictions(
        metadata, {"perfect": np.asarray([0.1, 0.2, 0.8, 0.9])}
    )
    assert len(predictions) == 4
    full = metrics.loc[metrics["evaluation_unit"] == "full_episode"].iloc[0]
    assert full["n_episodes"] == 2
    assert full["auroc"] == 1.0
