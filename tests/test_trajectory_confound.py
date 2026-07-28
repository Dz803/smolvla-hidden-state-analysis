import numpy as np
import pandas as pd

from scripts.audit_trajectory_confound import (
    blockwise_grouped_predictions,
    low_resolution_frame,
    ordered_action_chunk,
    ordered_action_prefix,
)


def test_ordered_action_prefix_excludes_landmark_action():
    steps = pd.DataFrame(
        {
            "episode_id": ["a", "a", "a", "b", "b", "b"],
            "env_step": [0, 1, 2, 0, 1, 2],
            "executed_action": [
                [1.0], [2.0], [999.0], [10.0], [20.0], [999.0],
            ],
        }
    )
    result = ordered_action_prefix(steps, np.asarray(["a", "b"]), landmark=2)
    np.testing.assert_allclose(result, [[1.0, 2.0], [10.0, 20.0]])


def test_ordered_action_chunk_accepts_object_arrays_from_parquet():
    row = np.empty(50, dtype=object)
    for index in range(50):
        row[index] = np.full(7, index, dtype=np.float32)
    result = ordered_action_chunk(pd.Series([row]))
    assert result.shape == (1, 350)
    np.testing.assert_allclose(result[0, :7], np.zeros(7))
    np.testing.assert_allclose(result[0, -7:], np.full(7, 49))


def test_low_resolution_frame_uses_exact_block_means():
    frame = np.arange(3 * 4 * 4, dtype=np.uint8).reshape(3, 4, 4)
    result = low_resolution_frame(frame, bins=2).reshape(3, 2, 2)
    expected = frame.reshape(3, 2, 2, 2, 2).mean(axis=(2, 4)) / 255.0
    np.testing.assert_allclose(result, expected)


def test_blockwise_grouped_predictions_hold_out_tasks():
    rng = np.random.default_rng(0)
    groups = np.repeat(["a", "b", "c", "d"], 4)
    labels = np.tile([0, 0, 1, 1], 4)
    episodes = np.asarray([f"episode_{index}" for index in range(len(labels))])
    signal = labels[:, None] + rng.normal(scale=0.02, size=(len(labels), 3))
    noise = rng.normal(size=(len(labels), 5))
    predictions, selected_regularization = blockwise_grouped_predictions(
        [signal, noise], labels, groups, episodes, pca_components=2, group_folds=2,
    )
    assert np.isfinite(predictions).all()
    assert np.isfinite(selected_regularization).all()
    assert predictions[labels == 1].mean() > predictions[labels == 0].mean()


def test_blockwise_grouped_predictions_handle_constant_blocks(recwarn):
    groups = np.repeat(["a", "b", "c", "d"], 4)
    labels = np.tile([0, 0, 1, 1], 4)
    episodes = np.asarray([f"episode_{index}" for index in range(len(labels))])
    predictions, _ = blockwise_grouped_predictions(
        [np.zeros((len(labels), 100))],
        labels,
        groups,
        episodes,
        pca_components=2,
        group_folds=2,
    )
    assert np.isfinite(predictions).all()
    assert not recwarn.list
