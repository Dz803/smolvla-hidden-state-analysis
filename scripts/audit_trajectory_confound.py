#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from smolvla_analysis.hidden_state_analysis import load_hidden_state_dataset
from smolvla_analysis.metrics import grouped_episode_splits


LANDMARKS = (0, 50, 100)
REGULARIZATION_VALUES = (0.001, 0.01, 0.1, 1.0, 10.0)


def ordered_action_prefix(
    steps: pd.DataFrame, episode_ids: np.ndarray, landmark: int,
) -> np.ndarray:
    """Flatten actions executed before the landmark observation, never at or after it."""
    if landmark == 0:
        return np.zeros((len(episode_ids), 1), dtype=np.float32)
    indexed = steps.set_index(["episode_id", "env_step"])["executed_action"]
    rows = []
    expected_steps = list(range(landmark))
    for episode_id in episode_ids:
        episode = indexed.loc[episode_id]
        missing = sorted(set(expected_steps) - set(episode.index))
        if missing:
            raise ValueError(f"Missing pre-landmark actions for {episode_id}: {missing[:5]}")
        rows.append(
            np.concatenate(
                [np.asarray(episode.loc[step], dtype=np.float32) for step in expected_steps]
            )
        )
    return np.stack(rows)


def ordered_action_chunk(values: pd.Series) -> np.ndarray:
    rows = []
    for value in values:
        if isinstance(value, np.ndarray) and value.dtype == object:
            value = value.tolist()
        chunk = np.asarray(value, dtype=np.float32)
        if chunk.shape != (50, 7):
            raise ValueError(f"Expected a full 50x7 chunk at a replanning boundary, found {chunk.shape}")
        rows.append(chunk.reshape(-1))
    return np.stack(rows)


def low_resolution_frame(frame: np.ndarray, bins: int) -> np.ndarray:
    image = np.asarray(frame, dtype=np.float32)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected a channel-first RGB frame, found {image.shape}")
    height, width = image.shape[1:]
    if height % bins or width % bins:
        raise ValueError(f"Frame shape {image.shape} is not divisible into {bins} bins")
    pooled = image.reshape(3, bins, height // bins, bins, width // bins).mean(axis=(2, 4))
    return (pooled / 255.0).reshape(-1)


def load_visual_landmarks(
    run: Path, episode_ids: np.ndarray, *, bins: int,
) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    """Load exact pre-action frames once per episode and return current/change features."""
    result = {}
    for episode_id in episode_ids:
        path = run / "observations" / f"{episode_id}.npz"
        with np.load(path, allow_pickle=False) as observations:
            camera1 = observations["camera1"]
            camera2 = observations["camera2"]
            initial = np.concatenate(
                [low_resolution_frame(camera1[0], bins), low_resolution_frame(camera2[0], bins)]
            )
            for landmark in LANDMARKS:
                if landmark >= len(camera1) or landmark >= len(camera2):
                    continue
                current = np.concatenate(
                    [
                        low_resolution_frame(camera1[landmark], bins),
                        low_resolution_frame(camera2[landmark], bins),
                    ]
                )
                result[(str(episode_id), landmark)] = current, np.abs(current - initial)
    return result


def _transform_blocks(
    blocks: list[np.ndarray], train: np.ndarray, test: np.ndarray, pca_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    transformed_train = []
    transformed_test = []
    for block in blocks:
        varying = np.ptp(block[train], axis=0) > 1e-12
        if not varying.any():
            transformed_train.append(np.zeros((len(train), 1), dtype=float))
            transformed_test.append(np.zeros((len(test), 1), dtype=float))
            continue
        scaler = StandardScaler()
        train_block = scaler.fit_transform(block[train][:, varying])
        test_block = scaler.transform(block[test][:, varying])
        if train_block.shape[1] > pca_components:
            components = min(pca_components, len(train) - 1, train_block.shape[1])
            pca = PCA(n_components=components, svd_solver="randomized", random_state=0)
            train_block = pca.fit_transform(train_block)
            test_block = pca.transform(test_block)
        transformed_train.append(train_block)
        transformed_test.append(test_block)
    return np.concatenate(transformed_train, axis=1), np.concatenate(transformed_test, axis=1)


def blockwise_grouped_predictions(
    blocks: list[np.ndarray],
    labels: np.ndarray,
    task_groups: np.ndarray,
    episode_ids: np.ndarray,
    *,
    pca_components: int,
    group_folds: int,
    regularization_values: tuple[float, ...] = REGULARIZATION_VALUES,
) -> tuple[np.ndarray, np.ndarray]:
    """Select regularization in nested task folds and predict outer held-out tasks."""
    arrays = [np.asarray(block, dtype=float) for block in blocks]
    row_count = len(labels)
    if not arrays or any(block.ndim != 2 or len(block) != row_count for block in arrays):
        raise ValueError("Every feature block must be a non-empty 2D matrix with one row per label")
    if not regularization_values or any(value <= 0 for value in regularization_values):
        raise ValueError("regularization_values must contain positive values")
    predictions = np.full(row_count, np.nan)
    selected_regularization = np.full(row_count, np.nan)
    for train, test in grouped_episode_splits(task_groups, n_splits=group_folds):
        if set(episode_ids[train]) & set(episode_ids[test]):
            raise ValueError("Episode leakage across folds")
        inner_scores = {value: np.full(len(train), np.nan) for value in regularization_values}
        inner_groups = task_groups[train]
        inner_folds = min(3, np.unique(inner_groups).size)
        for inner_train_local, inner_test_local in grouped_episode_splits(
            inner_groups, n_splits=inner_folds,
        ):
            inner_train = train[inner_train_local]
            inner_test = train[inner_test_local]
            x_train, x_test = _transform_blocks(
                arrays, inner_train, inner_test, pca_components,
            )
            for value in regularization_values:
                model = LogisticRegression(C=value, class_weight="balanced", max_iter=2000)
                model.fit(x_train, labels[inner_train])
                inner_scores[value][inner_test_local] = model.predict_proba(x_test)[:, 1]
        best_value = min(
            regularization_values,
            key=lambda value: log_loss(labels[train], inner_scores[value], labels=[0, 1]),
        )
        x_train, x_test = _transform_blocks(arrays, train, test, pca_components)
        model = LogisticRegression(C=best_value, class_weight="balanced", max_iter=2000)
        model.fit(x_train, labels[train])
        predictions[test] = model.predict_proba(x_test)[:, 1]
        selected_regularization[test] = best_value
    if not np.isfinite(predictions).all():
        raise ValueError("Some task-held-out predictions are missing or non-finite")
    return predictions, selected_regularization


def within_task_pairwise_auc(labels: np.ndarray, scores: np.ndarray, groups: np.ndarray) -> float:
    weighted_auc = 0.0
    comparable_pairs = 0
    for group in np.unique(groups):
        mask = groups == group
        if np.unique(labels[mask]).size != 2:
            continue
        failures = int(labels[mask].sum())
        successes = int(mask.sum() - failures)
        pairs = failures * successes
        weighted_auc += roc_auc_score(labels[mask], scores[mask]) * pairs
        comparable_pairs += pairs
    return float(weighted_auc / comparable_pairs) if comparable_pairs else float("nan")


def metric_row(
    feature_set: str,
    landmark: int,
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float | int | str]:
    prevalence = float(labels.mean())
    auprc = float(average_precision_score(labels, scores))
    return {
        "feature_set": feature_set,
        "landmark": landmark,
        "episodes_at_risk": len(labels),
        "failure_prevalence": prevalence,
        "auprc": auprc,
        "normalized_auprc_gain": (auprc - prevalence) / (1 - prevalence),
        "auroc": float(roc_auc_score(labels, scores)),
        "within_task_pairwise_auc": within_task_pairwise_auc(labels, scores, groups),
        "brier": float(brier_score_loss(labels, scores)),
        "log_loss": float(log_loss(labels, scores, labels=[0, 1])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--group-folds", type=int, default=5)
    parser.add_argument("--image-bins", type=int, default=16)
    args = parser.parse_args()

    dataset = load_hidden_state_dataset(args.run)
    steps = pd.read_parquet(args.run / "steps.parquet")
    all_episodes = dataset.metadata.loc[
        dataset.metadata["env_step"].isin(LANDMARKS), "episode_id"
    ].astype(str).unique()
    print(f"Loading exact landmark images for {len(all_episodes)} episodes...")
    visual_landmarks = load_visual_landmarks(args.run, all_episodes, bins=args.image_bins)
    rows = []
    prediction_frames = []
    for landmark in LANDMARKS:
        mask = dataset.metadata["env_step"].to_numpy() == landmark
        metadata = dataset.metadata.loc[mask].reset_index(drop=True)
        if metadata.empty:
            raise ValueError(f"No activation-aligned observations at landmark {landmark}")
        labels = metadata["failure"].to_numpy(int)
        groups = metadata["task_group"].astype(str).to_numpy()
        episodes = metadata["episode_id"].astype(str).to_numpy()
        boundary_rows = steps.loc[
            (steps["env_step"] == landmark) & steps["episode_id"].isin(episodes)
        ].set_index("episode_id").loc[episodes]
        action_prefix = ordered_action_prefix(steps, episodes, landmark)
        action_chunk = ordered_action_chunk(boundary_rows["predicted_action_chunk"])
        robot_state = dataset.state_features[mask]
        action_expert = dataset.activations["action_expert_layer_15"][mask]
        vlm = dataset.activations["vlm_layer_15"][mask]
        current_pixels = np.stack([visual_landmarks[(episode, landmark)][0] for episode in episodes])
        pixel_change = np.stack([visual_landmarks[(episode, landmark)][1] for episode in episodes])
        block_sets = {
            "robot_state": [robot_state],
            "executed_action_prefix": [action_prefix],
            "ordered_action_chunk": [action_chunk],
            "behavioral_context": [robot_state, action_prefix, action_chunk],
            "lowres_current_pixels": [current_pixels],
            "lowres_pixel_change": [pixel_change],
            "visual_behavioral_context": [
                robot_state, action_prefix, action_chunk, current_pixels, pixel_change,
            ],
            "action_expert_final": [action_expert],
            "vlm_final": [vlm],
            "behavioral_context_plus_action_expert": [
                robot_state, action_prefix, action_chunk, action_expert,
            ],
            "behavioral_context_plus_vlm": [robot_state, action_prefix, action_chunk, vlm],
            "visual_behavioral_context_plus_action_expert": [
                robot_state, action_prefix, action_chunk, current_pixels, pixel_change, action_expert,
            ],
            "visual_behavioral_context_plus_vlm": [
                robot_state, action_prefix, action_chunk, current_pixels, pixel_change, vlm,
            ],
        }
        for feature_set, blocks in block_sets.items():
            scores, selected_regularization = blockwise_grouped_predictions(
                blocks,
                labels,
                groups,
                episodes,
                pca_components=args.pca_components,
                group_folds=args.group_folds,
            )
            rows.append(metric_row(feature_set, landmark, labels, scores, groups))
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "episode_id": episodes,
                        "task_group": groups,
                        "landmark": landmark,
                        "failure": labels,
                        "feature_set": feature_set,
                        "failure_probability": scores,
                        "selected_regularization_c": selected_regularization,
                    }
                )
            )

    args.output.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows).sort_values(["landmark", "auprc"], ascending=[True, False])
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics.to_csv(args.output / "trajectory_confound_metrics.csv", index=False)
    predictions.to_parquet(args.output / "trajectory_confound_predictions.parquet", index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
