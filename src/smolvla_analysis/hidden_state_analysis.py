from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .probes import grouped_majority_predictions, grouped_probe_predictions


@dataclass
class HiddenStateDataset:
    metadata: pd.DataFrame
    activations: dict[str, np.ndarray]
    policy_features: np.ndarray
    state_features: np.ndarray
    uncertainty_features: np.ndarray
    activation_summary: pd.DataFrame


def _numeric(value) -> np.ndarray:
    if isinstance(value, np.ndarray) and value.dtype == object:
        value = value.tolist()
    return np.asarray(value, dtype=np.float32)


def _policy_features(row) -> np.ndarray:
    chunk = _numeric(row.predicted_action_chunk)
    if chunk.ndim != 2 or chunk.shape[1] != 7:
        raise ValueError(f"Invalid action chunk shape at {row.episode_id}/{row.env_step}: {chunk.shape}")
    return np.concatenate(
        [
            _numeric(row.executed_action), chunk.mean(axis=0), chunk.std(axis=0),
            chunk.min(axis=0), chunk.max(axis=0),
            np.asarray([row.action_norm, row.action_smoothness, row.action_jerk, row.gripper_action], dtype=np.float32),
        ]
    )


def load_hidden_state_dataset(run_dir: str | Path) -> HiddenStateDataset:
    root = Path(run_dir)
    episodes = pd.read_parquet(root / "episodes.parquet")
    steps = pd.read_parquet(root / "steps.parquet")
    sampled = steps.loc[steps["activation_reference"].notna()].copy()
    sampled = sampled.merge(
        episodes[["episode_id", "suite", "task_id", "condition", "success", "total_steps", "termination_reason"]],
        on="episode_id", how="left", validate="many_to_one",
    )
    uncertainty = pd.read_parquet(root / "summaries/offline_uncertainty.parquet")
    sampled = sampled.merge(uncertainty, on=["episode_id", "env_step"], how="left", validate="one_to_one")
    if sampled["translation_variance"].isna().any():
        raise ValueError("Missing uncertainty rows for activation-aligned steps")
    sampled["failure"] = (~sampled["success"].astype(bool)).astype(int)
    sampled["progress_bin"] = np.minimum((sampled["normalized_progress"] * 10).astype(int), 9)
    sampled["task_group"] = sampled["suite"].astype(str) + ":" + sampled["task_id"].astype(str)

    store = zarr.open_group(str(root / "activations.zarr"), mode="r")
    names: list[str] | None = None
    activation_lists: dict[str, list[np.ndarray]] = {}
    activation_rows = []
    for row in sampled.itertuples(index=False):
        group_path = str(row.activation_reference).removeprefix("activations.zarr/")
        group = store[group_path]
        current_names = sorted(name for name, _ in group.arrays())
        if names is None:
            names = current_names
            activation_lists = {name: [] for name in names}
        elif current_names != names:
            raise ValueError(f"Inconsistent activation targets at {group_path}")
        for name in names:
            array = group[name]
            vector = np.asarray(array[:], dtype=np.float32)
            if not np.isfinite(vector).all():
                raise ValueError(f"Non-finite activation: {group_path}/{name}")
            activation_lists[name].append(vector)
            activation_rows.append(
                {
                    "episode_id": row.episode_id, "env_step": row.env_step,
                    "normalized_progress": row.normalized_progress, "progress_bin": row.progress_bin,
                    "suite": row.suite, "task_id": row.task_id, "condition": row.condition, "success": row.success,
                    "pathway": array.attrs["pathway"], "layer_index": int(array.attrs["layer_index"]),
                    "pooled_l2": float(np.linalg.norm(vector)), "pooled_mean": float(vector.mean()),
                    "pooled_std": float(vector.std()),
                    "source_l2_norm_mean": float(array.attrs["l2_norm_mean"]),
                    "source_activation_mean": float(array.attrs["activation_mean"]),
                    "source_activation_std": float(array.attrs["activation_std"]),
                }
            )
    activations = {name: np.stack(values) for name, values in activation_lists.items()}
    policy = np.stack([_policy_features(row) for row in sampled.itertuples(index=False)])
    state = np.stack(
        [
            np.concatenate([_numeric(row.robot_state), _numeric(row.eef_state), _numeric(row.gripper_state)])
            for row in sampled.itertuples(index=False)
        ]
    )
    uncertainty_matrix = np.stack(
        [
            np.concatenate(
                [
                    np.asarray(
                        [
                            row.translation_variance, row.rotation_variance, row.gripper_disagreement,
                            row.mean_pairwise_chunk_distance,
                        ], dtype=np.float32,
                    ),
                    _numeric(row.variance_over_horizon),
                ]
            )
            for row in sampled.itertuples(index=False)
        ]
    )
    for label, matrix in (("policy", policy), ("state", state), ("uncertainty", uncertainty_matrix)):
        if not np.isfinite(matrix).all():
            raise ValueError(f"Non-finite {label} features")
    return HiddenStateDataset(
        sampled, activations, policy, state, uncertainty_matrix, pd.DataFrame(activation_rows)
    )


def load_multi_hidden_state_dataset(run_dirs: list[str | Path]) -> HiddenStateDataset:
    datasets = [load_hidden_state_dataset(path) for path in run_dirs]
    names = set(datasets[0].activations)
    if any(set(dataset.activations) != names for dataset in datasets[1:]):
        raise ValueError("Activation targets differ across runs")
    metadata_frames = []
    summary_frames = []
    for dataset in datasets:
        metadata = dataset.metadata.copy()
        summary = dataset.activation_summary.copy()
        prefix = metadata["condition"].astype(str) + ":"
        mapping = dict(zip(metadata["episode_id"], prefix + metadata["episode_id"], strict=False))
        metadata["episode_id"] = metadata["episode_id"].map(mapping)
        summary["episode_id"] = summary["episode_id"].map(mapping)
        metadata_frames.append(metadata)
        summary_frames.append(summary)
    return HiddenStateDataset(
        metadata=pd.concat(metadata_frames, ignore_index=True),
        activations={name: np.concatenate([dataset.activations[name] for dataset in datasets]) for name in sorted(names)},
        policy_features=np.concatenate([dataset.policy_features for dataset in datasets]),
        state_features=np.concatenate([dataset.state_features for dataset in datasets]),
        uncertainty_features=np.concatenate([dataset.uncertainty_features for dataset in datasets]),
        activation_summary=pd.concat(summary_frames, ignore_index=True),
    )


def _safe_metric(function, labels, scores) -> float:
    return float(function(labels, scores)) if np.unique(labels).size == 2 else float("nan")


def evaluate_predictions(metadata: pd.DataFrame, predictions: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    metrics = []
    columns = ["episode_id", "env_step", "normalized_progress", "task_id", "task_group", "failure"]
    if "condition" in metadata:
        columns.append("condition")
    base = metadata[columns]

    def add_metrics(feature_set: str, frame: pd.DataFrame, evaluation_unit: str) -> None:
        episode = frame.groupby("episode_id", as_index=False).agg(
            failure=("failure", "first"), task_id=("task_id", "first"),
            failure_probability=("failure_probability", "mean"),
        )
        labels = episode["failure"].to_numpy(int)
        probability = episode["failure_probability"].to_numpy(float)
        prediction = probability >= 0.5
        successful = episode["failure"] == 0
        metrics.append(
            {
                "feature_set": feature_set, "evaluation_unit": evaluation_unit, "n_episodes": len(episode),
                "auroc": _safe_metric(roc_auc_score, labels, probability),
                "auprc": _safe_metric(average_precision_score, labels, probability),
                "precision": float(precision_score(labels, prediction, zero_division=0)),
                "recall": float(recall_score(labels, prediction, zero_division=0)),
                "f1": float(f1_score(labels, prediction, zero_division=0)),
                "brier": float(brier_score_loss(labels, probability)),
                "false_alarm_episodes": int((prediction[successful]).sum()),
                "false_alarm_rate_per_success_episode": float(prediction[successful].mean()),
            }
        )

    for feature_set, scores in predictions.items():
        if not np.isfinite(scores).all():
            raise ValueError(f"Non-finite predictions for {feature_set}")
        frame = base.copy()
        frame["feature_set"] = feature_set
        frame["failure_probability"] = scores
        rows.append(frame)
        add_metrics(feature_set, frame, "full_episode")
        for horizon in (0, 25, 50, 75, 100):
            add_metrics(feature_set, frame.loc[frame["env_step"] <= horizon], f"up_to_step_{horizon}")
    return pd.concat(rows, ignore_index=True), pd.DataFrame(metrics)


def run_grouped_probes(
    dataset: HiddenStateDataset, *, pca_components: int = 32, group_folds: int | None = None,
):
    metadata = dataset.metadata
    labels = metadata["failure"].to_numpy(int)
    groups = metadata["task_group"].to_numpy()
    episodes = metadata["episode_id"].to_numpy()
    vlm_final = dataset.activations["vlm_layer_15"]
    expert_final = dataset.activations["action_expert_layer_15"]
    feature_sets = {
        "policy_output": dataset.policy_features,
        "robot_state": dataset.state_features,
        "action_uncertainty": dataset.uncertainty_features,
        "vlm_final": vlm_final,
        "action_expert_final": expert_final,
        "vlm_action_expert": np.concatenate([vlm_final, expert_final], axis=1),
        "all_diagnostic_features": np.concatenate(
            [dataset.policy_features, dataset.state_features, dataset.uncertainty_features, vlm_final, expert_final], axis=1
        ),
    }
    predictions = {"majority_baseline": grouped_majority_predictions(labels, groups, episodes, group_folds=group_folds)}
    for name, values in feature_sets.items():
        predictions[name] = grouped_probe_predictions(
            values, labels, groups, episodes,
            max_components=pca_components if values.shape[1] > pca_components else None,
            group_folds=group_folds,
        )
    for name, values in dataset.activations.items():
        predictions[f"layer_{name}"] = grouped_probe_predictions(
            values, labels, groups, episodes, max_components=pca_components, group_folds=group_folds,
        )
    return evaluate_predictions(metadata, predictions)


def activation_statistics(dataset: HiddenStateDataset) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = dataset.metadata
    summary = dataset.activation_summary.copy()
    centroid_rows = []
    divergence_rows = []
    for name, values in dataset.activations.items():
        pathway, layer_text = name.rsplit("_layer_", 1)
        layer = int(layer_text)
        distances = np.full(len(metadata), np.nan)
        similarities = np.full(len(metadata), np.nan)
        thresholds: dict[tuple[str, int], float] = {}
        for (task_group, progress_bin), indices in metadata.groupby(["task_group", "progress_bin"]).groups.items():
            indices = np.asarray(list(indices), dtype=int)
            success_indices = indices[metadata.iloc[indices]["success"].to_numpy(bool)]
            if len(success_indices) < 2:
                continue
            centroid = values[success_indices].mean(axis=0)
            centroid_norm = np.linalg.norm(centroid)
            group_values = values[indices]
            distance = np.linalg.norm(group_values - centroid, axis=1)
            similarity = (group_values @ centroid) / np.maximum(np.linalg.norm(group_values, axis=1) * centroid_norm, 1e-12)
            distances[indices] = distance
            similarities[indices] = similarity
            success_distances = np.linalg.norm(values[success_indices] - centroid, axis=1)
            thresholds[(str(task_group), int(progress_bin))] = float(np.quantile(success_distances, 0.95))
        for index, row in metadata.iterrows():
            centroid_rows.append(
                {
                    "episode_id": row.episode_id, "env_step": row.env_step,
                    "normalized_progress": row.normalized_progress, "progress_bin": row.progress_bin,
                    "suite": row.suite, "task_id": row.task_id, "condition": row.condition,
                    "success": row.success, "pathway": pathway, "layer_index": layer,
                    "distance_to_task_progress_success_centroid": distances[index],
                    "cosine_to_task_progress_success_centroid": similarities[index],
                }
            )
        failures = metadata.loc[~metadata["success"].astype(bool)]
        for episode_id, episode_rows in failures.groupby("episode_id"):
            episode_rows = episode_rows.sort_values("env_step")
            flags = []
            for index, row in episode_rows.iterrows():
                threshold = thresholds.get((str(row.task_group), int(row.progress_bin)))
                flags.append(bool(threshold is not None and np.isfinite(distances[index]) and distances[index] > threshold))
            first = None
            for offset in range(max(0, len(flags) - 1)):
                if flags[offset] and flags[offset + 1]:
                    first = episode_rows.iloc[offset]
                    break
            divergence_rows.append(
                {
                    "episode_id": episode_id, "suite": episode_rows.iloc[0].suite,
                    "task_id": int(episode_rows.iloc[0].task_id), "condition": episode_rows.iloc[0].condition,
                    "pathway": pathway, "layer_index": layer,
                    "first_divergence_step": None if first is None else int(first.env_step),
                    "first_divergence_progress": None if first is None else float(first.normalized_progress),
                    "reference_available": any(
                        (str(row.task_group), int(row.progress_bin)) in thresholds for row in episode_rows.itertuples()
                    ),
                    "criterion": "two consecutive samples above task/progress-bin successful-centroid 95th-percentile distance",
                }
            )
    action_distances = np.full(len(metadata), np.nan)
    action_thresholds: dict[tuple[str, int], float] = {}
    for (task_group, progress_bin), indices in metadata.groupby(["task_group", "progress_bin"]).groups.items():
        indices = np.asarray(list(indices), dtype=int)
        success_indices = indices[metadata.iloc[indices]["success"].to_numpy(bool)]
        if len(success_indices) < 2:
            continue
        reference = dataset.policy_features[success_indices].mean(axis=0)
        action_distances[indices] = np.linalg.norm(dataset.policy_features[indices] - reference, axis=1)
        action_thresholds[(str(task_group), int(progress_bin))] = float(
            np.quantile(np.linalg.norm(dataset.policy_features[success_indices] - reference, axis=1), 0.95)
        )
    failures = metadata.loc[~metadata["success"].astype(bool)]
    for episode_id, episode_rows in failures.groupby("episode_id"):
        episode_rows = episode_rows.sort_values("env_step")
        flags = [
            bool(
                (threshold := action_thresholds.get((str(row.task_group), int(row.progress_bin)))) is not None
                and np.isfinite(action_distances[index]) and action_distances[index] > threshold
            )
            for index, row in episode_rows.iterrows()
        ]
        first = None
        for offset in range(max(0, len(flags) - 1)):
            if flags[offset] and flags[offset + 1]:
                first = episode_rows.iloc[offset]
                break
        divergence_rows.append(
            {
                "episode_id": episode_id, "suite": episode_rows.iloc[0].suite,
                "task_id": int(episode_rows.iloc[0].task_id), "condition": episode_rows.iloc[0].condition,
                "pathway": "action_output", "layer_index": -1,
                "first_divergence_step": None if first is None else int(first.env_step),
                "first_divergence_progress": None if first is None else float(first.normalized_progress),
                "reference_available": any(
                    (str(row.task_group), int(row.progress_bin)) in action_thresholds for row in episode_rows.itertuples()
                ),
                "criterion": "two consecutive samples above task/progress-bin successful-action-centroid 95th-percentile distance",
            }
        )
    return summary, pd.DataFrame(centroid_rows), pd.DataFrame(divergence_rows)


def hidden_pca(dataset: HiddenStateDataset) -> pd.DataFrame:
    rows = []
    for name in ("vlm_layer_15", "action_expert_layer_15"):
        values = dataset.activations[name]
        coordinates = PCA(n_components=2, random_state=0).fit_transform(values)
        pathway = name.removesuffix("_layer_15")
        for index, row in dataset.metadata.iterrows():
            rows.append(
                {
                    "episode_id": row.episode_id, "env_step": row.env_step, "suite": row.suite,
                    "task_id": row.task_id, "condition": row.condition, "success": row.success,
                    "task_phase": row.task_phase, "pathway": pathway,
                    "pc1": float(coordinates[index, 0]), "pc2": float(coordinates[index, 1]),
                }
            )
    return pd.DataFrame(rows)
