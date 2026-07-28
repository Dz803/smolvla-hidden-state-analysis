#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from smolvla_analysis.hidden_state_analysis import HiddenStateDataset, load_hidden_state_dataset
from smolvla_analysis.probes import grouped_probe_predictions
from smolvla_analysis.robustness import (
    leave_one_group_out_probe_predictions,
    recent_state_action_history_features,
    task_grouped_condition_transfer_predictions,
)


HORIZONS = (0, 50, 100)
PRIMARY_FEATURES = {
    "action_expert_final",
    "action_uncertainty",
    "policy_output",
    "robot_state",
    "state_action_history",
    "vlm_final",
}
BOOTSTRAP_METRICS = (
    "auprc",
    "auroc",
    "brier",
    "task_centered_auprc",
    "task_centered_auroc",
    "within_task_pairwise_auc",
    "task_eta_squared",
)


def safe_metric(function, labels: np.ndarray, scores: np.ndarray) -> float:
    return float(function(labels, scores)) if np.unique(labels).size == 2 else float("nan")


def _metric_arrays(
    labels: np.ndarray,
    scores: np.ndarray,
    group_codes: np.ndarray,
) -> dict[str, float | int]:
    unique_groups, normalized_groups = np.unique(group_codes, return_inverse=True)
    counts = np.bincount(normalized_groups)
    means = np.bincount(normalized_groups, weights=scores) / counts
    centered = scores - means[normalized_groups]
    weighted_auc = 0.0
    comparable_pairs = 0
    eligible_tasks = 0
    for group_index in range(len(unique_groups)):
        mask = normalized_groups == group_index
        group_labels = labels[mask]
        if np.unique(group_labels).size != 2:
            continue
        failures = int(group_labels.sum())
        successes = int(len(group_labels) - failures)
        pairs = failures * successes
        weighted_auc += roc_auc_score(group_labels, scores[mask]) * pairs
        comparable_pairs += pairs
        eligible_tasks += 1
    grand = scores.mean()
    total = np.square(scores - grand).sum()
    between = float(np.sum(counts * np.square(means - grand)))
    return {
        "episodes": len(labels),
        "failure_prevalence": float(labels.mean()),
        "auprc": safe_metric(average_precision_score, labels, scores),
        "auroc": safe_metric(roc_auc_score, labels, scores),
        "brier": float(brier_score_loss(labels, scores)),
        "task_centered_auprc": safe_metric(average_precision_score, labels, centered),
        "task_centered_auroc": safe_metric(roc_auc_score, labels, centered),
        "within_task_pairwise_auc": (
            float(weighted_auc / comparable_pairs) if comparable_pairs else float("nan")
        ),
        "eligible_tasks": eligible_tasks,
        "comparable_success_failure_pairs": comparable_pairs,
        "task_eta_squared": float(between / total) if total else 0.0,
    }


def metric_values(frame: pd.DataFrame) -> dict[str, float | int]:
    group_codes = pd.factorize(frame["task_group"], sort=True)[0]
    return _metric_arrays(
        frame["failure"].to_numpy(int), frame["score"].to_numpy(float), group_codes,
    )


def _percentile_interval(values: list[float], confidence: float) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan"), float("nan")
    alpha = (1 - confidence) / 2
    return float(np.quantile(finite, alpha)), float(np.quantile(finite, 1 - alpha))


def bootstrap_metric_intervals(
    frame: pd.DataFrame,
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> dict[str, float]:
    """Return deterministic episode- and task-cluster percentile intervals."""
    if samples < 1:
        raise ValueError("samples must be at least one")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    rng = np.random.default_rng(seed)
    episode_draws = {metric: [] for metric in BOOTSTRAP_METRICS}
    task_draws = {metric: [] for metric in BOOTSTRAP_METRICS}
    labels = frame["failure"].to_numpy(int)
    scores = frame["score"].to_numpy(float)
    group_codes = pd.factorize(frame["task_group"], sort=True)[0]
    task_indices = [np.flatnonzero(group_codes == code) for code in np.unique(group_codes)]

    for _ in range(samples):
        indices = rng.integers(0, len(frame), len(frame))
        values = _metric_arrays(labels[indices], scores[indices], group_codes[indices])
        for metric in BOOTSTRAP_METRICS:
            episode_draws[metric].append(float(values[metric]))

        sampled_tasks = rng.integers(0, len(task_indices), len(task_indices))
        cluster_indices = np.concatenate([task_indices[task] for task in sampled_tasks])
        cluster_groups = np.concatenate(
            [np.full(len(task_indices[task]), occurrence) for occurrence, task in enumerate(sampled_tasks)]
        )
        values = _metric_arrays(labels[cluster_indices], scores[cluster_indices], cluster_groups)
        for metric in BOOTSTRAP_METRICS:
            task_draws[metric].append(float(values[metric]))

    result: dict[str, float] = {}
    for metric in BOOTSTRAP_METRICS:
        low, high = _percentile_interval(episode_draws[metric], confidence)
        result[f"{metric}_episode_ci_low"] = low
        result[f"{metric}_episode_ci_high"] = high
        low, high = _percentile_interval(task_draws[metric], confidence)
        result[f"{metric}_task_cluster_ci_low"] = low
        result[f"{metric}_task_cluster_ci_high"] = high
    return result


def aggregate(predictions: pd.DataFrame, horizon: int) -> pd.DataFrame:
    frame = predictions.loc[predictions["env_step"] <= horizon]
    columns = ["feature_set", "episode_id", "task_group", "failure"]
    if "suite" in frame:
        columns.append("suite")
    return frame.groupby(columns, as_index=False).agg(score=("failure_probability", "mean"))


def episode_metadata(episodes: pd.DataFrame) -> pd.DataFrame:
    metadata = episodes[["episode_id", "suite", "task_id", "total_steps", "success", "condition"]].copy()
    metadata["task_group"] = metadata["suite"].astype(str) + ":" + metadata["task_id"].astype(str)
    metadata["failure"] = (~metadata["success"].astype(bool)).astype(int)
    return metadata


def dataset_feature_sets(dataset: HiddenStateDataset) -> dict[str, np.ndarray]:
    vlm_final = dataset.activations["vlm_layer_15"]
    expert_final = dataset.activations["action_expert_layer_15"]
    history = recent_state_action_history_features(
        dataset.metadata, dataset.state_features, dataset.policy_features, window=5,
    )
    return {
        "robot_state": dataset.state_features,
        "policy_output": dataset.policy_features,
        "action_uncertainty": dataset.uncertainty_features,
        "state_action_history": history,
        "vlm_final": vlm_final,
        "action_expert_final": expert_final,
    }


def audit_existing_predictions(
    predictions: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "suite" not in predictions:
        predictions = predictions.merge(
            metadata[["episode_id", "suite"]], on="episode_id", validate="many_to_one",
        )
    rows: list[dict[str, object]] = []
    score_tables: dict[int, pd.DataFrame] = {}
    offset = 0
    for horizon in HORIZONS:
        scored = aggregate(predictions, horizon)
        score_tables[horizon] = scored
        for feature, frame in scored.groupby("feature_set"):
            row: dict[str, object] = {"feature_set": feature, "horizon": horizon}
            row.update(metric_values(frame))
            if feature in PRIMARY_FEATURES:
                print(f"Bootstrap: feature={feature} horizon={horizon} samples={bootstrap_samples}")
                row.update(bootstrap_metric_intervals(
                    frame, samples=bootstrap_samples, confidence=confidence, seed=seed + offset,
                ))
            rows.append(row)
            offset += 1

    metrics = pd.DataFrame(rows).sort_values(["horizon", "auprc"], ascending=[True, False])
    suite_rows: list[dict[str, object]] = []
    for horizon, scored in score_tables.items():
        for (feature, suite), frame in scored.groupby(["feature_set", "suite"]):
            row = {"feature_set": feature, "horizon": horizon, "suite": suite}
            row.update(metric_values(frame))
            suite_rows.append(row)
    suite_metrics = pd.DataFrame(suite_rows).sort_values(["horizon", "feature_set", "suite"])

    step0 = score_tables[0][["feature_set", "episode_id", "failure", "score"]].rename(
        columns={"score": "score_step_0"},
    )
    step100 = score_tables[100][["feature_set", "episode_id", "score"]].rename(
        columns={"score": "score_through_100"},
    )
    growth = step0.merge(step100, on=["feature_set", "episode_id"], validate="one_to_one")
    growth["score_growth"] = growth["score_through_100"] - growth["score_step_0"]
    growth_summary = growth.groupby(["feature_set", "failure"], as_index=False).agg(
        episodes=("episode_id", "size"),
        mean_score_step_0=("score_step_0", "mean"),
        mean_score_through_100=("score_through_100", "mean"),
        mean_score_growth=("score_growth", "mean"),
        median_score_growth=("score_growth", "median"),
    )
    return metrics, suite_metrics, growth_summary


def prediction_frame(metadata: pd.DataFrame, feature_set: str, scores: np.ndarray) -> pd.DataFrame:
    columns = ["episode_id", "env_step", "suite", "task_group", "failure"]
    frame = metadata[columns].copy()
    frame["feature_set"] = feature_set
    frame["failure_probability"] = scores
    return frame


def leave_one_suite_out_analysis(
    dataset: HiddenStateDataset,
    *,
    pca_components: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    metadata = dataset.metadata
    labels = metadata["failure"].to_numpy(int)
    suites = metadata["suite"].to_numpy()
    episodes = metadata["episode_id"].to_numpy()
    feature_sets = dataset_feature_sets(dataset)
    frames = []
    predictions_by_feature = {}
    for feature, values in feature_sets.items():
        scores = leave_one_group_out_probe_predictions(
            values,
            labels,
            suites,
            episodes,
            max_components=pca_components if values.shape[1] > pca_components else None,
        )
        if not np.isfinite(scores).all():
            raise ValueError(f"Non-finite leave-one-suite-out predictions for {feature}")
        predictions_by_feature[feature] = scores
        frames.append(prediction_frame(metadata, feature, scores))
    predictions = pd.concat(frames, ignore_index=True)
    rows = []
    for horizon in HORIZONS:
        scored = aggregate(predictions, horizon)
        for (feature, suite), frame in scored.groupby(["feature_set", "suite"]):
            row = {"feature_set": feature, "horizon": horizon, "held_out_suite": suite}
            row.update(metric_values(frame))
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["horizon", "feature_set", "held_out_suite"]), predictions_by_feature


def add_history_predictions(
    predictions: pd.DataFrame,
    dataset: HiddenStateDataset,
    history_features: np.ndarray,
    *,
    pca_components: int,
) -> pd.DataFrame:
    metadata = dataset.metadata
    scores = grouped_probe_predictions(
        history_features,
        metadata["failure"].to_numpy(int),
        metadata["task_group"].to_numpy(),
        metadata["episode_id"].to_numpy(),
        max_components=pca_components if history_features.shape[1] > pca_components else None,
    )
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite task-grouped history predictions")
    history = prediction_frame(metadata, "state_action_history", scores)
    history = history.drop(columns="suite")
    return pd.concat([predictions, history], ignore_index=True)


def condition_transfer_analysis(
    datasets: dict[str, HiddenStateDataset],
    *,
    pca_components: int,
) -> pd.DataFrame:
    if "clean" not in datasets:
        raise ValueError("condition transfer requires a clean source run")
    feature_sets = {condition: dataset_feature_sets(dataset) for condition, dataset in datasets.items()}
    directions = []
    for condition in sorted(datasets):
        if condition == "clean":
            continue
        directions.extend([("clean", condition), (condition, "clean")])

    rows = []
    for source_condition, target_condition in directions:
        source = datasets[source_condition]
        target = datasets[target_condition]
        source_metadata = source.metadata
        target_metadata = target.metadata
        source_episode_ids = source_condition + ":" + source_metadata["episode_id"].astype(str).to_numpy()
        target_episode_ids = target_condition + ":" + target_metadata["episode_id"].astype(str).to_numpy()
        for feature in feature_sets[source_condition]:
            scores = task_grouped_condition_transfer_predictions(
                feature_sets[source_condition][feature],
                source_metadata["failure"].to_numpy(int),
                source_metadata["task_group"].to_numpy(),
                source_episode_ids,
                feature_sets[target_condition][feature],
                target_metadata["task_group"].to_numpy(),
                target_episode_ids,
                max_components=(
                    pca_components
                    if feature_sets[source_condition][feature].shape[1] > pca_components
                    else None
                ),
            )
            step_frame = prediction_frame(target_metadata, feature, scores)
            for horizon in HORIZONS:
                scored = aggregate(step_frame, horizon)
                frame = scored.loc[scored["feature_set"] == feature]
                finite = np.isfinite(frame["score"].to_numpy(float))
                row: dict[str, object] = {
                    "source_condition": source_condition,
                    "target_condition": target_condition,
                    "feature_set": feature,
                    "horizon": horizon,
                    "episodes": len(frame),
                    "prediction_coverage": float(finite.mean()),
                    "estimable": bool(finite.all()),
                }
                if finite.all():
                    row.update(metric_values(frame))
                    row["mean_failure_probability"] = float(frame["score"].mean())
                else:
                    row["failure_prevalence"] = float(frame["failure"].mean())
                    for metric in BOOTSTRAP_METRICS:
                        row[metric] = float("nan")
                    row["eligible_tasks"] = 0
                    row["comparable_success_failure_pairs"] = 0
                    row["mean_failure_probability"] = float("nan")
                rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["source_condition", "target_condition", "horizon", "feature_set"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--condition-run", type=Path, action="append", default=[])
    args = parser.parse_args()

    original_predictions = pd.read_parquet(args.run / "summaries/probe_predictions.parquet")
    episodes = pd.read_parquet(args.run / "episodes.parquet")
    metadata = episode_metadata(episodes)

    print("Loading benchmark activations for leave-one-suite-out and history baselines...")
    benchmark = load_hidden_state_dataset(args.run)
    loso_metrics, loso_predictions = leave_one_suite_out_analysis(
        benchmark, pca_components=args.pca_components,
    )
    history_features = dataset_feature_sets(benchmark)["state_action_history"]
    original_predictions = add_history_predictions(
        original_predictions, benchmark, history_features, pca_components=args.pca_components,
    )
    metrics, suite_metrics, growth_summary = audit_existing_predictions(
        original_predictions,
        metadata,
        bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence,
        seed=args.seed,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / "warning_confound_metrics.csv", index=False)
    suite_metrics.to_csv(args.output / "warning_suite_metrics.csv", index=False)
    growth_summary.to_csv(args.output / "warning_score_growth.csv", index=False)
    loso_metrics.to_csv(args.output / "warning_leave_one_suite_out.csv", index=False)

    if args.condition_run:
        del benchmark, history_features, loso_predictions
        gc.collect()
        condition_datasets: dict[str, HiddenStateDataset] = {}
        for run in args.condition_run:
            dataset = load_hidden_state_dataset(run)
            conditions = dataset.metadata["condition"].astype(str).unique()
            if len(conditions) != 1:
                raise ValueError(f"Expected one condition in {run}, found {conditions.tolist()}")
            condition = str(conditions[0])
            if condition in condition_datasets:
                raise ValueError(f"Duplicate condition run: {condition}")
            condition_datasets[condition] = dataset
        transfer = condition_transfer_analysis(
            condition_datasets, pca_components=args.pca_components,
        )
        transfer.to_csv(args.output / "warning_cross_condition_transfer.csv", index=False)

    selected = metrics.loc[
        metrics["feature_set"].isin(PRIMARY_FEATURES)
    ]
    print(selected.to_string(index=False))
    print("\nLeave-one-suite-out results")
    print(loso_metrics.to_string(index=False))
    print("\nScore growth through step 100")
    print(
        growth_summary.loc[
            growth_summary["feature_set"].isin(
                ["vlm_final", "action_expert_final", "state_action_history", "action_uncertainty"],
            )
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
