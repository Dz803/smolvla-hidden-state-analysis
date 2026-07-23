#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def safe_metric(function, labels: np.ndarray, scores: np.ndarray) -> float:
    return float(function(labels, scores)) if np.unique(labels).size == 2 else float("nan")


def task_eta_squared(frame: pd.DataFrame) -> float:
    values = frame["score"].to_numpy(float)
    grand = values.mean()
    total = np.square(values - grand).sum()
    if total == 0:
        return 0.0
    between = sum(
        len(group) * float(np.square(group["score"].mean() - grand))
        for _, group in frame.groupby("task_group")
    )
    return float(between / total)


def within_task_auc(frame: pd.DataFrame) -> tuple[float, int, int]:
    weighted = 0.0
    pairs = 0
    eligible = 0
    for _, group in frame.groupby("task_group"):
        labels = group["failure"].to_numpy(int)
        if np.unique(labels).size != 2:
            continue
        failures = int(labels.sum())
        successes = int(len(labels) - failures)
        task_pairs = failures * successes
        weighted += roc_auc_score(labels, group["score"].to_numpy(float)) * task_pairs
        pairs += task_pairs
        eligible += 1
    return (float(weighted / pairs) if pairs else float("nan"), eligible, pairs)


def aggregate(predictions: pd.DataFrame, horizon: int) -> pd.DataFrame:
    frame = predictions.loc[predictions["env_step"] <= horizon]
    return frame.groupby(
        ["feature_set", "episode_id", "task_group", "failure"], as_index=False,
    ).agg(score=("failure_probability", "mean"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions = pd.read_parquet(args.run / "summaries/probe_predictions.parquet")
    episodes = pd.read_parquet(args.run / "episodes.parquet")
    episode_meta = episodes[["episode_id", "suite", "task_id", "total_steps", "success"]].copy()
    episode_meta["task_group"] = episode_meta["suite"].astype(str) + ":" + episode_meta["task_id"].astype(str)

    rows: list[dict[str, object]] = []
    score_tables: dict[int, pd.DataFrame] = {}
    for horizon in (0, 50, 100):
        scored = aggregate(predictions, horizon).merge(
            episode_meta[["episode_id", "suite", "total_steps"]], on="episode_id", validate="many_to_one",
        )
        score_tables[horizon] = scored
        for feature, frame in scored.groupby("feature_set"):
            labels = frame["failure"].to_numpy(int)
            scores = frame["score"].to_numpy(float)
            centered = scores - frame.groupby("task_group")["score"].transform("mean").to_numpy(float)
            within_auc, eligible_tasks, comparable_pairs = within_task_auc(frame)
            rows.append({
                "feature_set": feature,
                "horizon": horizon,
                "episodes": len(frame),
                "failure_prevalence": labels.mean(),
                "auprc": safe_metric(average_precision_score, labels, scores),
                "auroc": safe_metric(roc_auc_score, labels, scores),
                "brier": brier_score_loss(labels, scores),
                "task_centered_auprc": safe_metric(average_precision_score, labels, centered),
                "task_centered_auroc": safe_metric(roc_auc_score, labels, centered),
                "within_task_pairwise_auc": within_auc,
                "eligible_tasks": eligible_tasks,
                "comparable_success_failure_pairs": comparable_pairs,
                "task_eta_squared": task_eta_squared(frame),
            })

    metrics = pd.DataFrame(rows).sort_values(["horizon", "auprc"], ascending=[True, False])
    suite_rows: list[dict[str, object]] = []
    for horizon, scored in score_tables.items():
        for (feature, suite), frame in scored.groupby(["feature_set", "suite"]):
            labels = frame["failure"].to_numpy(int)
            scores = frame["score"].to_numpy(float)
            suite_rows.append({
                "feature_set": feature,
                "horizon": horizon,
                "suite": suite,
                "episodes": len(frame),
                "failure_prevalence": labels.mean(),
                "auprc": safe_metric(average_precision_score, labels, scores),
                "auroc": safe_metric(roc_auc_score, labels, scores),
                "brier": brier_score_loss(labels, scores),
            })
    suite_metrics = pd.DataFrame(suite_rows).sort_values(["horizon", "feature_set", "suite"])

    step0 = score_tables[0][["feature_set", "episode_id", "failure", "score"]].rename(columns={"score": "score_step_0"})
    step100 = score_tables[100][["feature_set", "episode_id", "score"]].rename(columns={"score": "score_through_100"})
    growth = step0.merge(step100, on=["feature_set", "episode_id"], validate="one_to_one")
    growth["score_growth"] = growth["score_through_100"] - growth["score_step_0"]
    growth_summary = growth.groupby(["feature_set", "failure"], as_index=False).agg(
        episodes=("episode_id", "size"),
        mean_score_step_0=("score_step_0", "mean"),
        mean_score_through_100=("score_through_100", "mean"),
        mean_score_growth=("score_growth", "mean"),
        median_score_growth=("score_growth", "median"),
    )

    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / "warning_confound_metrics.csv", index=False)
    suite_metrics.to_csv(args.output / "warning_suite_metrics.csv", index=False)
    growth_summary.to_csv(args.output / "warning_score_growth.csv", index=False)

    selected = metrics.loc[metrics["feature_set"].isin(["vlm_final", "action_expert_final", "policy_output", "robot_state", "action_uncertainty"])]
    print(selected.to_string(index=False))
    print("\nScore growth through step 100")
    print(growth_summary.loc[growth_summary["feature_set"].isin(["vlm_final", "action_expert_final", "action_uncertainty"])].to_string(index=False))


if __name__ == "__main__":
    main()
