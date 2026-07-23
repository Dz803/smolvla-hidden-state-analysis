from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _save(fig, output_stem: Path) -> list[Path]:
    paths = []
    for suffix in (".png", ".svg"):
        path = output_stem.with_suffix(suffix)
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def generate_smoke_plots(
    episodes: pd.DataFrame, steps: pd.DataFrame, output_dir: str | Path, run_id: str,
    uncertainty: pd.DataFrame | None = None,
) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = []

    success = episodes.groupby("task_id", as_index=False)["success"].agg(["mean", "count"]).reset_index()
    fig, axis = plt.subplots(figsize=(6, 4))
    sns.barplot(data=success, x="task_id", y="mean", ax=axis)
    axis.set(ylabel="Success rate", xlabel="LIBERO task ID", ylim=(0, 1), title=f"Success by task (n={len(episodes)})")
    _save(fig, output / "success_by_task")
    manifest.append(("success_by_task", "generate_smoke_plots", "episodes.parquet", "all", "success_rate"))

    fig, axis = plt.subplots(figsize=(6, 4))
    sns.histplot(data=steps, x="policy_latency_ms", ax=axis)
    axis.set(xlabel="Policy latency (ms)", title=f"Policy latency (n={len(steps)} steps)")
    _save(fig, output / "latency_distribution")
    manifest.append(("latency_distribution", "generate_smoke_plots", "steps.parquet", "all", "policy_latency_ms"))

    fig, axis = plt.subplots(figsize=(6, 4))
    action_columns = ["normalized_progress", "action_norm"]
    if "episode_id" in steps:
        action_columns.insert(0, "episode_id")
    action_curve = steps[action_columns].copy()
    if "episode_id" not in action_curve:
        action_curve["episode_id"] = "synthetic"
    action_curve["progress_bin"] = (action_curve["normalized_progress"] * 20).round() / 20
    action_curve = action_curve.groupby(["episode_id", "progress_bin"], as_index=False)["action_norm"].mean()
    sns.lineplot(data=action_curve, x="progress_bin", y="action_norm", errorbar=("ci", 95), ax=axis)
    axis.set(xlabel="Normalized trajectory progress", ylabel="Action L2 norm", title="Action magnitude over progress")
    _save(fig, output / "action_magnitude_progress")
    manifest.append(("action_magnitude_progress", "generate_smoke_plots", "steps.parquet", "all", "action_norm"))

    if "suite" in episodes:
        fig, axis = plt.subplots(figsize=(6, 4))
        sns.barplot(data=episodes, x="suite", y="success", errorbar=("ci", 95), ax=axis)
        axis.set(ylabel="Success rate", xlabel="LIBERO suite", ylim=(0, 1), title=f"Success by suite (n={len(episodes)})")
        axis.tick_params(axis="x", rotation=20)
        _save(fig, output / "success_by_suite")
        manifest.append(("success_by_suite", "generate_smoke_plots", "episodes.parquet", "all", "success_rate"))

    if {"task_id", "condition"}.issubset(episodes):
        heatmap = episodes.pivot_table(index="task_id", columns="condition", values="success", aggfunc="mean")
        fig, axis = plt.subplots(figsize=(6, 5))
        sns.heatmap(heatmap, vmin=0, vmax=1, annot=True, fmt=".2f", cmap="viridis", ax=axis)
        axis.set(title="Task × condition success rate", xlabel="Condition", ylabel="LIBERO task ID")
        _save(fig, output / "task_condition_success_heatmap")
        manifest.append(("task_condition_success_heatmap", "generate_smoke_plots", "episodes.parquet", "all", "success_rate"))

    if "total_steps" in episodes:
        fig, axis = plt.subplots(figsize=(6, 4))
        sns.histplot(data=episodes, x="total_steps", hue="success", multiple="layer", ax=axis)
        axis.set(xlabel="Episode length (environment steps)", title=f"Episode length (n={len(episodes)})")
        _save(fig, output / "episode_length_distribution")
        manifest.append(("episode_length_distribution", "generate_smoke_plots", "episodes.parquet", "all", "total_steps"))

    if "failure_class" in episodes:
        failures = episodes.loc[~episodes["success"].astype(bool), "failure_class"].fillna("unannotated").value_counts()
        fig, axis = plt.subplots(figsize=(7, 4))
        sns.barplot(x=failures.values, y=failures.index, ax=axis)
        axis.set(xlabel="Episode count", ylabel="Failure class", title=f"Failure taxonomy (n={int((~episodes['success'].astype(bool)).sum())} failures)")
        _save(fig, output / "failure_taxonomy")
        manifest.append(("failure_taxonomy", "generate_smoke_plots", "episodes.parquet", "failed episodes", "failure_class_count"))

    if "action_jerk" in steps:
        outcome_steps = steps.merge(episodes[["episode_id", "success"]], on="episode_id", how="left")
        fig, axis = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=outcome_steps, x="success", y="action_jerk", showfliers=False, ax=axis)
        axis.set(xlabel="Episode success", ylabel="Action jerk (L2)", title=f"Action jerk by outcome (n={len(steps)} steps)")
        _save(fig, output / "action_jerk_by_outcome")
        manifest.append(("action_jerk_by_outcome", "generate_smoke_plots", "steps.parquet, episodes.parquet", "all", "action_jerk"))

    if {"wall_time_s", "peak_gpu_memory_mb", "total_steps"}.issubset(episodes):
        resources = episodes.copy()
        resources["throughput_steps_s"] = resources["total_steps"] / resources["wall_time_s"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(data=resources, x="peak_gpu_memory_mb", ax=axes[0])
        sns.histplot(data=resources, x="throughput_steps_s", ax=axes[1])
        axes[0].set(xlabel="Peak allocated GPU memory (MiB)", title="GPU memory")
        axes[1].set(xlabel="Throughput (environment steps/s)", title="Episode throughput")
        _save(fig, output / "memory_and_throughput")
        manifest.append(("memory_and_throughput", "generate_smoke_plots", "episodes.parquet", "all", "peak_gpu_memory_mb, throughput_steps_s"))

    if "gripper_action" in steps:
        examples = []
        for outcome in (True, False):
            candidates = episodes.loc[episodes["success"].astype(bool) == outcome, "episode_id"]
            if len(candidates):
                examples.append(candidates.iloc[0])
        example_steps = steps.loc[steps["episode_id"].isin(examples)]
        fig, axis = plt.subplots(figsize=(8, 4))
        sns.lineplot(data=example_steps, x="env_step", y="gripper_action", hue="episode_id", ax=axis)
        axis.set(xlabel="Environment step", ylabel="Gripper action", title="Representative gripper timelines")
        axis.legend(fontsize=7)
        _save(fig, output / "gripper_timeline_examples")
        manifest.append(("gripper_timeline_examples", "generate_smoke_plots", "steps.parquet, episodes.parquet", "one success and one failure when available", "gripper_action"))

    if uncertainty is not None and len(uncertainty):
        progress = uncertainty.merge(
            episodes[["episode_id", "total_steps", "success"]], on="episode_id", how="left"
        )
        progress["normalized_progress"] = progress["env_step"] / (progress["total_steps"] - 1).clip(lower=1)
        fig, axis = plt.subplots(figsize=(7, 4))
        sns.lineplot(
            data=progress, x="normalized_progress", y="mean_pairwise_chunk_distance",
            hue="success", errorbar=("ci", 95), ax=axis,
        )
        axis.set(xlabel="Normalized trajectory progress", ylabel="Mean pairwise chunk distance", title="Offline action uncertainty over progress")
        _save(fig, output / "uncertainty_over_progress")
        manifest.append(("uncertainty_over_progress", "generate_smoke_plots", "summaries/offline_uncertainty.parquet, episodes.parquet", "every fifth environment step", "mean_pairwise_chunk_distance"))

    timestamp = datetime.now(UTC).isoformat()
    frame = pd.DataFrame(
        [
            {
                "plot_filename": name + ".png/.svg", "generating_function": function,
                "source_run_ids": run_id, "source_tables": table, "filters": filters,
                "metrics": metric, "timestamp": timestamp,
            }
            for name, function, table, filters, metric in manifest
        ]
    )
    frame.to_csv(output / "plot_manifest.csv", index=False)
    return frame
