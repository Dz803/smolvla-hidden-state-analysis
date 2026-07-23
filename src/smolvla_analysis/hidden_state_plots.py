from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve


def _save(fig, output: Path) -> None:
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _episode_scores(predictions: pd.DataFrame, feature: str, horizon: int | None = None) -> pd.DataFrame:
    frame = predictions.loc[predictions["feature_set"] == feature]
    if horizon is not None:
        frame = frame.loc[frame["env_step"] <= horizon]
    return frame.groupby("episode_id", as_index=False).agg(
        failure=("failure", "first"), score=("failure_probability", "mean"),
    )


def generate_hidden_state_plots(run_dir: str | Path) -> pd.DataFrame:
    root = Path(run_dir)
    output = root / "plots"
    summary_dir = root / "summaries"
    activations = pd.read_parquet(summary_dir / "activation_summary.parquet")
    pca = pd.read_parquet(summary_dir / "hidden_pca.parquet")
    metrics = pd.read_parquet(summary_dir / "probe_metrics.parquet")
    predictions = pd.read_parquet(summary_dir / "probe_predictions.parquet")
    divergence = pd.read_parquet(summary_dir / "representation_divergence.parquet")
    uncertainty = pd.read_parquet(summary_dir / "offline_uncertainty.parquet")
    episodes = pd.read_parquet(root / "episodes.parquet")
    records = []

    for pathway in ("vlm", "action_expert"):
        frame = activations.loc[activations["pathway"] == pathway]
        fig, axis = plt.subplots(figsize=(8, 5))
        sns.lineplot(
            data=frame, x="normalized_progress", y="pooled_l2", hue="success",
            style="layer_index", errorbar=("ci", 95), ax=axis,
        )
        axis.set(
            xlabel="Normalized trajectory progress", ylabel="Pooled hidden-state L2 norm",
            title=f"{pathway.replace('_', ' ').title()} activation norm (n={frame.episode_id.nunique()} episodes)",
        )
        name = f"{pathway}_activation_norm_over_progress"
        _save(fig, output / name)
        records.append((name, "activation_summary.parquet", f"pathway={pathway}", "pooled_l2"))

        pathway_pca = pca.loc[pca["pathway"] == pathway]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        sns.scatterplot(data=pathway_pca, x="pc1", y="pc2", hue="success", alpha=.45, s=12, ax=axes[0])
        sns.scatterplot(data=pathway_pca, x="pc1", y="pc2", hue="task_id", palette="tab10", alpha=.45, s=12, ax=axes[1])
        sns.scatterplot(data=pathway_pca, x="pc1", y="pc2", hue="task_phase", alpha=.45, s=12, ax=axes[2])
        axes[0].set_title("Outcome"); axes[1].set_title("Task"); axes[2].set_title("Progress-fallback phase")
        fig.suptitle(f"{pathway.replace('_', ' ').title()} final-layer PCA (n={len(pathway_pca)} samples)")
        name = f"{pathway}_pca"
        _save(fig, output / name)
        records.append((name, "hidden_pca.parquet", f"pathway={pathway}", "PCA PC1/PC2"))

    layer_metrics = metrics.loc[
        metrics["feature_set"].str.startswith("layer_")
        & metrics["evaluation_unit"].isin(["full_episode", "up_to_step_50"])
    ].copy()
    layer_metrics["target"] = layer_metrics["feature_set"].str.removeprefix("layer_")
    fig, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(data=layer_metrics, x="target", y="auprc", hue="evaluation_unit", ax=axis)
    axis.axhline(episodes.success.map(lambda value: not value).mean(), color="black", linestyle="--", label="Failure prevalence")
    axis.set(xlabel="Hidden-state target", ylabel="Episode AUPRC", ylim=(0, 1), title="Layer-wise held-out-task failure separation")
    axis.tick_params(axis="x", rotation=35)
    name = "layer_wise_failure_auprc"
    _save(fig, output / name)
    records.append((name, "probe_metrics.parquet", "layer probes; full and through step 50", "AUPRC"))

    selected = ["policy_output", "robot_state", "action_uncertainty", "vlm_final", "action_expert_final", "vlm_action_expert"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for feature in selected:
        scored = _episode_scores(predictions, feature)
        fpr, tpr, _ = roc_curve(scored.failure, scored.score)
        precision, recall, _ = precision_recall_curve(scored.failure, scored.score)
        axes[0].plot(fpr, tpr, label=feature)
        axes[1].plot(recall, precision, label=feature)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=.5)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="Held-out-task ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Held-out-task precision–recall")
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
    name = "probe_roc_precision_recall"
    _save(fig, output / name)
    records.append((name, "probe_predictions.parquet", "full-episode aggregation", "ROC and precision-recall"))

    fig, axis = plt.subplots(figsize=(7, 5))
    for feature in ("vlm_final", "action_expert_final", "action_uncertainty"):
        scored = _episode_scores(predictions, feature)
        observed, predicted = calibration_curve(scored.failure, scored.score, n_bins=8, strategy="quantile")
        axis.plot(predicted, observed, marker="o", label=feature)
    axis.plot([0, 1], [0, 1], "k--", alpha=.5)
    axis.set(xlabel="Predicted failure probability", ylabel="Observed failure rate", title="Probe calibration (100 episodes)")
    axis.legend()
    name = "probe_calibration"
    _save(fig, output / name)
    records.append((name, "probe_predictions.parquet", "full-episode aggregation", "calibration"))

    vlm = predictions.loc[predictions["feature_set"] == "vlm_final"].merge(
        episodes[["episode_id", "total_steps"]], on="episode_id", how="left"
    )
    curve_rows = []
    for threshold in np.linspace(.1, .9, 17):
        first = vlm.loc[vlm.failure_probability >= threshold].groupby("episode_id").env_step.min()
        by_episode = episodes.set_index("episode_id")[["success", "total_steps"]].join(first.rename("warning_step"))
        successful = by_episode.loc[by_episode.success.astype(bool)]
        failed = by_episode.loc[~by_episode.success.astype(bool)]
        detected = failed.warning_step.notna()
        lead = failed.loc[detected, "total_steps"] - failed.loc[detected, "warning_step"]
        curve_rows.append(
            {
                "threshold": threshold, "false_alarm_rate": float(successful.warning_step.notna().mean()),
                "failure_detection_rate": float(detected.mean()),
                "median_steps_before_termination": float(lead.median()) if len(lead) else np.nan,
            }
        )
    curve = pd.DataFrame(curve_rows)
    curve.to_parquet(summary_dir / "lead_time_false_alarm_curve.parquet", index=False)
    fig, axis = plt.subplots(figsize=(7, 5))
    points = axis.scatter(
        curve.false_alarm_rate, curve.median_steps_before_termination,
        c=curve.failure_detection_rate, cmap="viridis", s=60,
    )
    fig.colorbar(points, ax=axis, label="Failure detection rate")
    axis.set(xlabel="False-alarm episodes / successful episodes", ylabel="Median warning steps before termination", title="VLM warning lead time versus false alarms")
    name = "lead_time_vs_false_alarm"
    _save(fig, output / name)
    records.append((name, "probe_predictions.parquet, episodes.parquet", "vlm_final; thresholds 0.1–0.9", "lead time to termination and false alarms"))

    aligned = uncertainty.merge(episodes[["episode_id", "success", "total_steps"]], on="episode_id", how="left")
    aligned = aligned.loc[~aligned.success.astype(bool)].copy()
    aligned["steps_to_termination"] = aligned.env_step - aligned.total_steps
    aligned = aligned.loc[aligned.steps_to_termination >= -100]
    fig, axis = plt.subplots(figsize=(7, 5))
    sns.lineplot(data=aligned, x="steps_to_termination", y="mean_pairwise_chunk_distance", errorbar=("ci", 95), ax=axis)
    axis.set(xlabel="Environment steps relative to timeout termination", ylabel="Mean pairwise chunk distance", title="Uncertainty before failed-episode termination")
    name = "uncertainty_aligned_before_termination"
    _save(fig, output / name)
    records.append((name, "offline_uncertainty.parquet, episodes.parquet", "failed episodes; last 100 steps", "uncertainty aligned to termination"))

    available = divergence.loc[divergence.first_divergence_progress.notna()]
    fig, axis = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=available, x="pathway", y="first_divergence_progress", hue="layer_index", ax=axis)
    axis.set(xlabel="Representation/action pathway", ylabel="First divergence (normalized progress)", title="Representation and action divergence from successful references")
    name = "representation_action_divergence_time"
    _save(fig, output / name)
    records.append((name, "representation_divergence.parquet", "failed episodes with task-specific success reference", "first divergence progress"))

    examples = []
    for success in (True, False):
        candidates = episodes.loc[episodes.success.astype(bool) == success]
        row = candidates.iloc[0]
        with np.load(root / row.observation_path) as archive:
            frames = archive["camera1"]
        indices = np.linspace(0, len(frames) - 1, 4).astype(int)
        examples.append((success, frames[indices], indices))
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for row_index, (success, frames, indices) in enumerate(examples):
        for column, (frame, step) in enumerate(zip(frames, indices, strict=True)):
            axes[row_index, column].imshow(np.moveaxis(frame, 0, -1) if frame.shape[0] == 3 else frame)
            axes[row_index, column].set_title(f"{'success' if success else 'failure'}; step {step}")
            axes[row_index, column].axis("off")
    fig.suptitle("Synchronized representative trajectory frames")
    name = "case_study_frame_grid"
    _save(fig, output / name)
    records.append((name, "observations/*.npz, episodes.parquet", "one success and one failure", "camera frame grid"))

    timestamp = datetime.now(UTC).isoformat()
    new_manifest = pd.DataFrame(
        [
            {
                "plot_filename": f"{name}.png/.svg", "generating_function": "generate_hidden_state_plots",
                "source_run_ids": root.name, "source_tables": tables, "filters": filters,
                "metrics": metric, "timestamp": timestamp,
            }
            for name, tables, filters, metric in records
        ]
    )
    manifest_path = output / "plot_manifest.csv"
    if manifest_path.exists():
        old = pd.read_csv(manifest_path)
        old = old.loc[~old.plot_filename.isin(new_manifest.plot_filename)]
        new_manifest = pd.concat([old, new_manifest], ignore_index=True)
    new_manifest.to_csv(manifest_path, index=False)
    return new_manifest
