#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from smolvla_analysis.hidden_state_analysis import (
    activation_statistics,
    hidden_pca,
    load_multi_hidden_state_dataset,
    run_grouped_probes,
)


parser = argparse.ArgumentParser()
parser.add_argument("--runs", type=Path, nargs="+", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--pca-components", type=int, default=32)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
plots = args.output / "plots"; plots.mkdir(exist_ok=True)


def save(fig, name: str) -> None:
    for suffix in (".png", ".svg"):
        fig.savefig((plots / name).with_suffix(suffix), dpi=220, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, digits: int = 3) -> str:
    def render(value) -> str:
        return f"{value:.{digits}f}" if isinstance(value, (float, np.floating)) else str(value)
    rows = [[render(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    return "\n".join([header, separator] + ["| " + " | ".join(row) + " |" for row in rows])


dataset = load_multi_hidden_state_dataset(args.runs)
activation_summary, centroid_metrics, divergence = activation_statistics(dataset)
pca = hidden_pca(dataset)
predictions, probe_metrics = run_grouped_probes(dataset, pca_components=args.pca_components)

condition_rows = []
for (feature, condition), frame in predictions.groupby(["feature_set", "condition"]):
    episode = frame.groupby("episode_id", as_index=False).agg(failure=("failure", "first"), score=("failure_probability", "mean"))
    labels = episode.failure.to_numpy(int); scores = episode.score.to_numpy(float)
    condition_rows.append({
        "feature_set": feature, "condition": condition, "episodes": len(episode),
        "failure_prevalence": float(labels.mean()),
        "auroc": float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else np.nan,
        "auprc": float(average_precision_score(labels, scores)) if np.unique(labels).size == 2 else np.nan,
        "brier": float(brier_score_loss(labels, scores)),
    })
condition_metrics = pd.DataFrame(condition_rows)

tables = {
    "activation_summary": activation_summary, "activation_centroid_metrics": centroid_metrics,
    "representation_divergence": divergence, "hidden_pca": pca,
    "probe_predictions": predictions, "probe_metrics": probe_metrics,
    "condition_probe_metrics": condition_metrics,
}
for name, frame in tables.items():
    frame.to_parquet(args.output / f"{name}.parquet", index=False)

sns.set_theme(style="whitegrid")
for pathway in ("vlm", "action_expert"):
    frame = pca.loc[pca.pathway == pathway]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(data=frame, x="pc1", y="pc2", hue="condition", alpha=.35, s=10, ax=axes[0])
    sns.scatterplot(data=frame, x="pc1", y="pc2", hue="success", alpha=.35, s=10, ax=axes[1])
    axes[0].set_title("Condition"); axes[1].set_title("Outcome")
    fig.suptitle(f"{pathway.replace('_', ' ').title()} final-layer PCA (n={len(frame)} samples)")
    save(fig, f"{pathway}_pca_by_condition")

    norms = activation_summary.loc[(activation_summary.pathway == pathway) & (activation_summary.layer_index == 15)]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=norms, x="normalized_progress", y="pooled_l2", hue="condition", style="success", errorbar=("ci", 95), ax=ax)
    ax.set(xlabel="Normalized trajectory progress", ylabel="Pooled hidden-state L2 norm", title=f"{pathway.replace('_', ' ').title()} final-layer norm")
    save(fig, f"{pathway}_norm_by_condition")

available = divergence.loc[divergence.first_divergence_progress.notna()]
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=available, x="condition", y="first_divergence_progress", hue="pathway", showfliers=False, ax=ax)
ax.set(xlabel="Condition", ylabel="First divergence (normalized progress)", title="Failure-trajectory divergence by condition")
ax.tick_params(axis="x", rotation=20); save(fig, "divergence_by_condition")

run_ids = ";".join(path.name for path in args.runs)
pd.DataFrame([
    {"plot_filename": f"{pathway}_pca_by_condition.png/.svg", "generating_function": "analyze_condition_hidden_states", "source_run_ids": run_ids, "source_tables": "hidden_pca.parquet", "filters": f"pathway={pathway}", "metrics": "PCA PC1/PC2 by condition and outcome", "timestamp": datetime.now(UTC).isoformat()}
    for pathway in ("vlm", "action_expert")
] + [
    {"plot_filename": f"{pathway}_norm_by_condition.png/.svg", "generating_function": "analyze_condition_hidden_states", "source_run_ids": run_ids, "source_tables": "activation_summary.parquet", "filters": f"pathway={pathway}; layer=15", "metrics": "pooled L2 norm over normalized progress", "timestamp": datetime.now(UTC).isoformat()}
    for pathway in ("vlm", "action_expert")
] + [
    {"plot_filename": "divergence_by_condition.png/.svg", "generating_function": "analyze_condition_hidden_states", "source_run_ids": run_ids, "source_tables": "representation_divergence.parquet", "filters": "failed episodes with successful task/progress reference", "metrics": "first divergence progress", "timestamp": datetime.now(UTC).isoformat()}
]).to_csv(plots / "plot_manifest.csv", index=False)

manifest = {
    "timestamp": datetime.now(UTC).isoformat(), "runs": [path.name for path in args.runs],
    "activation_samples": len(dataset.metadata), "episodes": int(dataset.metadata.episode_id.nunique()),
    "probe_split": "leave-one-suite/task-out with episode-balanced training weights and fold-local PCA",
    "limitations": [
        "Condition comparisons are associations under deterministic interventions; probe scores are diagnostic, not policy training.",
        "Task phases are normalized-progress fallbacks, not observed physical grasp/place transitions.",
        "Failure subtype and onset require manual video annotation.",
    ],
    "outputs": {name: len(frame) for name, frame in tables.items()},
}
(args.output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2))
(args.output / "condition_hidden_state_report.md").write_text("\n".join([
    "# Condition-aware hidden-state analysis", "", f"Analyzed {manifest['episodes']} episodes and {manifest['activation_samples']} sampled time points across four paired conditions.", "",
    "## Full-trajectory grouped probe metrics", "", markdown_table(probe_metrics.loc[probe_metrics.evaluation_unit == "full_episode"].sort_values("auprc", ascending=False)), "",
    "## Metrics by condition", "", markdown_table(condition_metrics.loc[condition_metrics.feature_set.isin(["majority_baseline", "vlm_final", "action_expert_final", "action_uncertainty"])]), "",
    "Full-trajectory probe scores are retrospective separation measures, not early-warning claims. Early horizons must be read from `probe_metrics.parquet`. Failure subtype/onset results are not reported without manual annotations.",
]))
print(json.dumps(manifest, indent=2))
