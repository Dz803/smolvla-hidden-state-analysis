#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from smolvla_analysis.hidden_state_analysis import (
    activation_statistics,
    hidden_pca,
    load_hidden_state_dataset,
    run_grouped_probes,
)


parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--pca-components", type=int, default=32)
parser.add_argument("--group-folds", type=int)
parser.add_argument("--reuse-probes", action="store_true")
args = parser.parse_args()
manifest = json.loads((args.run / "manifest.json").read_text())
if manifest.get("completion_status") != "complete":
    raise SystemExit("Refusing hidden-state analysis on an incomplete run")

dataset = load_hidden_state_dataset(args.run)
activation_summary, centroid_metrics, divergence = activation_statistics(dataset)
pca = hidden_pca(dataset)
if args.reuse_probes and (args.run / "summaries/probe_predictions.parquet").exists():
    import pandas as pd

    predictions = pd.read_parquet(args.run / "summaries/probe_predictions.parquet")
    probe_metrics = pd.read_parquet(args.run / "summaries/probe_metrics.parquet")
else:
    predictions, probe_metrics = run_grouped_probes(
        dataset, pca_components=args.pca_components, group_folds=args.group_folds,
    )

summary_dir = args.run / "summaries"
outputs = {
    "activation_summary.parquet": activation_summary,
    "activation_centroid_metrics.parquet": centroid_metrics,
    "representation_divergence.parquet": divergence,
    "hidden_pca.parquet": pca,
    "probe_predictions.parquet": predictions,
    "probe_metrics.parquet": probe_metrics,
}
for name, frame in outputs.items():
    temporary = summary_dir / f".{name}.tmp"
    frame.to_parquet(temporary, index=False)
    temporary.replace(summary_dir / name)
analysis_manifest = {
    "timestamp": datetime.now(UTC).isoformat(), "source_run_id": manifest["run_id"],
    "activation_samples": len(dataset.metadata),
    "probe_split": f"{args.group_folds}-fold task-grouped" if args.group_folds else "leave-one-task-out",
    "episode_balanced_training_weights": True, "pca_components_inside_probe_folds": args.pca_components,
    "target": "eventual episode failure using information available at sampled time t",
    "limitations": [
        "Failure subtype and onset are unavailable without manual annotation.",
        "Warning lead time can only be measured to episode termination, not true failure onset.",
        "Token boundaries are unknown; pooled VLM tokens are not separated by modality.",
    ],
    "outputs": {name: len(frame) for name, frame in outputs.items()},
}
(summary_dir / "hidden_state_analysis_manifest.json").write_text(json.dumps(analysis_manifest, indent=2))
print(
    probe_metrics.loc[probe_metrics["evaluation_unit"] == "full_episode"]
    .sort_values("auprc", ascending=False).to_string(index=False)
)
