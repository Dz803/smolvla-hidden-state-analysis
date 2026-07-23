from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


PAIR_KEYS = ["suite", "task_id", "seed"]


@dataclass(frozen=True)
class PairedStudy:
    episodes: pd.DataFrame
    paired_outcomes: pd.DataFrame
    condition_effects: pd.DataFrame
    episode_metrics: pd.DataFrame
    step_metrics: pd.DataFrame


def _bootstrap_mean_ci(values: np.ndarray, *, seed: int = 20260722, draws: int = 10_000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def analyze_paired_study(run_dirs: list[Path]) -> PairedStudy:
    episodes = pd.concat([pd.read_parquet(path / "episodes.parquet") for path in run_dirs], ignore_index=True)
    if episodes["infrastructure_failure"].any():
        raise ValueError("Infrastructure failures are present; paired policy comparisons would be confounded")
    conditions = sorted(episodes["condition"].unique())
    if "clean" not in conditions:
        raise ValueError("A clean reference condition is required")
    expected = episodes.loc[episodes.condition == "clean", PAIR_KEYS + ["initial_state_id"]].copy()
    if expected.duplicated(PAIR_KEYS).any():
        raise ValueError("Clean reference contains duplicate paired keys")
    for condition in conditions:
        current = episodes.loc[episodes.condition == condition, PAIR_KEYS + ["initial_state_id"]]
        merged = expected.merge(current, on=PAIR_KEYS, how="outer", suffixes=("_clean", "_condition"), indicator=True)
        if not (merged["_merge"] == "both").all() or not (
            merged["initial_state_id_clean"] == merged["initial_state_id_condition"]
        ).all():
            raise ValueError(f"Condition {condition!r} is not exactly paired to clean")

    wide = episodes.pivot(index=PAIR_KEYS, columns="condition", values="success").reset_index()
    effects = []
    for offset, condition in enumerate(c for c in conditions if c != "clean"):
        delta = wide[condition].astype(float).to_numpy() - wide["clean"].astype(float).to_numpy()
        ci_low, ci_high = _bootstrap_mean_ci(delta, seed=20260722 + offset)
        clean_only = int(((wide.clean == True) & (wide[condition] == False)).sum())  # noqa: E712
        condition_only = int(((wide.clean == False) & (wide[condition] == True)).sum())  # noqa: E712
        discordant = clean_only + condition_only
        p_value = float(binomtest(min(clean_only, condition_only), discordant, 0.5).pvalue) if discordant else 1.0
        effects.append(
            {
                "condition": condition,
                "n_pairs": len(wide),
                "clean_success_rate": float(wide.clean.mean()),
                "condition_success_rate": float(wide[condition].mean()),
                "paired_success_delta": float(delta.mean()),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "clean_only_successes": clean_only,
                "condition_only_successes": condition_only,
                "mcnemar_exact_p": p_value,
            }
        )

    episode_metrics = (
        episodes.groupby(["condition", "suite", "task_id"], as_index=False)
        .agg(episodes=("episode_id", "size"), successes=("success", "sum"), success_rate=("success", "mean"),
             mean_steps=("total_steps", "mean"), median_steps=("total_steps", "median"))
    )
    step_frames = []
    for path in run_dirs:
        condition = pd.read_parquet(path / "episodes.parquet", columns=["condition"])["condition"].iloc[0]
        frame = pd.read_parquet(path / "steps.parquet", columns=["action_norm", "action_smoothness", "action_jerk"])
        frame["condition"] = condition
        step_frames.append(frame)
    step_metrics = (
        pd.concat(step_frames, ignore_index=True).groupby("condition", as_index=False)
        .agg(steps=("action_norm", "size"), action_norm_mean=("action_norm", "mean"),
             action_smoothness_mean=("action_smoothness", "mean"), action_jerk_mean=("action_jerk", "mean"))
    )
    return PairedStudy(episodes, wide, pd.DataFrame(effects), episode_metrics, step_metrics)
