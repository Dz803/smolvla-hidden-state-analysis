#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(difference))))


def _alignment(value: np.ndarray, original: np.ndarray, alternate: np.ndarray) -> tuple[float, float, float]:
    distance_to_original = _rmse(value, original)
    distance_to_alternate = _rmse(value, alternate)
    score = (distance_to_original - distance_to_alternate) / max(
        distance_to_original + distance_to_alternate, 1e-12
    )
    return distance_to_original, distance_to_alternate, score


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace explicit language-conflict alignment through SmolVLA denoising."
    )
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--executed-dim", type=int, default=7)
    args = parser.parse_args()

    store = zarr.open_group(str(args.raw), mode="r")
    suffix = "_original_noise101"
    state_ids = sorted(
        name[: -len(suffix)]
        for name in store.group_keys()
        if name.endswith(suffix) and not name.endswith("_repeat_original_noise101")
    )
    if not state_ids:
        raise ValueError("No original noise-101 queries found")

    rows = []
    for state_id in state_ids:
        groups = {
            condition: store[f"{state_id}_{condition}_noise101"]
            for condition in ("original", "alternate_goal", "contradictory_alternate_goal")
        }
        arrays = {}
        for condition, group in groups.items():
            velocity = np.asarray(group["denoising_velocity"])
            flow_state = np.asarray(group["denoising_x_t"])
            arrays[condition] = {
                "expert_hidden": np.asarray(
                    group[f"activations/action_expert_layer_{args.layer:02d}"]
                ),
                "velocity_executed": velocity[..., : args.executed_dim],
                "velocity_padding": velocity[..., args.executed_dim :],
                "velocity_full": velocity,
                "flow_state_executed": flow_state[..., : args.executed_dim],
                "flow_state_padding": flow_state[..., args.executed_dim :],
            }
        timesteps = np.asarray(groups["original"]["denoising_timestep"]).reshape(-1)
        env_step = int(state_id.rsplit("step", 1)[1])
        for representation in arrays["original"]:
            for denoising_index, timestep in enumerate(timesteps):
                distance_to_original, distance_to_alternate, score = _alignment(
                    arrays["contradictory_alternate_goal"][representation][denoising_index],
                    arrays["original"][representation][denoising_index],
                    arrays["alternate_goal"][representation][denoising_index],
                )
                rows.append(
                    {
                        "state_id": state_id,
                        "env_step": env_step,
                        "representation": representation,
                        "denoising_index": denoising_index,
                        "timestep": float(timestep),
                        "distance_to_original": distance_to_original,
                        "distance_to_alternate": distance_to_alternate,
                        "alternate_alignment": score,
                    }
                )

    frame = pd.DataFrame(rows)
    output_csv = args.report / "contradiction_dynamics.csv"
    frame.to_csv(output_csv, index=False)
    selected = frame[frame["denoising_index"].isin([0, 4, 9])]
    summary_rows = (
        selected.groupby(["env_step", "representation", "denoising_index"])
        .median(numeric_only=True)
        .reset_index()
        .to_dict(orient="records")
    )
    summary = {
        "schema_version": 1,
        "raw_capture": str(args.raw.resolve()),
        "states": state_ids,
        "layer": args.layer,
        "executed_dimension": args.executed_dim,
        "alignment_definition": (
            "(distance_to_original - distance_to_alternate) / "
            "(distance_to_original + distance_to_alternate); positive is alternate-goal aligned"
        ),
        "selected_denoising_medians": summary_rows,
    }
    output_json = args.report / "contradiction_dynamics_summary.json"
    output_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"csv": str(output_csv), "summary": str(output_json), **summary}, indent=2))


if __name__ == "__main__":
    main()
