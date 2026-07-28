#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr

from smolvla_analysis.config import load_config
from smolvla_analysis.libero_observation import orient_archived_camera_for_policy
from smolvla_analysis.model_inspection import resolve_pathways
from smolvla_analysis.phase2_capture import capture_action_query, fixed_flow_noise
from smolvla_analysis.phase2_storage import write_action_query
from smolvla_analysis.runtime import load_runtime


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = PROJECT / "archive/full_experiment/runs/benchmark_400_20260723T021424Z_e0638bea"
STATE_SPECS = (
    ("libero_goal_task03_ep000_seed0", 50),
    ("libero_goal_task03_ep000_seed0", 100),
    ("libero_goal_task03_ep001_seed1", 50),
    ("libero_goal_task03_ep001_seed1", 100),
)
NOISE_SEEDS = (101, 202, 303, 404)
PARAPHRASE = "Open the upper drawer, then place the bowl inside it."
CONTRADICTORY_ALTERNATE_GOAL = (
    "Do not open the top drawer or put the bowl inside it. "
    "Instead, put the bowl on top of the cabinet."
)


def _rmse(left, right) -> float:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(difference))))


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _within_group_correlation(frame: pd.DataFrame, left: str, right: str, group: str) -> float | None:
    centered = frame[[group, left, right]].copy()
    centered[[left, right]] -= centered.groupby(group)[[left, right]].transform("mean")
    return _correlation(centered[left].tolist(), centered[right].tolist())


def _row_basis(weight: np.ndarray) -> np.ndarray:
    _, singular_values, right_vectors = np.linalg.svd(weight.astype(np.float64), full_matrices=False)
    tolerance = np.finfo(np.float64).eps * max(weight.shape) * singular_values.max(initial=0.0)
    return right_vectors[singular_values > tolerance]


def _row_energy_fraction(delta: np.ndarray, basis: np.ndarray) -> float:
    flattened = np.asarray(delta, dtype=np.float64).reshape(-1, delta.shape[-1])
    total = float(np.square(flattened).sum())
    if total == 0:
        return 0.0
    projected = flattened @ basis.T
    return float(np.square(projected).sum() / total)


def _linear_output_rmse(delta: np.ndarray, weight: np.ndarray) -> float:
    output = np.asarray(delta, dtype=np.float64) @ np.asarray(weight, dtype=np.float64).T
    return float(np.sqrt(np.mean(np.square(output))))


def _as_float_chunk(value) -> np.ndarray:
    if isinstance(value, np.ndarray) and value.dtype == object:
        value = value.tolist()
    return np.asarray(value, dtype=np.float32)


def _observation(npz, step: int, instruction: str, condition: str, alternate_goal: str) -> dict:
    camera1 = torch.from_numpy(
        orient_archived_camera_for_policy(npz["camera1"][step]).astype(np.float32) / 255.0
    )
    camera2 = torch.from_numpy(
        orient_archived_camera_for_policy(npz["camera2"][step]).astype(np.float32) / 255.0
    )
    task = instruction
    if condition == "paraphrase":
        task = PARAPHRASE
    elif condition == "alternate_goal":
        task = alternate_goal
    elif condition == "contradictory_alternate_goal":
        task = CONTRADICTORY_ALTERNATE_GOAL
    elif condition == "main_mean":
        camera1 = camera1.mean(dim=(1, 2), keepdim=True).expand_as(camera1).clone()
    elif condition == "wrist_mean":
        camera2 = camera2.mean(dim=(1, 2), keepdim=True).expand_as(camera2).clone()
    elif condition not in {"original", "repeat_original"}:
        raise ValueError(f"Unknown condition: {condition}")
    return {
        "observation.state": torch.from_numpy(npz["policy_state"][step].astype(np.float32)),
        "observation.images.image": camera1,
        "observation.images.image2": camera2,
        "task": task,
    }


def _postprocess_chunk(chunk: np.ndarray, device: str, postprocessor) -> np.ndarray:
    result = []
    for action_index in range(chunk.shape[1]):
        action = torch.as_tensor(chunk[:, action_index], dtype=torch.float32, device=device)
        result.append(postprocessor(action).detach().cpu().numpy())
    return np.stack(result, axis=1)


def _masked_prefix_rmse(left, right, values_left, values_right, span=None) -> float:
    mask = left.prefix_pad_mask & right.prefix_pad_mask
    if span is not None:
        span_mask = np.zeros_like(mask)
        span_mask[:, span.start : span.stop] = True
        mask &= span_mask
    if not mask.any():
        return 0.0
    expanded = mask[..., None]
    difference = np.asarray(values_left, dtype=np.float32) - np.asarray(values_right, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(difference[expanded.repeat(difference.shape[-1], axis=-1)]))))


def _vlm_rmse(left, right, layer: int, span=None) -> float:
    left_values = left.activation_stack("vlm", layer)[0]
    right_values = right.activation_stack("vlm", layer)[0]
    return _masked_prefix_rmse(left, right, left_values, right_values, span)


def _action_expert_rmse(left, right, layer: int, invocation: int = -1) -> float:
    return _rmse(
        left.activation_stack("action_expert", layer)[invocation],
        right.activation_stack("action_expert", layer)[invocation],
    )


def _kv_norm_rmse(left, right) -> float:
    left_values = np.stack([record.key_token_l2 for record in left.kv_cache_norms])
    right_values = np.stack([record.key_token_l2 for record in right.kv_cache_norms])
    return _rmse(left_values, right_values)


def _safe_json(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/base.yaml")
    parser.add_argument("--source-run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--local-output", type=Path, default=PROJECT / "local/phase2_forward_gate")
    parser.add_argument("--report-output", type=Path, default=PROJECT / "reports/phase2_forward_gate")
    parser.add_argument("--max-states", type=int, default=len(STATE_SPECS))
    args = parser.parse_args()

    config = load_config(args.config)
    if config["model"]["device"] != "cuda":
        raise ValueError("The canonical Phase 2 gate requires the checkpoint's CUDA inference path")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"phase2_forward_gate_{timestamp}"
    raw_dir = args.local_output / run_id
    report_dir = args.report_output / run_id
    raw_dir.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True, exist_ok=False)

    episodes = pd.read_parquet(args.source_run / "episodes.parquet").set_index("episode_id")
    steps = pd.read_parquet(args.source_run / "steps.parquet")
    alternate_goal = episodes.loc["libero_goal_task04_ep000_seed0", "instruction"]
    source_chunks = {
        (row.episode_id, int(row.env_step)): _as_float_chunk(row.predicted_action_chunk)
        for row in steps.loc[
            steps["episode_id"].isin([episode for episode, _ in STATE_SPECS])
            & steps["env_step"].isin([50, 100])
        ].itertuples()
    }

    policy_cfg, policy, preprocessor, postprocessor, model_load_time = load_runtime(config, PROJECT)
    targets = resolve_pathways(policy, config["activations"]["relative_layer_positions"])
    layers = sorted({int(target["layer_index"]) for target in targets})
    action_head_weight = policy.model.action_out_proj.weight.detach().float().cpu().numpy()
    active_action_dim = int(policy.config.action_feature.shape[0])
    active_head_weight = action_head_weight[:active_action_dim]
    padding_head_weight = action_head_weight[active_action_dim:]
    active_row_basis = _row_basis(active_head_weight)
    padding_row_basis = _row_basis(padding_head_weight)
    full_row_basis = _row_basis(action_head_weight)
    raw_store = zarr.open_group(str(raw_dir / "queries.zarr"), mode="w")
    queries = {}
    query_rows = []
    fidelity = None
    conditions = (
        "original",
        "repeat_original",
        "paraphrase",
        "alternate_goal",
        "contradictory_alternate_goal",
        "main_mean",
        "wrist_mean",
    )

    for episode_id, step in STATE_SPECS[: args.max_states]:
        episode = episodes.loc[episode_id]
        npz = np.load(args.source_run / episode.observation_path)
        if step >= len(npz["policy_state"]):
            raise ValueError(f"State {episode_id}:{step} is outside the saved trajectory")
        state_id = f"{episode_id}_step{step:04d}"
        plan = [("original", seed) for seed in NOISE_SEEDS]
        plan += [(condition, NOISE_SEEDS[0]) for condition in conditions if condition != "original"]
        for condition, noise_seed in plan:
            observation = _observation(npz, step, episode.instruction, condition, alternate_goal)
            batch = preprocessor(deepcopy(observation))
            policy.reset()
            query_id = f"{state_id}_{condition}_noise{noise_seed}"
            if fidelity is None and condition == "original" and noise_seed == NOISE_SEEDS[0]:
                reference_noise = fixed_flow_noise(policy, batch, noise_seed)
                with torch.inference_mode():
                    reference_chunk = policy.predict_action_chunk(deepcopy(batch), noise=reference_noise)
                policy.reset()
            _, query = capture_action_query(
                policy,
                batch,
                targets,
                query_id=query_id,
                flow_noise_seed=noise_seed,
            )
            if fidelity is None and condition == "original" and noise_seed == NOISE_SEEDS[0]:
                fidelity = {
                    "max_abs_chunk_difference": float(
                        np.max(np.abs(reference_chunk.detach().cpu().numpy() - query.model_action_chunk))
                    ),
                    "reference_shape": list(reference_chunk.shape),
                }
            environment_chunk = _postprocess_chunk(query.model_action_chunk, policy_cfg.device, postprocessor)
            write_action_query(raw_store, query, environment_chunk)
            queries[(state_id, condition, noise_seed)] = (query, environment_chunk)
            archived = source_chunks[(episode_id, step)]
            query_rows.append(
                {
                    "state_id": state_id,
                    "episode_id": episode_id,
                    "env_step": step,
                    "rollout_success": bool(episode.success),
                    "condition": condition,
                    "noise_seed": noise_seed,
                    "action_chunk_rms": float(np.sqrt(np.mean(np.square(query.model_action_chunk)))),
                    "archive_chunk_rmse": _rmse(environment_chunk[0], archived),
                    "flow_noise_sha256": query.flow_noise_sha256,
                }
            )

    noise_rows = []
    for state_id in sorted({key[0] for key in queries}):
        for left_seed, right_seed in combinations(NOISE_SEEDS, 2):
            left, _ = queries[(state_id, "original", left_seed)]
            right, _ = queries[(state_id, "original", right_seed)]
            head_delta = left.action_head_inputs[-1].astype(np.float32) - right.action_head_inputs[-1]
            noise_rows.append(
                {
                    "state_id": state_id,
                    "env_step": int(state_id.rsplit("step", 1)[1]),
                    "left_seed": left_seed,
                    "right_seed": right_seed,
                    "action_rmse": _rmse(left.model_action_chunk, right.model_action_chunk),
                    "vlm_last_layer_rmse": _vlm_rmse(left, right, layers[-1]),
                    "action_expert_last_layer_rmse": _action_expert_rmse(left, right, layers[-1]),
                    "kv_key_norm_rmse": _kv_norm_rmse(left, right),
                    "action_head_input_rmse": _rmse(left.action_head_inputs[-1], right.action_head_inputs[-1]),
                    "active_head_output_rmse": _linear_output_rmse(head_delta, active_head_weight),
                    "padding_head_output_rmse": _linear_output_rmse(head_delta, padding_head_weight),
                    "active_row_energy_fraction": _row_energy_fraction(head_delta, active_row_basis),
                    "padding_row_energy_fraction": _row_energy_fraction(head_delta, padding_row_basis),
                    "full_output_row_energy_fraction": _row_energy_fraction(head_delta, full_row_basis),
                    "output_null_energy_fraction": 1.0 - _row_energy_fraction(head_delta, full_row_basis),
                }
            )

    factor_rows = []
    modality_rows = []
    denoising_rows = []
    repeat_rows = []
    for state_id in sorted({key[0] for key in queries}):
        original, _ = queries[(state_id, "original", NOISE_SEEDS[0])]
        repeat, _ = queries[(state_id, "repeat_original", NOISE_SEEDS[0])]
        repeat_rows.append(
            {
                "state_id": state_id,
                "action_max_abs": float(np.max(np.abs(original.model_action_chunk - repeat.model_action_chunk))),
                "vlm_max_abs": float(
                    np.max(
                        np.abs(
                            original.activation_stack("vlm", layers[-1])
                            - repeat.activation_stack("vlm", layers[-1])
                        )
                    )
                ),
                "action_expert_max_abs": float(
                    np.max(
                        np.abs(
                            original.activation_stack("action_expert", layers[-1])
                            - repeat.activation_stack("action_expert", layers[-1])
                        )
                    )
                ),
            }
        )
        for condition in (
            "paraphrase",
            "alternate_goal",
            "contradictory_alternate_goal",
            "main_mean",
            "wrist_mean",
        ):
            counterfactual, _ = queries[(state_id, condition, NOISE_SEEDS[0])]
            head_delta = (
                counterfactual.action_head_inputs[-1].astype(np.float32)
                - original.action_head_inputs[-1].astype(np.float32)
            )
            vlm_delta = _vlm_rmse(original, counterfactual, layers[-1])
            expert_delta = _action_expert_rmse(original, counterfactual, layers[-1])
            action_delta = _rmse(original.model_action_chunk, counterfactual.model_action_chunk)
            factor_rows.append(
                {
                    "state_id": state_id,
                    "env_step": int(state_id.rsplit("step", 1)[1]),
                    "condition": condition,
                    "action_rmse": action_delta,
                    "action_first10_rmse": _rmse(
                        original.model_action_chunk[:, :10], counterfactual.model_action_chunk[:, :10]
                    ),
                    "prefix_embedding_rmse": _masked_prefix_rmse(
                        original,
                        counterfactual,
                        original.prefix_embeddings,
                        counterfactual.prefix_embeddings,
                    ),
                    "vlm_last_layer_rmse": vlm_delta,
                    "action_expert_last_layer_rmse": expert_delta,
                    "action_to_vlm_gain": action_delta / max(vlm_delta, 1e-12),
                    "action_expert_to_vlm_gain": expert_delta / max(vlm_delta, 1e-12),
                    "velocity_mean_rmse": _rmse(
                        np.stack([record.velocity for record in original.denoising]),
                        np.stack([record.velocity for record in counterfactual.denoising]),
                    ),
                    "kv_key_norm_rmse": _kv_norm_rmse(original, counterfactual),
                    "action_head_input_rmse": _rmse(
                        original.action_head_inputs[-1], counterfactual.action_head_inputs[-1]
                    ),
                    "active_head_output_rmse": _linear_output_rmse(head_delta, active_head_weight),
                    "padding_head_output_rmse": _linear_output_rmse(head_delta, padding_head_weight),
                    "active_row_energy_fraction": _row_energy_fraction(head_delta, active_row_basis),
                    "padding_row_energy_fraction": _row_energy_fraction(head_delta, padding_row_basis),
                    "full_output_row_energy_fraction": _row_energy_fraction(head_delta, full_row_basis),
                    "output_null_energy_fraction": 1.0 - _row_energy_fraction(head_delta, full_row_basis),
                }
            )
            span_map = {span.name: span for span in original.token_spans}
            for layer in layers:
                for span_name, span in span_map.items():
                    if span_name == "padding":
                        continue
                    modality_rows.append(
                        {
                            "state_id": state_id,
                            "condition": condition,
                            "layer": layer,
                            "span": span_name,
                            "vlm_rmse": _vlm_rmse(original, counterfactual, layer, span),
                        }
                    )
                for denoising_index, denoising_record in enumerate(original.denoising):
                    denoising_rows.append(
                        {
                            "state_id": state_id,
                            "condition": condition,
                            "layer": layer,
                            "denoising_index": denoising_index,
                            "timestep": float(denoising_record.timestep[0]),
                            "action_expert_rmse": _action_expert_rmse(
                                original, counterfactual, layer, denoising_index
                            ),
                            "velocity_rmse": _rmse(
                                denoising_record.velocity,
                                counterfactual.denoising[denoising_index].velocity,
                            ),
                        }
                    )

    contradiction_rows = []
    for state_id in sorted({key[0] for key in queries}):
        original, _ = queries[(state_id, "original", NOISE_SEEDS[0])]
        alternate, _ = queries[(state_id, "alternate_goal", NOISE_SEEDS[0])]
        contradictory, _ = queries[
            (state_id, "contradictory_alternate_goal", NOISE_SEEDS[0])
        ]
        distances = (
            (
                "action_chunk",
                _rmse(contradictory.model_action_chunk, original.model_action_chunk),
                _rmse(contradictory.model_action_chunk, alternate.model_action_chunk),
            ),
            (
                "vlm_last_layer",
                _vlm_rmse(contradictory, original, layers[-1]),
                _vlm_rmse(contradictory, alternate, layers[-1]),
            ),
            (
                "action_expert_last_layer",
                _action_expert_rmse(contradictory, original, layers[-1]),
                _action_expert_rmse(contradictory, alternate, layers[-1]),
            ),
        )
        for representation, distance_to_original, distance_to_alternate in distances:
            contradiction_rows.append(
                {
                    "state_id": state_id,
                    "env_step": int(state_id.rsplit("step", 1)[1]),
                    "representation": representation,
                    "distance_to_original": distance_to_original,
                    "distance_to_alternate": distance_to_alternate,
                    "alternate_alignment": (
                        (distance_to_original - distance_to_alternate)
                        / max(distance_to_original + distance_to_alternate, 1e-12)
                    ),
                }
            )

    query_frame = pd.DataFrame(query_rows)
    noise_frame = pd.DataFrame(noise_rows)
    factor_frame = pd.DataFrame(factor_rows)
    modality_frame = pd.DataFrame(modality_rows)
    denoising_frame = pd.DataFrame(denoising_rows)
    repeat_frame = pd.DataFrame(repeat_rows)
    contradiction_frame = pd.DataFrame(contradiction_rows)
    for name, frame in (
        ("queries", query_frame),
        ("noise_pair_coupling", noise_frame),
        ("factor_effects", factor_frame),
        ("modality_effects", modality_frame),
        ("denoising_effects", denoising_frame),
        ("determinism", repeat_frame),
        ("contradiction_alignment", contradiction_frame),
    ):
        frame.to_csv(report_dir / f"{name}.csv", index=False)

    factor_medians = (
        factor_frame.groupby("condition")
        .median(numeric_only=True)
        .reset_index()
        .to_dict(orient="records")
    )
    summary = {
        "run_id": run_id,
        "source_run": str(args.source_run.resolve()),
        "raw_capture_path": str((raw_dir / "queries.zarr").resolve()),
        "model_load_time_s": model_load_time,
        "states": sorted({key[0] for key in queries}),
        "queries": len(queries),
        "layers": layers,
        "hook_fidelity": fidelity,
        "determinism_max": {
            column: float(repeat_frame[column].max())
            for column in ("action_max_abs", "vlm_max_abs", "action_expert_max_abs")
        },
        "noise_plan_coupling": {
            "pooled_action_to_action_expert_correlation": _correlation(
                noise_frame["action_rmse"].tolist(),
                noise_frame["action_expert_last_layer_rmse"].tolist(),
            ),
            "within_state_action_to_action_expert_correlation": _within_group_correlation(
                noise_frame, "action_rmse", "action_expert_last_layer_rmse", "state_id"
            ),
            "within_state_action_to_active_head_correlation": _within_group_correlation(
                noise_frame, "action_rmse", "active_head_output_rmse", "state_id"
            ),
            "maximum_vlm_rmse": float(noise_frame["vlm_last_layer_rmse"].max()),
            "maximum_kv_key_norm_rmse": float(noise_frame["kv_key_norm_rmse"].max()),
        },
        "executable_subspace": {
            "hidden_dimension": int(action_head_weight.shape[1]),
            "internal_action_dimension": int(action_head_weight.shape[0]),
            "executed_action_dimension": active_action_dim,
            "active_row_rank": int(active_row_basis.shape[0]),
            "padding_row_rank": int(padding_row_basis.shape[0]),
            "full_output_row_rank": int(full_row_basis.shape[0]),
            "isotropic_active_row_fraction": float(active_row_basis.shape[0] / action_head_weight.shape[1]),
            "isotropic_full_output_row_fraction": float(full_row_basis.shape[0] / action_head_weight.shape[1]),
            "median_noise_active_row_energy_fraction": float(
                noise_frame["active_row_energy_fraction"].median()
            ),
            "median_noise_padding_row_energy_fraction": float(
                noise_frame["padding_row_energy_fraction"].median()
            ),
            "median_noise_full_output_row_energy_fraction": float(
                noise_frame["full_output_row_energy_fraction"].median()
            ),
            "median_noise_output_null_energy_fraction": float(
                noise_frame["output_null_energy_fraction"].median()
            ),
        },
        "factor_medians": factor_medians,
        "contradiction_alignment_medians": (
            contradiction_frame.groupby("representation")
            .median(numeric_only=True)
            .reset_index()
            .to_dict(orient="records")
        ),
        "scientific_boundary": (
            "Fixed-observation evidence only: estimates proposal and input-factor sensitivity, not branched "
            "recoverability, causal mediation, or training-trajectory memorisation."
        ),
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_safe_json))
    (report_dir / "README.md").write_text(
        "# Phase 2 exact-forward gate\n\n"
        f"Run: `{run_id}`\n\n"
        "This report uses immutable saved observations and explicit fixed flow noise. Raw structured "
        "activations remain under `local/` and are intentionally excluded from Git. See `summary.json` "
        "for the predeclared scientific boundary.\n"
    )
    print(json.dumps(summary, indent=2, default=_safe_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
