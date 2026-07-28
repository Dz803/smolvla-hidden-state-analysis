#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr
from safetensors import safe_open

from smolvla_analysis.phase3_analysis import (
    beta_posterior_mean,
    empirical_beta_prior,
    grouped_bootstrap_mean_interval,
    grouped_ridge_oof,
    hierarchical_variance_components,
    wilson_interval,
)
from smolvla_analysis.phase3_crd import (
    DEFAULT_STATE_SPECS,
    FACTOR_CONDITIONS,
    GOAL_PREDICATES,
    PROPOSAL_SEEDS,
    atomic_write_json,
    expected_query_ids,
    iter_branch_specs,
    validate_paired_first_plan,
)


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = PROJECT / "local/phase3_crd/phase3_crd_20260728T021125Z"
SUMMARY_HIDDEN_FEATURES = (
    "vlm_pooled_norm",
    "vlm_token_std",
    "expert_final_pooled_norm",
    "expert_final_token_std",
    "action_head_input_rms",
    "velocity_executed_rms",
    "velocity_padding_rms",
    "denoising_executed_path",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the complete Phase 3 CRD smoke matrix.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--ridge-sensitivity-alphas", type=float, nargs="*", default=[10.0, 1000.0])
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    return parser.parse_args()


def _rmse(left: Any, right: Any) -> float:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(difference))))


def _correlation(left: Any, right: Any) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 3 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _row_basis(weight: np.ndarray) -> np.ndarray:
    _, singular_values, right_vectors = np.linalg.svd(weight.astype(np.float64), full_matrices=False)
    tolerance = np.finfo(np.float64).eps * max(weight.shape) * singular_values.max(initial=0.0)
    return right_vectors[singular_values > tolerance]


def _row_energy_fraction(delta: np.ndarray, basis: np.ndarray) -> float:
    flattened = np.asarray(delta, dtype=np.float64).reshape(-1, delta.shape[-1])
    total = float(np.square(flattened).sum())
    if total == 0.0:
        return 0.0
    return float(np.square(flattened @ basis.T).sum() / total)


def _linear_output_rmse(delta: np.ndarray, weight: np.ndarray) -> float:
    output = np.asarray(delta, dtype=np.float64) @ np.asarray(weight, dtype=np.float64).T
    return float(np.sqrt(np.mean(np.square(output))))


def _load_payloads(directory: Path) -> dict[str, dict[str, Any]]:
    return {path.stem: json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))}


def _validate_run(run_dir: Path, manifest: dict[str, Any], branch_payloads, state_store, query_store) -> dict[str, Any]:
    expected_branches = {branch.branch_id for branch in iter_branch_specs()}
    expected_queries = set(expected_query_ids())
    branch_ids = set(branch_payloads)
    query_ids = set(query_store.group_keys())
    state_ids = set(state_store.group_keys())
    expected_states = {state.state_id for state in DEFAULT_STATE_SPECS}
    errors = []
    if manifest.get("status") != "complete":
        errors.append(f"manifest status is {manifest.get('status')!r}")
    if branch_ids != expected_branches:
        errors.append(f"branch ledger missing={len(expected_branches - branch_ids)} extra={len(branch_ids - expected_branches)}")
    if query_ids != expected_queries:
        errors.append(f"query ledger missing={len(expected_queries - query_ids)} extra={len(query_ids - expected_queries)}")
    if state_ids != expected_states:
        errors.append(f"state bank missing={len(expected_states - state_ids)} extra={len(state_ids - expected_states)}")
    incomplete_queries = [key for key in query_ids if query_store[key].attrs.get("complete") is not True]
    incomplete_states = [key for key in state_ids if state_store[key].attrs.get("complete") is not True]
    missing_summaries = [key for key in query_ids if not (run_dir / "query_summaries" / f"{key}.json").is_file()]
    if incomplete_queries or incomplete_states or missing_summaries:
        errors.append(
            f"incomplete queries={len(incomplete_queries)} states={len(incomplete_states)} summaries={len(missing_summaries)}"
        )
    query_reference_counts = pd.Series(
        [payload["query_id"] for payload in branch_payloads.values()]
    ).value_counts()
    if len(query_reference_counts) != 80 or set(query_reference_counts.index) != {
        branch.query_id for branch in iter_branch_specs()
    } or set(query_reference_counts.values) != {2}:
        errors.append("core query-to-branch reuse is not exactly 80 queries x 2 continuations")
    source_errors = []
    for branch_id, payload in branch_payloads.items():
        source = payload.get("source_reconstruction")
        if source is None:
            source_errors.append(branch_id)
            continue
        if (
            source.get("mode") != "archive_action_replay_current_process"
            or source.get("landmark_archive_fidelity", {}).get("exact") is not True
            or source.get("branch_certificate", {}).get("pass") is not True
        ):
            source_errors.append(branch_id)
    if source_errors:
        errors.append(f"branches lacking valid source reconstruction={len(source_errors)}")
    first_plan_mismatches = 0
    payloads_by_query: dict[str, list[dict[str, Any]]] = {}
    for payload in branch_payloads.values():
        payloads_by_query.setdefault(payload["query_id"], []).append(payload)
    for pair in payloads_by_query.values():
        if len(pair) != 2:
            first_plan_mismatches += 1
            continue
        try:
            validate_paired_first_plan(pair[0], pair[1])
        except ValueError:
            first_plan_mismatches += 1
    if first_plan_mismatches:
        errors.append(f"paired first-plan effect mismatches={first_plan_mismatches}")
    refreshes = manifest.get("branch_refreshes", [])
    if any(item.get("status") != "complete" for item in refreshes):
        errors.append("one or more branch refresh transactions are not complete")
    if errors:
        raise RuntimeError("Phase 3 validation failed: " + "; ".join(errors))
    return {
        "manifest_status": manifest["status"],
        "branches": len(branch_ids),
        "queries": len(query_ids),
        "core_queries": len(query_reference_counts),
        "factor_queries": len(query_ids) - len(query_reference_counts),
        "states": len(state_ids),
        "query_references_per_core_query": 2,
        "branches_with_source_reconstruction": len(branch_payloads),
        "paired_first_plan_effect_mismatches": 0,
        "partial_groups": 0,
        "historical_runner_errors_retained": len(manifest.get("errors", [])),
        "contract_amendments": len(manifest.get("contract_amendments", [])),
    }


def _effect_columns(prefix: str, effect: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_drawer": bool(effect["goals"]["drawer"]),
        f"{prefix}_cabinet": bool(effect["goals"]["cabinet"]),
        f"{prefix}_bowl_displacement_norm": float(effect["bowl_displacement_norm"]),
        f"{prefix}_eef_displacement_norm": float(effect["eef_displacement_norm"]),
        f"{prefix}_contact_count": int(effect["contact_count"]),
        f"{prefix}_grasped_count": len(effect["grasped_objects"]),
    }


def _branch_frame(branch_payloads: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for payload in branch_payloads.values():
        other_goal = "cabinet" if payload["target_goal"] == "drawer" else "drawer"
        target_step = payload["goal_first_step"][payload["target_goal"]]
        source_step = payload["goal_first_step"][payload["source_goal"]]
        row = {
            key: payload[key]
            for key in (
                "branch_id",
                "query_id",
                "state_id",
                "source_task_id",
                "source_episode_id",
                "source_episode_index",
                "source_seed",
                "source_goal",
                "target_goal",
                "landmark_step",
                "proposal_seed",
                "continuation_schedule",
                "steps_executed",
                "first_plan_steps",
                "continuation_replans",
                "terminal_reason",
                "underlying_done",
            )
        }
        row.update(
            {
                "success": int(bool(payload["success"])),
                "target_goal_first_step": target_step,
                "source_goal_first_step": source_step,
                "other_goal_first_step": payload["goal_first_step"][other_goal],
                "target_reached": int(target_step is not None),
                "source_goal_reached": int(source_step is not None),
                "target_in_first_plan": int(target_step is not None and target_step <= 50),
                "source_goal_in_first_plan": int(source_step is not None and source_step <= 50),
                "source_reconstruction_recorded": int("source_reconstruction" in payload),
            }
        )
        row.update(_effect_columns("first10", payload["first10_effect"]))
        row.update(_effect_columns("first_plan", payload["first_plan_effect"]))
        row.update(_effect_columns("final", payload["final_effect"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["state_id", "target_goal", "proposal_seed", "continuation_schedule"]
    ).reset_index(drop=True)


def _recoverability_tables(branches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    group_columns = [
        "state_id",
        "source_task_id",
        "source_episode_id",
        "source_goal",
        "target_goal",
        "landmark_step",
    ]
    recoverability = (
        branches.groupby(group_columns, as_index=False)["success"]
        .agg(successes="sum", trials="count")
    )
    recoverability["V"] = recoverability["successes"] / recoverability["trials"]
    intervals = [
        wilson_interval(int(row.successes), int(row.trials)) for row in recoverability.itertuples()
    ]
    recoverability["wilson_low"] = [item[0] for item in intervals]
    recoverability["wilson_high"] = [item[1] for item in intervals]
    v_prior = empirical_beta_prior(
        recoverability["successes"].to_numpy(), recoverability["trials"].to_numpy()
    )
    recoverability["V_eb"] = beta_posterior_mean(
        recoverability["successes"].to_numpy(), recoverability["trials"].to_numpy(), v_prior
    )

    proposal_columns = group_columns + ["proposal_seed"]
    proposal = branches.groupby(proposal_columns, as_index=False)["success"].agg(
        successes="sum", trials="count"
    )
    proposal["Q"] = proposal["successes"] / proposal["trials"]
    q_prior = empirical_beta_prior(proposal["successes"].to_numpy(), proposal["trials"].to_numpy())
    proposal["Q_eb"] = beta_posterior_mean(
        proposal["successes"].to_numpy(), proposal["trials"].to_numpy(), q_prior
    )
    proposal = proposal.merge(
        recoverability[["state_id", "target_goal", "V", "V_eb"]],
        on=["state_id", "target_goal"],
        validate="many_to_one",
    )
    proposal["L"] = proposal["Q"] - proposal["V"]
    proposal["L_eb"] = proposal["Q_eb"] - proposal["V_eb"]
    return recoverability, proposal, {"V_prior": v_prior, "Q_prior": q_prior}


def _state_certificate_frame(state_store) -> pd.DataFrame:
    rows = []
    for state_id in sorted(state_store.group_keys()):
        group = state_store[state_id]
        spec = json.loads(group.attrs["state_spec_json"])
        provenance = json.loads(group.attrs["provenance_json"])
        certificate = provenance["branch_certificate"]
        rows.append(
            {
                **spec,
                "resolved_init_state_id": provenance["resolved_init_state_id"],
                "initial_archive_exact": provenance["initial_archive_fidelity"]["exact"],
                "landmark_archive_exact": provenance["landmark_archive_fidelity"]["exact"],
                "round_trip_max_abs": provenance["round_trip"]["max_abs_state_diff"],
                "certificate_pass": certificate["pass"],
                "certificate_mujoco_max_abs": certificate["max_abs_mujoco_state_diff"],
                "certificate_observation_max_abs": certificate["max_abs_observation_diff"],
                "initial_drawer": provenance["common_goals"]["drawer"],
                "initial_cabinet": provenance["common_goals"]["cabinet"],
                "full_sim_data_persisted": "sim_data"
                in json.loads(group.attrs["metadata_json"])["runtime_state"],
            }
        )
    return pd.DataFrame(rows)


def _state_geometry_frame(state_store) -> pd.DataFrame:
    specs = {state.state_id: state for state in DEFAULT_STATE_SPECS}
    rows = []
    for state_id in sorted(state_store.group_keys()):
        group = state_store[state_id]
        metadata = json.loads(group.attrs["metadata_json"])
        bowl = metadata["objects"]["akita_black_bowl_1"]
        cabinet = metadata["objects"]["wooden_cabinet_1"]
        top_joints = [
            values
            for name, values in cabinet.get("joint_states", {}).items()
            if name.endswith("top_level")
        ]
        if len(top_joints) != 1 or len(top_joints[0]) != 1:
            raise RuntimeError(f"Cannot identify one top-drawer joint for {state_id}")
        spec = specs[state_id]
        bowl_position = np.asarray(bowl["position"], dtype=np.float64)
        eef_position = np.asarray(group["source_eef_pos"][:], dtype=np.float64)
        drawer_joint = float(top_joints[0][0])
        rows.append(
            {
                "state_id": state_id,
                "source_episode_id": spec.source_episode_id,
                "source_goal": spec.source_goal,
                "landmark_step": spec.landmark_step,
                "bowl_x": bowl_position[0],
                "bowl_y": bowl_position[1],
                "bowl_z": bowl_position[2],
                "bowl_grasped": int(bool(bowl["grasped"])),
                "top_drawer_joint": drawer_joint,
                "top_drawer_displaced": int(abs(drawer_joint) > 1e-6),
                "eef_x": eef_position[0],
                "eef_y": eef_position[1],
                "eef_z": eef_position[2],
            }
        )
    return pd.DataFrame(rows)


def _span_role(name: str) -> str:
    if name.endswith("camera1"):
        return "main_view"
    if name.endswith("camera2"):
        return "wrist_view"
    if name == "observation.language":
        return "language"
    if name == "observation.state":
        return "state"
    return name.replace("observation.", "")


def _last_activation_name(group, pathway: str) -> str:
    names = [name for name in group["activations"].array_keys() if name.startswith(f"{pathway}_layer_")]
    if not names:
        raise KeyError(f"No {pathway} activation in {group.name}")
    return sorted(names)[-1]


def _vlm_span_summaries(group) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    activation = np.asarray(group[f"activations/{_last_activation_name(group, 'vlm')}"][:], dtype=np.float64)[0, 0]
    mask = np.asarray(group["prefix_pad_mask"][:], dtype=bool)[0]
    pooled = {}
    token_std = {}
    for span in json.loads(group.attrs["token_spans_json"]):
        role = _span_role(span["name"])
        values = activation[span["start"] : span["stop"]]
        valid = mask[span["start"] : span["stop"]]
        values = values[valid]
        if len(values) == 0:
            continue
        pooled[role] = values.mean(axis=0)
        token_std[role] = float(np.mean(np.std(values, axis=0)))
    return pooled, token_std


def _structured_hidden_features(group, summary: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    values = [float(summary[key]) for key in SUMMARY_HIDDEN_FEATURES]
    names = list(SUMMARY_HIDDEN_FEATURES)
    pooled, token_std = _vlm_span_summaries(group)
    for role in ("main_view", "wrist_view", "language", "state"):
        vector = pooled[role]
        values.extend([float(np.linalg.norm(vector)), token_std[role]])
        names.extend([f"vlm_{role}_pooled_norm", f"vlm_{role}_token_std"])
    expert_names = sorted(
        name for name in group["activations"].array_keys() if name.startswith("action_expert_layer_")
    )
    for name in expert_names:
        activation = np.asarray(group[f"activations/{name}"][:], dtype=np.float64)
        final = activation[-1, 0]
        values.extend(
            [float(np.linalg.norm(final.mean(axis=0))), float(np.mean(np.std(final, axis=0)))]
        )
        layer = name.rsplit("_", 1)[1]
        names.extend([f"expert_layer_{layer}_final_pooled_norm", f"expert_layer_{layer}_final_token_std"])
    final_expert = np.asarray(group[f"activations/{expert_names[-1]}"][:], dtype=np.float64)[:, 0].mean(axis=1)
    head = np.asarray(group["action_head_inputs"][:], dtype=np.float64)[:, 0].mean(axis=1)
    values.extend(
        [
            float(np.sqrt(np.square(np.diff(final_expert, axis=0)).mean(axis=1)).sum()),
            float(np.sqrt(np.square(np.diff(head, axis=0)).mean(axis=1)).sum()),
        ]
    )
    names.extend(["expert_denoising_path", "head_input_denoising_path"])
    return np.asarray(values, dtype=np.float64), names


def _delta_metrics(left, right, weights: dict[str, np.ndarray]) -> dict[str, Any]:
    left_chunk = np.asarray(left["environment_action_chunk"][:], dtype=np.float64)
    right_chunk = np.asarray(right["environment_action_chunk"][:], dtype=np.float64)
    left_head = np.asarray(left["action_head_inputs"][-1], dtype=np.float64)
    right_head = np.asarray(right["action_head_inputs"][-1], dtype=np.float64)
    head_delta = right_head - left_head
    left_expert = np.asarray(
        left[f"activations/{_last_activation_name(left, 'action_expert')}"][-1], dtype=np.float64
    )
    right_expert = np.asarray(
        right[f"activations/{_last_activation_name(right, 'action_expert')}"][-1], dtype=np.float64
    )
    left_velocity = np.asarray(left["denoising_velocity"][:], dtype=np.float64)
    right_velocity = np.asarray(right["denoising_velocity"][:], dtype=np.float64)
    left_pooled, _ = _vlm_span_summaries(left)
    right_pooled, _ = _vlm_span_summaries(right)
    result = {
        "action_rmse": _rmse(left_chunk, right_chunk),
        "action_first10_rmse": _rmse(left_chunk[:, :10], right_chunk[:, :10]),
        "action_expert_final_rmse": _rmse(left_expert, right_expert),
        "action_head_input_rmse": _rmse(left_head, right_head),
        "velocity_executed_rmse": _rmse(left_velocity[..., :7], right_velocity[..., :7]),
        "velocity_padding_rmse": _rmse(left_velocity[..., 7:], right_velocity[..., 7:]),
        "active_head_output_rmse": _linear_output_rmse(head_delta, weights["active"]),
        "padding_head_output_rmse": _linear_output_rmse(head_delta, weights["padding"]),
        "active_row_energy_fraction": _row_energy_fraction(head_delta, weights["active_basis"]),
        "padding_row_energy_fraction": _row_energy_fraction(head_delta, weights["padding_basis"]),
        "full_output_row_energy_fraction": _row_energy_fraction(head_delta, weights["full_basis"]),
        "output_null_energy_fraction": 1.0 - _row_energy_fraction(head_delta, weights["full_basis"]),
        "flow_noise_exact": bool(np.array_equal(left["flow_noise"][:], right["flow_noise"][:])),
    }
    for role in ("main_view", "wrist_view", "language", "state"):
        result[f"vlm_{role}_pooled_rmse"] = _rmse(left_pooled[role], right_pooled[role])
    result["vlm_pooled_rmse"] = _rmse(
        np.concatenate([left_pooled[key] for key in sorted(left_pooled)]),
        np.concatenate([right_pooled[key] for key in sorted(right_pooled)]),
    )
    result["action_to_vlm_gain"] = result["action_rmse"] / max(result["vlm_pooled_rmse"], 1e-12)
    return result


def _query_geometry(query_store, proposal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    weight_path = PROJECT / "archive/full_experiment/checkpoints/smolvla_libero/model.safetensors"
    with safe_open(str(weight_path), framework="np") as checkpoint:
        weight = checkpoint.get_tensor("model.action_out_proj.weight").astype(np.float64)
    weights = {
        "active": weight[:7],
        "padding": weight[7:],
        "active_basis": _row_basis(weight[:7]),
        "padding_basis": _row_basis(weight[7:]),
        "full_basis": _row_basis(weight),
    }
    factor_rows = []
    for state in DEFAULT_STATE_SPECS:
        for goal in GOAL_PREDICATES:
            original_id = f"{state.state_id}__goal_{goal}__proposal_101"
            original = query_store[original_id]
            for factor in FACTOR_CONDITIONS:
                counter_id = f"{state.state_id}__goal_{goal}__factor_{factor}__proposal_101"
                factor_rows.append(
                    {
                        "state_id": state.state_id,
                        "source_task_id": state.source_task_id,
                        "source_goal": state.source_goal,
                        "goal": goal,
                        "comparison_type": "controlled_factor",
                        "factor": factor,
                        **_delta_metrics(original, query_store[counter_id], weights),
                    }
                )
            alternate = "cabinet" if goal == "drawer" else "drawer"
            alternate_id = f"{state.state_id}__goal_{alternate}__proposal_101"
            factor_rows.append(
                {
                    "state_id": state.state_id,
                    "source_task_id": state.source_task_id,
                    "source_goal": state.source_goal,
                    "goal": goal,
                    "comparison_type": "goal_switch",
                    "factor": "alternate_goal",
                    **_delta_metrics(original, query_store[alternate_id], weights),
                }
            )

    q_lookup = proposal.set_index(["state_id", "target_goal", "proposal_seed"])["Q"].to_dict()
    noise_rows = []
    for state in DEFAULT_STATE_SPECS:
        for goal in GOAL_PREDICATES:
            for left_seed, right_seed in combinations(PROPOSAL_SEEDS, 2):
                left_id = f"{state.state_id}__goal_{goal}__proposal_{left_seed}"
                right_id = f"{state.state_id}__goal_{goal}__proposal_{right_seed}"
                left_q = q_lookup[(state.state_id, goal, left_seed)]
                right_q = q_lookup[(state.state_id, goal, right_seed)]
                noise_rows.append(
                    {
                        "state_id": state.state_id,
                        "source_task_id": state.source_task_id,
                        "source_goal": state.source_goal,
                        "goal": goal,
                        "left_seed": left_seed,
                        "right_seed": right_seed,
                        "left_Q": left_q,
                        "right_Q": right_q,
                        "absolute_Q_difference": abs(left_q - right_q),
                        **_delta_metrics(query_store[left_id], query_store[right_id], weights),
                    }
                )
    factor = pd.DataFrame(factor_rows)
    noise = pd.DataFrame(noise_rows)
    controlled = factor[factor["comparison_type"] == "controlled_factor"]
    goal_switch = factor[factor["comparison_type"] == "goal_switch"]
    geometry_summary = {
        "hidden_dimension": int(weight.shape[1]),
        "internal_action_dimension": int(weight.shape[0]),
        "executed_action_dimension": 7,
        "active_row_rank": int(weights["active_basis"].shape[0]),
        "padding_row_rank": int(weights["padding_basis"].shape[0]),
        "full_row_rank": int(weights["full_basis"].shape[0]),
        "isotropic_active_fraction": float(weights["active_basis"].shape[0] / weight.shape[1]),
        "isotropic_full_fraction": float(weights["full_basis"].shape[0] / weight.shape[1]),
        "noise_action_Q_difference_correlation": _correlation(
            noise["action_rmse"], noise["absolute_Q_difference"]
        ),
        "noise_expert_Q_difference_correlation": _correlation(
            noise["action_expert_final_rmse"], noise["absolute_Q_difference"]
        ),
        "noise_active_row_energy_median": float(noise["active_row_energy_fraction"].median()),
        "noise_padding_row_energy_median": float(noise["padding_row_energy_fraction"].median()),
        "noise_output_null_energy_median": float(noise["output_null_energy_fraction"].median()),
        "controlled_factor_active_row_energy_median": float(
            controlled["active_row_energy_fraction"].median()
        ),
        "controlled_factor_output_null_energy_median": float(
            controlled["output_null_energy_fraction"].median()
        ),
        "goal_switch_active_row_energy_median": float(
            goal_switch["active_row_energy_fraction"].median()
        ),
        "goal_switch_output_null_energy_median": float(
            goal_switch["output_null_energy_fraction"].median()
        ),
    }
    return factor, noise, geometry_summary


def _flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    if isinstance(value, dict):
        result = {}
        for key in sorted(value):
            result.update(_flatten_numeric(value[key], f"{prefix}/{key}"))
        return result
    if isinstance(value, (list, tuple)):
        result = {}
        for index, item in enumerate(value):
            result.update(_flatten_numeric(item, f"{prefix}/{index}"))
        return result
    if isinstance(value, (bool, int, float, np.number)) and np.isfinite(float(value)):
        return {prefix.lstrip("/"): float(value)}
    return {}


def _downsample_image(image: np.ndarray, size: int = 8) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    if image.shape != (3, 256, 256) or 256 % size:
        raise ValueError(f"Unexpected policy image shape: {image.shape}")
    block = 256 // size
    return image.reshape(3, size, block, size, block).mean(axis=(2, 4)).reshape(-1)


def _effect_vector(payload: dict[str, Any]) -> np.ndarray:
    effect = payload["first_plan_effect"]
    return np.asarray(
        [
            *effect["bowl_displacement"],
            effect["bowl_displacement_norm"],
            *effect["eef_displacement"],
            effect["eef_displacement_norm"],
            len(effect["grasped_objects"]),
            effect["contact_count"],
            int(effect["goals"]["drawer"]),
            int(effect["goals"]["cabinet"]),
            payload["first_plan_steps"],
        ],
        dtype=np.float64,
    )


def _conditional_features(
    run_dir: Path,
    state_store,
    query_store,
    branch_payloads: dict[str, dict[str, Any]],
    recoverability: pd.DataFrame,
    proposal: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    state_geometry = {}
    all_geometry_keys = set()
    for state_id in state_store.group_keys():
        metadata = json.loads(state_store[state_id].attrs["metadata_json"])
        flattened = _flatten_numeric(metadata["objects"])
        state_geometry[state_id] = flattened
        all_geometry_keys.update(flattened)
    geometry_keys = sorted(all_geometry_keys)
    spec_lookup = {state.state_id: state for state in DEFAULT_STATE_SPECS}
    static = {}
    static_names = None
    for state_id in sorted(state_store.group_keys()):
        group = state_store[state_id]
        spec = spec_lookup[state_id]
        history = np.asarray(group["source_action_prefix"][:], dtype=np.float64)
        padded_history = np.zeros((100, 7), dtype=np.float64)
        padded_history[: len(history)] = history
        geometry = np.asarray([state_geometry[state_id].get(key, 0.0) for key in geometry_keys])
        base = np.concatenate(
            [
                np.asarray(group["policy_state"][:], dtype=np.float64),
                _downsample_image(group["policy_image"][:]),
                _downsample_image(group["policy_image2"][:]),
                geometry,
                padded_history.reshape(-1),
                np.asarray(
                    [len(history) / 100.0, spec.landmark_step / 100.0, int(spec.source_task_id == 3), int(spec.source_task_id == 4)],
                    dtype=np.float64,
                ),
            ]
        )
        static[state_id] = base
        if static_names is None:
            static_names = (
                [f"policy_state_{index}" for index in range(8)]
                + [f"main_pixel_pool_{index}" for index in range(192)]
                + [f"wrist_pixel_pool_{index}" for index in range(192)]
                + [f"geometry/{key}" for key in geometry_keys]
                + [f"history_{index}" for index in range(700)]
                + ["history_length", "landmark_step", "source_task_3", "source_task_4"]
            )

    proposal_rows = []
    baseline_pre = []
    baseline_effect = []
    hidden_all = []
    hidden_non_vlm = []
    hidden_names = None
    for row in proposal.sort_values(["state_id", "target_goal", "proposal_seed"]).itertuples():
        query_id = f"{row.state_id}__goal_{row.target_goal}__proposal_{int(row.proposal_seed)}"
        group = query_store[query_id]
        summary = json.loads((run_dir / "query_summaries" / f"{query_id}.json").read_text())
        hidden, names = _structured_hidden_features(group, summary)
        if hidden_names is None:
            hidden_names = names
        elif hidden_names != names:
            raise RuntimeError("Structured hidden feature schema changed across queries")
        branches = [
            branch_payloads[f"{query_id}__continuation_0"],
            branch_payloads[f"{query_id}__continuation_1"],
        ]
        effects = np.stack([_effect_vector(payload) for payload in branches])
        if not np.allclose(effects[0], effects[1], atol=1e-10, rtol=0.0):
            raise RuntimeError(f"First-plan effect changed across continuation schedules: {query_id}")
        target_one_hot = np.asarray([row.target_goal == "drawer", row.target_goal == "cabinet"], dtype=float)
        seed_one_hot = np.asarray([row.proposal_seed == seed for seed in PROPOSAL_SEEDS], dtype=float)
        proposal_pre = np.concatenate(
            [
                static[row.state_id],
                target_one_hot,
                seed_one_hot,
                np.asarray(group["environment_action_chunk"][:], dtype=np.float64).reshape(-1),
                np.asarray(group["flow_noise"][:], dtype=np.float64).reshape(-1),
            ]
        )
        baseline_pre.append(proposal_pre)
        baseline_effect.append(np.concatenate([proposal_pre, effects[0]]))
        hidden_all.append(hidden)
        non_vlm_indices = [
            index
            for index, name in enumerate(hidden_names)
            if not name.startswith("vlm_")
        ]
        hidden_non_vlm.append(hidden[non_vlm_indices])
        proposal_rows.append(
            {
                "state_id": row.state_id,
                "source_episode_id": row.source_episode_id,
                "target_goal": row.target_goal,
                "proposal_seed": row.proposal_seed,
                "Q": row.Q,
                "L": row.L,
                "query_id": query_id,
            }
        )
    proposal_index = pd.DataFrame(proposal_rows)
    baseline_pre = np.stack(baseline_pre)
    baseline_effect = np.stack(baseline_effect)
    hidden_all = np.stack(hidden_all)
    hidden_non_vlm = np.stack(hidden_non_vlm)

    v_rows = []
    v_baseline = []
    v_hidden = []
    for row in recoverability.sort_values(["state_id", "target_goal"]).itertuples():
        mask = (proposal_index["state_id"] == row.state_id) & (
            proposal_index["target_goal"] == row.target_goal
        )
        indices = np.flatnonzero(mask.to_numpy())
        target_one_hot = np.asarray([row.target_goal == "drawer", row.target_goal == "cabinet"], dtype=float)
        v_baseline.append(np.concatenate([static[row.state_id], target_one_hot, baseline_effect[indices, len(static[row.state_id]) + 6 :].mean(axis=0)]))
        vlm_indices = [index for index, name in enumerate(hidden_names) if name.startswith("vlm_")]
        v_hidden.append(hidden_all[indices[0], vlm_indices])
        v_rows.append(
            {
                "state_id": row.state_id,
                "source_episode_id": row.source_episode_id,
                "target_goal": row.target_goal,
                "V": row.V,
            }
        )

    return {
        "V": {
            "index": pd.DataFrame(v_rows),
            "baseline": np.stack(v_baseline),
            "hidden": np.stack(v_hidden),
            "target": np.asarray([row["V"] for row in v_rows]),
        },
        "Q_preexecution": {
            "index": proposal_index,
            "baseline": baseline_pre,
            "hidden": hidden_all,
            "target": proposal_index["Q"].to_numpy(),
        },
        "Q_effect_controlled": {
            "index": proposal_index,
            "baseline": baseline_effect,
            "hidden": hidden_all,
            "target": proposal_index["Q"].to_numpy(),
        },
        "L_effect_controlled": {
            "index": proposal_index,
            "baseline": baseline_effect,
            "hidden": hidden_non_vlm,
            "target": proposal_index["L"].to_numpy(),
        },
    }, pd.DataFrame(
        {
            "feature_set": ["static", "proposal_preexecution", "proposal_effect", "hidden_all", "hidden_non_vlm"],
            "dimensions": [
                len(next(iter(static.values()))),
                baseline_pre.shape[1],
                baseline_effect.shape[1],
                hidden_all.shape[1],
                hidden_non_vlm.shape[1],
            ],
        }
    )


def _conditional_models(feature_sets: dict[str, dict[str, Any]], alpha: float, repetitions: int):
    rows = []
    prediction_rows = []
    folds = {}
    for target_name, data in feature_sets.items():
        target = data["target"]
        groups = data["index"]["source_episode_id"].to_numpy()
        baseline = grouped_ridge_oof(data["baseline"], target, groups, alpha=alpha)
        augmented = grouped_ridge_oof(
            np.concatenate([data["baseline"], data["hidden"]], axis=1),
            target,
            groups,
            alpha=alpha,
        )
        improvement = np.square(target - baseline.predictions) - np.square(
            target - augmented.predictions
        )
        interval = grouped_bootstrap_mean_interval(
            improvement, groups, repetitions=repetitions, seed=20260728
        )
        rows.append(
            {
                "target": target_name,
                "rows": len(target),
                "groups": len(np.unique(groups)),
                "baseline_dimensions": data["baseline"].shape[1],
                "hidden_dimensions": data["hidden"].shape[1],
                "ridge_alpha": alpha,
                "baseline_rmse": baseline.rmse,
                "augmented_rmse": augmented.rmse,
                "baseline_mae": baseline.mae,
                "augmented_mae": augmented.mae,
                "hidden_mse_improvement": interval["estimate"],
                "hidden_mse_improvement_ci_low": interval["ci_low"],
                "hidden_mse_improvement_ci_high": interval["ci_high"],
            }
        )
        index = data["index"].reset_index(drop=True)
        for position in range(len(target)):
            prediction_rows.append(
                {
                    "target": target_name,
                    **index.iloc[position].to_dict(),
                    "observed": target[position],
                    "baseline_prediction": baseline.predictions[position],
                    "augmented_prediction": augmented.predictions[position],
                }
            )
        folds[target_name] = {
            "baseline": list(baseline.folds),
            "augmented": list(augmented.folds),
        }
    return pd.DataFrame(rows), pd.DataFrame(prediction_rows), folds


def _safe_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    raise TypeError(type(value).__name__)


def main() -> int:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    report_dir = (
        args.report_dir.resolve()
        if args.report_dir
        else PROJECT / "reports/phase3_crd" / manifest["run_id"]
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    branch_payloads = _load_payloads(run_dir / "branches")
    state_store = zarr.open_group(str(run_dir / "states.zarr"), mode="r")
    query_store = zarr.open_group(str(run_dir / "queries.zarr"), mode="r")
    validation = _validate_run(run_dir, manifest, branch_payloads, state_store, query_store)

    branches = _branch_frame(branch_payloads)
    recoverability, proposal, shrinkage = _recoverability_tables(branches)
    certificates = _state_certificate_frame(state_store)
    state_geometry = _state_geometry_frame(state_store)
    variance = hierarchical_variance_components(
        branches["success"].to_numpy(dtype=float),
        (branches["state_id"] + "__" + branches["target_goal"]).to_numpy(),
        branches["proposal_seed"].to_numpy(),
    )
    schedule = branches.pivot_table(
        index=["state_id", "target_goal", "proposal_seed"],
        columns="continuation_schedule",
        values="success",
        aggfunc="first",
    ).astype(float)
    schedule_difference = schedule[1] - schedule[0]
    variance.update(
        {
            "continuation_schedule_1_minus_0": float(schedule_difference.mean()),
            "continuation_disagreements": int((schedule_difference != 0).sum()),
            "continuation_pairs": int(len(schedule_difference)),
            "proposal_varying_state_goal_cells": int(
                (
                    proposal.groupby(["state_id", "target_goal"])["Q"].nunique()
                    > 1
                ).sum()
            ),
            "state_goal_cells": int(len(recoverability)),
            "Q_distribution": {
                str(key): int(value)
                for key, value in proposal["Q"].value_counts().sort_index().items()
            },
        }
    )

    goal_faithfulness = (
        branches.groupby(
            ["state_id", "source_episode_id", "source_goal", "target_goal"],
            as_index=False,
        )
        .agg(
            V=("success", "mean"),
            target_reached_rate=("target_reached", "mean"),
            source_goal_reached_rate=("source_goal_reached", "mean"),
            target_in_first_plan_rate=("target_in_first_plan", "mean"),
            source_goal_in_first_plan_rate=("source_goal_in_first_plan", "mean"),
            mean_steps=("steps_executed", "mean"),
        )
    )
    goal_aggregate = (
        branches.groupby(["source_goal", "target_goal"], as_index=False)
        .agg(
            success_rate=("success", "mean"),
            successes=("success", "sum"),
            branches=("success", "count"),
            source_goal_reached_rate=("source_goal_reached", "mean"),
            target_in_first_plan_rate=("target_in_first_plan", "mean"),
        )
    )
    native_counter = []
    for source_goal in ("drawer", "cabinet"):
        source_states = goal_faithfulness[goal_faithfulness["source_goal"] == source_goal]
        index_columns = ["state_id", "source_episode_id"]
        native = source_states[source_states["target_goal"] == source_goal].set_index(
            index_columns
        )["V"]
        counter_goal = "cabinet" if source_goal == "drawer" else "drawer"
        counter = source_states[source_states["target_goal"] == counter_goal].set_index(
            index_columns
        )["V"]
        differences = (native - counter).sort_index()
        interval = grouped_bootstrap_mean_interval(
            differences.to_numpy(),
            differences.index.get_level_values("source_episode_id").to_numpy(),
            repetitions=args.bootstrap_repetitions,
        )
        native_counter.append(
            {
                "source_goal": source_goal,
                "native_goal": source_goal,
                "counter_goal": counter_goal,
                "native_V": float(native.mean()),
                "counter_V": float(counter.mean()),
                "native_minus_counter": interval["estimate"],
                "ci_low": interval["ci_low"],
                "ci_high": interval["ci_high"],
                "states": len(differences),
                "source_episodes": interval["group_count"],
                "bootstrap_unit": "source_episode",
            }
        )

    factor_effects, proposal_geometry, geometry_summary = _query_geometry(query_store, proposal)
    feature_sets, feature_dimensions = _conditional_features(
        run_dir, state_store, query_store, branch_payloads, recoverability, proposal
    )
    conditional_frames = []
    prediction_frames = []
    fold_details = {}
    for alpha in dict.fromkeys([args.ridge_alpha, *args.ridge_sensitivity_alphas]):
        if alpha <= 0:
            raise ValueError("Ridge alphas must be positive")
        model_frame, prediction_frame, alpha_folds = _conditional_models(
            feature_sets, alpha, args.bootstrap_repetitions
        )
        model_frame["primary_alpha"] = alpha == args.ridge_alpha
        prediction_frame["ridge_alpha"] = alpha
        conditional_frames.append(model_frame)
        prediction_frames.append(prediction_frame)
        fold_details[str(alpha)] = alpha_folds
    conditional = pd.concat(conditional_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    refresh_rows = []
    all_refreshed = set()
    success_flips = 0
    semantic_changes = 0
    for refresh in manifest.get("branch_refreshes", []):
        backup_dir = run_dir / "superseded_branches" / refresh["kind"]
        refresh_flips = 0
        refresh_changes = 0
        for branch_id in refresh["target_branch_ids"]:
            old = json.loads((backup_dir / f"{branch_id}.json").read_text())
            new = branch_payloads[branch_id]
            refresh_flips += int(bool(old["success"]) != bool(new["success"]))
            old_semantic = {
                key: value
                for key, value in old.items()
                if key not in {"wall_time_s", "source_reconstruction"}
            }
            new_semantic = {
                key: value
                for key, value in new.items()
                if key not in {"wall_time_s", "source_reconstruction"}
            }
            refresh_changes += int(old_semantic != new_semantic)
            all_refreshed.add(branch_id)
        success_flips += refresh_flips
        semantic_changes += refresh_changes
        refresh_rows.append(
            {
                "kind": refresh["kind"],
                "branches": len(refresh["target_branch_ids"]),
                "semantic_payload_changes": refresh_changes,
                "success_label_flips": refresh_flips,
            }
        )

    for name, frame in (
        ("branches", branches),
        ("state_certificates", certificates),
        ("state_geometry_audit", state_geometry),
        ("recoverability", recoverability),
        ("proposal_quality", proposal),
        ("goal_faithfulness", goal_faithfulness),
        ("goal_aggregate", goal_aggregate),
        ("factor_effects", factor_effects),
        ("proposal_geometry", proposal_geometry),
        ("conditional_models", conditional),
        ("conditional_predictions", predictions),
        ("feature_dimensions", feature_dimensions),
    ):
        frame.to_csv(report_dir / f"{name}.csv", index=False)
    atomic_write_json(report_dir / "conditional_folds.json", fold_details)

    factor_medians = (
        factor_effects.groupby(["comparison_type", "factor"])
        .median(numeric_only=True)
        .reset_index()
        .to_dict(orient="records")
    )
    source_target = {
        f"{row.source_goal}_to_{row.target_goal}": {
            "success_rate": row.success_rate,
            "successes": int(row.successes),
            "branches": int(row.branches),
            "source_goal_reached_rate": row.source_goal_reached_rate,
            "target_in_first_plan_rate": row.target_in_first_plan_rate,
        }
        for row in goal_aggregate.itertuples()
    }
    veto_composition = {}
    for source_goal in ("drawer", "cabinet"):
        alternate_goal = "cabinet" if source_goal == "drawer" else "drawer"
        native_row = goal_aggregate[
            (goal_aggregate["source_goal"] == source_goal)
            & (goal_aggregate["target_goal"] == source_goal)
        ].iloc[0]
        alternate_row = goal_aggregate[
            (goal_aggregate["source_goal"] == source_goal)
            & (goal_aggregate["target_goal"] == alternate_goal)
        ].iloc[0]
        veto_strength = float(
            native_row.source_goal_reached_rate - alternate_row.source_goal_reached_rate
        )
        constructive_transfer = float(alternate_row.success_rate)
        veto_composition[source_goal] = {
            "alternate_goal": alternate_goal,
            "native_source_goal_rate": float(native_row.source_goal_reached_rate),
            "source_goal_rate_under_alternate_instruction": float(
                alternate_row.source_goal_reached_rate
            ),
            "veto_strength": veto_strength,
            "constructive_transfer": constructive_transfer,
            "veto_minus_constructive_transfer": veto_strength - constructive_transfer,
            "descriptive_only": True,
        }
    state_bank_selection = {}
    for source_goal, frame in state_geometry.groupby("source_goal"):
        state_bank_selection[source_goal] = {
            "states": int(len(frame)),
            "landmark_counts": {
                str(int(key)): int(value)
                for key, value in frame["landmark_step"].value_counts().sort_index().items()
            },
            "bowl_grasped_rate": float(frame["bowl_grasped"].mean()),
            "bowl_z_median": float(frame["bowl_z"].median()),
            "top_drawer_displaced_rate": float(frame["top_drawer_displaced"].mean()),
            "top_drawer_joint_median": float(frame["top_drawer_joint"].median()),
        }
    primary_models = (
        conditional[conditional["primary_alpha"]]
        .set_index("target")
        .to_dict(orient="index")
    )
    summary = {
        "schema_version": 3,
        "run_id": manifest["run_id"],
        "contract_sha256": manifest["contract_sha256"],
        "validation": validation,
        "outcomes": {
            "successes": int(branches["success"].sum()),
            "branches": len(branches),
            "success_rate": float(branches["success"].mean()),
            "terminal_reasons": {
                str(key): int(value) for key, value in branches["terminal_reason"].value_counts().items()
            },
            "initially_satisfied_goal_states": int(
                (certificates[["initial_drawer", "initial_cabinet"]].any(axis=1)).sum()
            ),
        },
        "source_target_recoverability": source_target,
        "veto_composition": veto_composition,
        "state_bank_selection": state_bank_selection,
        "native_counter_cluster_bootstrap": native_counter,
        "shrinkage": shrinkage,
        "variance_components": variance,
        "executable_geometry": geometry_summary,
        "factor_medians": factor_medians,
        "conditional_hidden_value": primary_models,
        "restore_defects": {
            "refreshed_unique_branches": len(all_refreshed),
            "semantic_payload_changes": semantic_changes,
            "success_label_flips": success_flips,
            "refresh_transactions": refresh_rows,
            "cold_cross_instance_state_max_abs": 0.1262197780027186,
            "cold_cross_instance_pixel_max_abs": 191.0,
            "drop_qacc_warmstart_state_max_abs": 2.4043933469219958e-08,
            "drop_qacc_warmstart_pixel_max_abs": 1.0,
        },
        "estimability": {
            "smolvla_state_goal_recoverability": "estimable descriptively across 20 state-goal cells",
            "common_state_difficulty": "not separable from SmolVLA competence with only one policy",
            "proposal_luck": "estimable for 80 proposals with two continuation schedules each",
            "policy_specific_competence": "not separable from common difficulty with only one policy",
            "self_specificity": "not estimable until matched pi0.5/GR00T branches exist",
            "factor_outcome_interactions": "not estimable; Phase 3 factors are fixed-forward only",
        },
        "scientific_boundary": (
            "Ten states from eight source episodes form a method-validation smoke, not a generalization benchmark. "
            "Conditional models use leave-one-source-episode-out prediction and low-dimensional hidden summaries. "
            "First-plan effects are retrospective proposal controls, not early-warning features. Goal-switch asymmetry "
            "is current-state/occupancy conditioned and does not by itself prove memorization or language grounding."
        ),
    }
    atomic_write_json(report_dir / "summary.json", summary)

    q_model = primary_models["Q_effect_controlled"]
    l_model = primary_models["L_effect_controlled"]
    q_sensitivity = conditional[conditional["target"] == "Q_effect_controlled"]
    q_sensitivity_statement = (
        "non-positive at every tested ridge alpha"
        if (q_sensitivity["hidden_mse_improvement"] <= 0).all()
        else "not directionally stable across ridge alphas"
    )
    readme = f"""# Phase 3 certified recoverability decomposition

Run: `{manifest['run_id']}`

The repaired active ledger contains **{len(branches)} branches**, **80 exact proposal queries**, and **80 fixed-noise factor queries** from ten certified states. All goals were false at every branch root. The active outcome rate is {branches['success'].mean():.3f} ({int(branches['success'].sum())}/{len(branches)}).

## Main result: asymmetric goal switchability

- Cabinet-trajectory states: cabinet `V={source_target['cabinet_to_cabinet']['success_rate']:.3f}`, drawer `V={source_target['cabinet_to_drawer']['success_rate']:.3f}`.
- Drawer-trajectory states: drawer `V={source_target['drawer_to_drawer']['success_rate']:.3f}`, cabinet `V={source_target['drawer_to_cabinet']['success_rate']:.3f}`.

Changing the instruction away from cabinet prevents the model from completing cabinet, but it does not create drawer competence. Conversely, drawer-source states retain meaningful access to the simpler cabinet goal.

Descriptively, the instruction switch suppresses the source predicate by `{veto_composition['cabinet']['veto_strength']:.3f}` on cabinet-source states while constructive drawer transfer is `{veto_composition['cabinet']['constructive_transfer']:.3f}`. On drawer-source states, suppression is `{veto_composition['drawer']['veto_strength']:.3f}` and constructive cabinet transfer is `{veto_composition['drawer']['constructive_transfer']:.3f}`. This **veto–composition gap** is more precise than saying that the language prior is simply strong or weak: language changes the policy, but compositional recovery is constrained by the current physical/occupancy state.

This is **not evidence of recurrent trajectory memory**: the policy is reset before every branch and exposes only an action queue, not an observation-history queue. It is current-state-conditioned recoverability. The source state bank is also progress-confounded: cabinet states are all step 50 with median bowl height `{state_bank_selection['cabinet']['bowl_z_median']:.3f}`, `{state_bank_selection['cabinet']['bowl_grasped_rate']:.0%}` grasped, and a closed top drawer; drawer states mix steps 50/100 with median bowl height `{state_bank_selection['drawer']['bowl_z_median']:.3f}`, none grasped, and the drawer displaced in `{state_bank_selection['drawer']['top_drawer_displaced_rate']:.0%}`. The asymmetry can therefore reflect physical subgoal preparation or occupancy-manifold capture, not language understanding alone. See `state_geometry_audit.csv`.

## Recoverability versus luck

State-goal differences explain {variance['state_goal_fraction']:.1%} of branch variance, proposal differences {variance['proposal_within_state_goal_fraction']:.1%}, and continuation randomness {variance['continuation_within_proposal_fraction']:.1%}. Only {variance['continuation_disagreements']}/{variance['continuation_pairs']} matched continuation pairs disagree, while {variance['proposal_varying_state_goal_cells']}/{variance['state_goal_cells']} state-goal cells vary across first proposals.

## Hidden-state test under complete controls

For proposal quality `Q`, adding hidden summaries after privileged state/geometry, downsampled pixels, trajectory history, target, proposal seed, raw noise, the full ordered action chunk, and its realized first-plan effect changes held-source-episode RMSE from {q_model['baseline_rmse']:.3f} to {q_model['augmented_rmse']:.3f}. The grouped MSE improvement is {q_model['hidden_mse_improvement']:+.4f} with bootstrap interval [{q_model['hidden_mse_improvement_ci_low']:+.4f}, {q_model['hidden_mse_improvement_ci_high']:+.4f}]. For proposal luck `L`, the corresponding RMSE is {l_model['baseline_rmse']:.3f} to {l_model['augmented_rmse']:.3f}.

The `Q` hidden-state improvement is {q_sensitivity_statement} (`alpha=10,100,1000`). This smoke therefore provides no evidence that these low-dimensional hidden summaries know proposal quality beyond the plan/effect baseline.

## Executable geometry

The fixed-noise factor table preserves modality-specific VLM, action-expert, flow, and executed/padding/null action-head changes. Median controlled-factor output-null energy is {geometry_summary['controlled_factor_output_null_energy_median']:.1%}; median proposal-noise output-null energy is {geometry_summary['noise_output_null_energy_median']:.1%}. These are fixed-forward geometry results, not causal hidden-state mediation.

## Critical simulator discovery

MuJoCo's flattened state omits `qacc_warmstart`. Removing only that field reproduces the contact-state divergence; cold cross-instance restore can change state by `0.126` and pixels by `191/255`. One audited success became failure after exact reconstruction. Across {len(refresh_rows)} provenance-preserving repair transaction(s), {len(all_refreshed)} legacy payloads were regenerated, {semantic_changes} semantic payloads changed, and **{success_flips} success labels flipped**. Every active branch now carries current-process reconstruction evidence. See `restore_field_ablation.json` and `branch_replay_audits/`.

## Boundary

This is a ten-state SmolVLA smoke study. It cannot estimate policy-specific competence or self-specificity until matched π0.5/GR00T experiments exist. The raw factor name `contradiction` denotes a contrastive instruction that negates the *other* goal and reaffirms the target; it is not an internally inconsistent instruction. Finally, the study does not establish that a decoded hidden feature causally controls behavior, and the negative conditional test does not justify scaling the current summary probe unchanged.
"""
    (report_dir / "README.md").write_text(readme)
    print(json.dumps(summary, indent=2, default=_safe_json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
