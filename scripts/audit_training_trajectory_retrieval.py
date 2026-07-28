#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import zarr


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING = PROJECT / "local/training_data/lerobot_libero_a1aaacb"
DEFAULT_SOURCE = PROJECT / "archive/full_experiment/runs/benchmark_400_20260723T021424Z_e0638bea"
DEFAULT_GATE = PROJECT / "local/phase2_forward_gate/phase2_forward_gate_20260727T114152Z/queries.zarr"
DEFAULT_GATE_REPORT = PROJECT / "reports/phase2_forward_gate/phase2_forward_gate_20260727T114152Z"
DEFAULT_OUTPUT = PROJECT / "reports/trajectory_memorization_audit/phase2_train_retrieval_a1aaacb_v3"
HORIZON = 50
SCREEN_POSITIONS = np.asarray([0, 4, 9, 19, 29, 39, 49])


@dataclass(frozen=True)
class TrainingEpisode:
    episode_index: int
    task_index: int
    states: np.ndarray
    actions: np.ndarray


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(difference))))


def _action_statistics(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_action_rms": float(np.sqrt(np.mean(np.square(values)))),
        f"{prefix}_translation_rms": float(np.sqrt(np.mean(np.square(values[:, :3])))),
        f"{prefix}_rotation_rms": float(np.sqrt(np.mean(np.square(values[:, 3:6])))),
        f"{prefix}_gripper_abs_mean": float(np.mean(np.abs(values[:, 6]))),
        f"{prefix}_temporal_std": float(np.mean(np.std(values, axis=0))),
    }


def _task_mapping(root: Path) -> tuple[dict[int, str], dict[str, int]]:
    frame = pq.read_table(root / "meta/tasks.parquet").to_pandas()
    index_to_instruction = {
        int(row.task_index): str(index).strip().lower()
        for index, row in frame.iterrows()
    }
    return index_to_instruction, {instruction: index for index, instruction in index_to_instruction.items()}


def _load_training(root: Path) -> tuple[list[TrainingEpisode], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    table = pads.dataset(root / "data", format="parquet").to_table(
        columns=["observation.state", "action", "episode_index", "frame_index", "task_index"]
    )
    frame = table.to_pandas().sort_values(["episode_index", "frame_index"])
    episodes = []
    all_states = []
    all_episode_indices = []
    all_frame_indices = []
    all_task_indices = []
    for episode_index, group in frame.groupby("episode_index", sort=True):
        states = np.stack(group["observation.state"].to_numpy()).astype(np.float32)
        actions = np.stack(group["action"].to_numpy()).astype(np.float32)
        task_values = group["task_index"].unique()
        if len(task_values) != 1 or not np.array_equal(group["frame_index"], np.arange(len(group))):
            raise AssertionError(f"Malformed training episode {episode_index}")
        task_index = int(task_values[0])
        episodes.append(TrainingEpisode(int(episode_index), task_index, states, actions))
        all_states.append(states)
        all_episode_indices.append(np.full(len(group), episode_index, dtype=np.int64))
        all_frame_indices.append(np.arange(len(group), dtype=np.int64))
        all_task_indices.append(np.full(len(group), task_index, dtype=np.int64))
    return (
        episodes,
        np.concatenate(all_states),
        np.concatenate(all_episode_indices),
        np.concatenate(all_frame_indices),
        np.concatenate(all_task_indices),
    )


def _build_window_index(episodes: list[TrainingEpisode], action_std: np.ndarray):
    descriptors = []
    states = []
    episode_indices = []
    starts = []
    task_indices = []
    lookup = {episode.episode_index: episode for episode in episodes}
    for episode in episodes:
        count = len(episode.actions) - HORIZON + 1
        if count <= 0:
            continue
        offsets = np.arange(count)[:, None] + SCREEN_POSITIONS[None, :]
        descriptors.append(episode.actions[offsets] / action_std)
        states.append(episode.states[:count])
        episode_indices.append(np.full(count, episode.episode_index, dtype=np.int64))
        starts.append(np.arange(count, dtype=np.int64))
        task_indices.append(np.full(count, episode.task_index, dtype=np.int64))
    return {
        "descriptor": np.concatenate(descriptors),
        "state": np.concatenate(states),
        "episode": np.concatenate(episode_indices),
        "start": np.concatenate(starts),
        "task": np.concatenate(task_indices),
        "lookup": lookup,
    }


def _exact_best(query: np.ndarray, candidate_ids: np.ndarray, windows, action_std: np.ndarray) -> dict:
    best = None
    for chunk_start in range(0, len(candidate_ids), 512):
        chunk_ids = candidate_ids[chunk_start : chunk_start + 512]
        values = np.stack(
            [
                windows["lookup"][int(windows["episode"][index])].actions[
                    int(windows["start"][index]) : int(windows["start"][index]) + HORIZON
                ]
                for index in chunk_ids
            ]
        )
        standardized = np.sqrt(np.mean(np.square((values - query) / action_std), axis=(1, 2)))
        local = int(np.argmin(standardized))
        candidate = {
            "window_id": int(chunk_ids[local]),
            "standardized_rmse": float(standardized[local]),
            "raw_rmse": _rmse(values[local], query),
            "max_abs_difference": float(np.max(np.abs(values[local] - query))),
            **_action_statistics(query, "query"),
            **_action_statistics(values[local], "candidate"),
        }
        if best is None or candidate["standardized_rmse"] < best["standardized_rmse"]:
            best = candidate
    if best is None:
        raise ValueError("Empty candidate set")
    index = best.pop("window_id")
    return best | {
        "training_episode_index": int(windows["episode"][index]),
        "training_start_frame": int(windows["start"][index]),
        "training_task_index": int(windows["task"][index]),
    }


def _screened_best(
    query: np.ndarray,
    candidate_ids: np.ndarray,
    windows,
    action_std: np.ndarray,
    *,
    keep: int = 512,
) -> dict:
    query_descriptor = query[SCREEN_POSITIONS] / action_std
    difference = windows["descriptor"][candidate_ids] - query_descriptor
    distances = np.mean(np.square(difference), axis=(1, 2))
    count = min(keep, len(candidate_ids))
    selected = np.argpartition(distances, count - 1)[:count]
    return _exact_best(query, candidate_ids[selected], windows, action_std)


def _state_conditioned_best(
    query: np.ndarray,
    query_state: np.ndarray,
    candidate_ids: np.ndarray,
    windows,
    action_std: np.ndarray,
    state_std: np.ndarray,
    *,
    keep: int = 128,
) -> dict:
    distances = np.sqrt(
        np.mean(np.square((windows["state"][candidate_ids] - query_state) / state_std), axis=1)
    )
    count = min(keep, len(candidate_ids))
    selected = np.argpartition(distances, count - 1)[:count]
    result = _exact_best(query, candidate_ids[selected], windows, action_std)
    result["candidate_state_standardized_rmse"] = float(
        np.sqrt(
            np.mean(
                np.square(
                    (
                        windows["lookup"][result["training_episode_index"]].states[
                            result["training_start_frame"]
                        ]
                        - query_state
                    )
                    / state_std
                )
            )
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit SmolVLA action plans against official training state/action trajectories."
    )
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--gate-report", type=Path, default=DEFAULT_GATE_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-samples", type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    stats = json.loads((args.training / "meta/stats.json").read_text())
    action_std = np.maximum(np.asarray(stats["action"]["std"], dtype=np.float32), 1e-6)
    state_std = np.maximum(np.asarray(stats["observation.state"]["std"], dtype=np.float32), 1e-6)
    index_to_instruction, instruction_to_index = _task_mapping(args.training)
    episodes, all_states, all_episode, all_frame, all_task = _load_training(args.training)
    windows = _build_window_index(episodes, action_std)

    episode_table = pd.read_parquet(args.source_run / "episodes.parquet").set_index("episode_id")
    gate_queries = pd.read_csv(args.gate_report / "queries.csv")
    state_specs = (
        gate_queries.loc[gate_queries["condition"].eq("original"), ["state_id", "episode_id", "env_step"]]
        .drop_duplicates()
        .sort_values("state_id")
    )
    source_instruction = str(episode_table.loc[state_specs.iloc[0].episode_id, "instruction"]).lower()
    source_task_index = instruction_to_index[source_instruction]
    alternate_episode_id = "libero_goal_task04_ep000_seed0"
    alternate_instruction = str(episode_table.loc[alternate_episode_id, "instruction"]).lower()
    alternate_task_index = instruction_to_index[alternate_instruction]

    state_rows = []
    query_records = []
    gate = zarr.open_group(str(args.gate), mode="r")
    for spec in state_specs.itertuples(index=False):
        episode = episode_table.loc[spec.episode_id]
        observation = np.load(args.source_run / episode.observation_path)
        query_state = observation["policy_state"][int(spec.env_step)].astype(np.float32)
        for scope, mask in (
            ("same_task", all_task == source_task_index),
            ("alternate_goal_task", all_task == alternate_task_index),
            ("other_tasks", (all_task != source_task_index) & (all_task != alternate_task_index)),
        ):
            ids = np.flatnonzero(mask)
            distances = np.sqrt(np.mean(np.square((all_states[ids] - query_state) / state_std), axis=1))
            local = int(np.argmin(distances))
            index = int(ids[local])
            state_rows.append(
                {
                    "state_id": spec.state_id,
                    "env_step": int(spec.env_step),
                    "scope": scope,
                    "standardized_state_rmse": float(distances[local]),
                    "raw_state_rmse": _rmse(all_states[index], query_state),
                    "max_abs_state_difference": float(np.max(np.abs(all_states[index] - query_state))),
                    "training_episode_index": int(all_episode[index]),
                    "training_frame_index": int(all_frame[index]),
                    "training_task_index": int(all_task[index]),
                    "training_instruction": index_to_instruction[int(all_task[index])],
                }
            )

        for seed in (101, 202, 303, 404):
            group = gate[f"{spec.state_id}_original_noise{seed}"]
            plan = np.asarray(group["environment_action_chunk"])[0].astype(np.float32)
            query_records.append((spec.state_id, int(spec.env_step), f"proposal_seed_{seed}", query_state, plan))
        future = observation["executed_actions"][int(spec.env_step) : int(spec.env_step) + HORIZON]
        if len(future) == HORIZON:
            query_records.append(
                (spec.state_id, int(spec.env_step), "rollout_future", query_state, future.astype(np.float32))
            )

    scope_ids = {
        "same_task": np.flatnonzero(windows["task"] == source_task_index),
        "alternate_goal_task": np.flatnonzero(windows["task"] == alternate_task_index),
        "other_tasks": np.flatnonzero(
            (windows["task"] != source_task_index) & (windows["task"] != alternate_task_index)
        ),
    }
    plan_rows = []
    for state_id, env_step, query_kind, query_state, plan in query_records:
        for scope, ids in scope_ids.items():
            best = (
                _exact_best(plan, ids, windows, action_std)
                if scope != "other_tasks"
                else _screened_best(plan, ids, windows, action_std)
            )
            candidate_state = windows["lookup"][best["training_episode_index"]].states[
                best["training_start_frame"]
            ]
            best["candidate_state_standardized_rmse"] = _rmse(
                candidate_state / state_std, query_state / state_std
            )
            plan_rows.append(
                {
                    "state_id": state_id,
                    "env_step": env_step,
                    "query_kind": query_kind,
                    "scope": scope,
                    **best,
                    "training_instruction": index_to_instruction[best["training_task_index"]],
                }
            )
        conditioned = _state_conditioned_best(
            plan,
            query_state,
            scope_ids["same_task"],
            windows,
            action_std,
            state_std,
        )
        plan_rows.append(
            {
                "state_id": state_id,
                "env_step": env_step,
                "query_kind": query_kind,
                "scope": "same_task_state_top128",
                **conditioned,
                "training_instruction": index_to_instruction[conditioned["training_task_index"]],
            }
        )

    same_ids = scope_ids["same_task"]
    sample_positions = np.linspace(0, len(same_ids) - 1, min(args.self_samples, len(same_ids)), dtype=int)
    redundancy_rows = []
    for source_id in same_ids[sample_positions]:
        source_episode = int(windows["episode"][source_id])
        source_start = int(windows["start"][source_id])
        source_plan = windows["lookup"][source_episode].actions[source_start : source_start + HORIZON]
        candidates = same_ids[windows["episode"][same_ids] != source_episode]
        best = _screened_best(source_plan, candidates, windows, action_std, keep=256)
        redundancy_rows.append(
            {
                "source_episode_index": source_episode,
                "source_start_frame": source_start,
                **best,
            }
        )

    proposal_rows = []
    for state_id, records in pd.DataFrame(
        [
            {"state_id": state_id, "query_kind": kind, "plan": plan}
            for state_id, _, kind, _, plan in query_records
            if kind.startswith("proposal_seed_")
        ]
    ).groupby("state_id"):
        values = list(records.itertuples(index=False))
        for left, right in combinations(values, 2):
            proposal_rows.append(
                {
                    "state_id": state_id,
                    "left": left.query_kind,
                    "right": right.query_kind,
                    "standardized_rmse": _rmse(left.plan / action_std, right.plan / action_std),
                    "raw_rmse": _rmse(left.plan, right.plan),
                }
            )

    proposal_records = [
        {
            "state_id": state_id,
            "env_step": env_step,
            "seed": int(kind.rsplit("_", 1)[1]),
            "state": query_state,
            "plan": plan,
        }
        for state_id, env_step, kind, query_state, plan in query_records
        if kind.startswith("proposal_seed_")
    ]
    cross_state_rows = []
    for left, right in combinations(proposal_records, 2):
        if left["seed"] != right["seed"] or left["state_id"] == right["state_id"]:
            continue
        left_episode = left["state_id"].rsplit("_step", 1)[0]
        right_episode = right["state_id"].rsplit("_step", 1)[0]
        if left["env_step"] == right["env_step"]:
            relation = "same_step_different_episode"
        elif left_episode == right_episode:
            relation = "same_episode_different_step"
        else:
            relation = "different_episode_and_step"
        cross_state_rows.append(
            {
                "left_state_id": left["state_id"],
                "right_state_id": right["state_id"],
                "seed": left["seed"],
                "relation": relation,
                "state_standardized_rmse": _rmse(left["state"] / state_std, right["state"] / state_std),
                "plan_standardized_rmse": _rmse(left["plan"] / action_std, right["plan"] / action_std),
                "plan_raw_rmse": _rmse(left["plan"], right["plan"]),
            }
        )

    state_frame = pd.DataFrame(state_rows)
    plan_frame = pd.DataFrame(plan_rows)
    redundancy_frame = pd.DataFrame(redundancy_rows)
    proposal_frame = pd.DataFrame(proposal_rows)
    cross_state_frame = pd.DataFrame(cross_state_rows)
    state_frame.to_csv(args.output / "state_neighbours.csv", index=False)
    plan_frame.to_csv(args.output / "plan_neighbours.csv", index=False)
    redundancy_frame.to_csv(args.output / "training_redundancy.csv", index=False)
    proposal_frame.to_csv(args.output / "proposal_dispersion.csv", index=False)
    cross_state_frame.to_csv(args.output / "cross_state_plan_dispersion.csv", index=False)

    self_median = float(redundancy_frame["standardized_rmse"].median())
    proposal_same = plan_frame[
        plan_frame["query_kind"].str.startswith("proposal_seed_")
        & plan_frame["scope"].eq("same_task")
    ]
    proposal_conditioned = plan_frame[
        plan_frame["query_kind"].str.startswith("proposal_seed_")
        & plan_frame["scope"].eq("same_task_state_top128")
    ]
    rollout_same = plan_frame[
        plan_frame["query_kind"].eq("rollout_future") & plan_frame["scope"].eq("same_task")
    ]
    scope_medians = (
        plan_frame[plan_frame["query_kind"].str.startswith("proposal_seed_")]
        .groupby("scope")[["standardized_rmse", "candidate_state_standardized_rmse"]]
        .median()
        .reset_index()
        .to_dict(orient="records")
    )
    redundancy_values = redundancy_frame["standardized_rmse"].to_numpy()
    cross_state_medians = (
        cross_state_frame.groupby("relation")[[
            "state_standardized_rmse", "plan_standardized_rmse", "plan_raw_rmse"
        ]]
        .median()
        .reset_index()
        .to_dict(orient="records")
    )
    summary = {
        "schema_version": 1,
        "training_repo": "lerobot/libero",
        "training_revision": "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4",
        "training_episodes": len(episodes),
        "training_frames": int(len(all_states)),
        "training_windows_horizon_50": int(len(windows["task"])),
        "source_task_index": source_task_index,
        "source_instruction": source_instruction,
        "alternate_task_index": alternate_task_index,
        "alternate_instruction": alternate_instruction,
        "evaluation_states": int(len(state_specs)),
        "evaluated_plans": len(query_records),
        "training_self_nearest_standardized_rmse": {
            "p01": float(redundancy_frame["standardized_rmse"].quantile(0.01)),
            "median": self_median,
            "p99": float(redundancy_frame["standardized_rmse"].quantile(0.99)),
        },
        "proposal_same_task_nearest_standardized_rmse": {
            "min": float(proposal_same["standardized_rmse"].min()),
            "median": float(proposal_same["standardized_rmse"].median()),
            "max": float(proposal_same["standardized_rmse"].max()),
        },
        "proposal_to_training_copying_ratio_median": float(
            proposal_same["standardized_rmse"].median() / self_median
        ),
        "proposal_training_redundancy_percentile_median": float(
            np.median(
                [
                    np.mean(redundancy_values <= value)
                    for value in proposal_same["standardized_rmse"].to_numpy()
                ]
            )
        ),
        "proposal_scope_medians": scope_medians,
        "proposal_state_conditioning_penalty": float(
            proposal_conditioned["standardized_rmse"].median()
            / proposal_same["standardized_rmse"].median()
        ),
        "rollout_future_same_task_nearest_standardized_rmse_median": float(
            rollout_same["standardized_rmse"].median()
        ),
        "proposal_pairwise_standardized_rmse_median": float(
            proposal_frame["standardized_rmse"].median()
        ),
        "cross_state_same_seed_medians": cross_state_medians,
        "exact_action_window_matches_at_1e-6": int(
            (proposal_same["max_abs_difference"] <= 1e-6).sum()
        ),
        "exact_same_task_state_matches_at_1e-6": int(
            (
                state_frame.loc[state_frame["scope"].eq("same_task"), "max_abs_state_difference"]
                <= 1e-6
            ).sum()
        ),
        "scientific_boundary": (
            "Action/state retrieval proxy only. State omits object geometry and videos were deliberately excluded; "
            "nearest demonstration motion cannot by itself establish training-example memorisation or causal influence."
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "README.md").write_text(
        "# Phase 2 training-trajectory retrieval audit\n\n"
        "Revision-pinned official state/action Parquet only; no video was downloaded. "
        "See `summary.json` for the interpretation boundary.\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
