from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .libero_state import (
    LiberoStateSnapshot,
    capture_libero_state,
    libero_problem_environment,
    restore_libero_state,
    validate_libero_round_trip,
)
from .phase3_crd import (
    MUJOCO_STATE_ATOL,
    NUMERIC_OBSERVATION_ATOL,
    PIXEL_OBSERVATION_ATOL,
    certificate_within_tolerance,
    evaluate_common_goals,
    nested_field_max_abs_differences,
)
from .phase3b_alignment import landmark_registered_point
from .phase3b_stage_a import (
    GOALS,
    LAYOUT_INIT_STATE_IDS,
    StageACandidateSpec,
    canonical_sha256,
    goal_distances,
    measure_joint_support,
    recovery_balanced_goal_axis_point,
    snapshot_sha256,
)
from .runtime import _asset_path, _prepare_libero_runtime_config


BOWL_NAME = "akita_black_bowl_1"
TOP_DRAWER_JOINT = "wooden_cabinet_1_top_level"
TOP_DRAWER_BODY = "wooden_cabinet_1_cabinet_top"
DRAWER_GOAL_SITE = "wooden_cabinet_1_top_region"
CABINET_GOAL_SITE = "wooden_cabinet_1_top_side"


@dataclass(frozen=True)
class DemoTrace:
    goal: str
    episode_index: int
    task_index: int
    frame_indices: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    action_sha256: str


@dataclass(frozen=True)
class ActionPhaseProposal:
    """One source proposal with a layout-specific, action-derived phase anchor."""

    source: DemoTrace
    suffix: DemoTrace
    anchor_position: np.ndarray
    anchor_orientation: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SupportReferenceBank:
    entries: tuple[dict[str, Any], ...]
    sha256: str
    provenance: dict[str, Any]


@dataclass
class ConstructedCandidate:
    snapshot: LiberoStateSnapshot
    initial_bowl_position: np.ndarray
    initial_eef_position: np.ndarray
    initial_eef_orientation: np.ndarray
    initial_joint_positions: np.ndarray
    recovery_waypoints: dict[str, np.ndarray] | None
    support_measurement: dict[str, Any] | None
    construction: dict[str, Any]
    root_validation: dict[str, Any]
    root_geometry: dict[str, Any]


@dataclass
class PreparedOracleRoot:
    snapshot: LiberoStateSnapshot
    phases: dict[str, Any]
    actions: tuple[np.ndarray, ...]
    eef_path_length_m: float
    control_effort: float
    motion_control_effort: float
    active_servo_steps: int
    done_count: int
    normalization_goal_ever: bool
    normalization_done_ever: bool
    normalized_goals: dict[str, bool]
    normalized_bowl_position_error_m: float


def load_demo_trace(
    dataset_root: Path, *, goal: str, episode_index: int, task_index: int
) -> DemoTrace:
    import pyarrow.dataset as arrow_dataset

    data_root = dataset_root / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(f"Missing LeRobot training data: {data_root}")
    dataset = arrow_dataset.dataset(data_root, format="parquet")
    table = dataset.to_table(
        columns=[
            "episode_index",
            "task_index",
            "frame_index",
            "observation.state",
            "action",
        ],
        filter=arrow_dataset.field("episode_index") == episode_index,
    ).to_pandas()
    if table.empty:
        raise ValueError(f"Demonstration episode {episode_index} is empty")
    table = table.sort_values("frame_index")
    if set(table["task_index"].astype(int)) != {task_index}:
        raise ValueError(
            f"Episode {episode_index} does not have expected task index {task_index}"
        )
    frame_indices = table["frame_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(frame_indices, np.arange(len(table), dtype=np.int64)):
        raise ValueError(f"Episode {episode_index} frames are not contiguous from zero")
    states = np.stack(table["observation.state"]).astype(np.float32)
    actions = np.stack(table["action"]).astype(np.float32)
    if states.shape[1:] != (8,) or actions.shape[1:] != (7,):
        raise ValueError(
            f"Unexpected demonstration shapes: states={states.shape}, actions={actions.shape}"
        )
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError(f"Episode {episode_index} contains non-finite values")
    digest = sha256()
    digest.update(np.asarray(actions.shape, dtype="<i8").tobytes())
    digest.update(actions.astype("<f4", copy=False).tobytes(order="C"))
    return DemoTrace(
        goal=goal,
        episode_index=episode_index,
        task_index=task_index,
        frame_indices=frame_indices,
        states=states,
        actions=actions,
        action_sha256=digest.hexdigest(),
    )


def list_demo_trace_inventory(
    dataset_root: Path, *, task_index: int
) -> tuple[dict[str, int], ...]:
    import pyarrow.dataset as arrow_dataset

    data_root = dataset_root / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(f"Missing LeRobot training data: {data_root}")
    table = arrow_dataset.dataset(data_root, format="parquet").to_table(
        columns=["episode_index", "task_index", "frame_index"],
        filter=arrow_dataset.field("task_index") == int(task_index),
    ).to_pandas()
    if table.empty:
        raise ValueError(f"No demonstrations found for task {task_index}")
    inventory = tuple(
        sorted(
            (
                {
                    "episode_index": int(episode_index),
                    "frame_count": int(len(group)),
                }
                for episode_index, group in table.groupby(
                    "episode_index", sort=False
                )
            ),
            key=lambda row: (row["frame_count"], row["episode_index"]),
        )
    )
    if len({row["episode_index"] for row in inventory}) != len(inventory):
        raise ValueError(f"Duplicate episode identities for task {task_index}")
    return inventory


def make_stage_a_environment(project: Path, runtime_dir: Path, config: dict[str, Any]):
    runtime_config = runtime_dir / "runtime_libero_config"
    runtime_config.parent.mkdir(parents=True, exist_ok=True)
    if not runtime_config.exists():
        _prepare_libero_runtime_config(project, runtime_config)
    os.environ["LIBERO_CONFIG_PATH"] = str(runtime_config)
    os.environ["SMOLVLA_LIBERO_ASSETS"] = str(
        _asset_path(project, "checkpoints/libero_assets")
    )

    import libero.libero as libero_runtime
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env

    libero_runtime._assets_path_cache = os.environ["SMOLVLA_LIBERO_ASSETS"]
    env_config = config["environment"]
    task_id = int(env_config["task_id"])
    env_cfg = LiberoEnv(
        task="libero_goal",
        task_ids=[task_id],
        episode_length=int(env_config["episode_length"]),
        obs_type="pixels_agent_pos",
        observation_height=int(env_config["observation_height"]),
        observation_width=int(env_config["observation_width"]),
        control_mode=str(env_config["control_mode"]),
        max_parallel_tasks=1,
    )
    environment = make_env(env_cfg, n_envs=1, use_async_envs=False)["libero_goal"][
        task_id
    ]
    requested_horizon = int(env_config["episode_length"])
    problem = libero_problem_environment(environment)
    problem.horizon = requested_horizon
    if int(problem.horizon) != requested_horizon:
        environment.close()
        raise RuntimeError(
            f"Failed to set native LIBERO horizon to {requested_horizon}"
        )
    return environment


class PolicyFreeController:
    def __init__(self, environment):
        self.environment = environment
        self.actions: list[np.ndarray] = []
        self.eef_path_length_m = 0.0
        self.control_effort = 0.0
        self.motion_control_effort = 0.0
        self.done_values: list[bool] = []
        self.goal_values: list[dict[str, bool]] = []
        self.grasp_values: list[bool] = []
        self.grasp_relative_positions: list[np.ndarray] = []

    @property
    def scalar(self):
        return self.environment.envs[0]

    @property
    def problem(self):
        return libero_problem_environment(self.environment)

    def reset_layout(self, init_state_id: int, seed: int) -> None:
        self.scalar.init_state_id = int(init_state_id)
        self.environment.reset(seed=int(seed))
        self.actions.clear()
        self.eef_path_length_m = 0.0
        self.control_effort = 0.0
        self.motion_control_effort = 0.0
        self.done_values.clear()
        self.goal_values.clear()
        self.grasp_values.clear()
        self.grasp_relative_positions.clear()

    def eef_position(self) -> np.ndarray:
        robot = self.problem.robots[0]
        return np.asarray(
            self.problem.sim.data.site_xpos[robot.eef_site_id], dtype=np.float64
        ).copy()

    def eef_orientation(self) -> np.ndarray:
        robot = self.problem.robots[0]
        return np.asarray(
            self.problem.sim.data.site_xmat[robot.eef_site_id], dtype=np.float64
        ).reshape(3, 3).copy()

    def bowl_position(self) -> np.ndarray:
        body_id = self.problem.obj_body_id[BOWL_NAME]
        return np.asarray(
            self.problem.sim.data.body_xpos[body_id], dtype=np.float64
        ).copy()

    def joint_positions(self) -> np.ndarray:
        robot = self.problem.robots[0]
        return np.asarray(
            self.problem.sim.data.qpos[robot._ref_joint_pos_indexes],
            dtype=np.float64,
        ).copy()

    def site_position(self, name: str) -> np.ndarray:
        return np.asarray(
            self.problem.sim.data.get_site_xpos(name), dtype=np.float64
        ).copy()

    def top_drawer_position(self) -> float:
        return float(self.problem.sim.data.get_joint_qpos(TOP_DRAWER_JOINT))

    def bowl_grasped(self) -> bool:
        return bool(
            self.problem._check_grasp(
                gripper=self.problem.robots[0].gripper,
                object_geoms=self.problem.get_object(BOWL_NAME),
            )
        )

    def top_drawer_handle_position(self) -> tuple[str, np.ndarray]:
        model = self.problem.sim.model
        body_id = model.body_name2id(TOP_DRAWER_BODY)
        candidates = [
            geom_id
            for geom_id, geom_body_id in enumerate(model.geom_bodyid)
            if int(geom_body_id) == int(body_id)
        ]
        if not candidates:
            raise ValueError("Top drawer body has no collision geometry")
        geom_id = max(candidates, key=lambda item: self.problem.sim.data.geom_xpos[item][1])
        name = model.geom_id2name(geom_id)
        if name is None:
            raise ValueError("Top drawer handle geometry is unnamed")
        return name, np.asarray(
            self.problem.sim.data.geom_xpos[geom_id], dtype=np.float64
        ).copy()

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        if (
            action.shape != (7,)
            or not np.isfinite(action).all()
            or np.max(np.abs(action), initial=0.0) > 1.000001
        ):
            raise ValueError(f"Invalid LIBERO action shape or values: {action}")
        before = self.eef_position()
        raw, reward, done, info = self.scalar._env.step(action)
        observation = self.scalar._format_raw_obs(raw)
        terminal = bool(done or self.problem.done)
        after = self.eef_position()
        self.actions.append(action.copy())
        self.eef_path_length_m += float(np.linalg.norm(after - before))
        self.control_effort += float(np.linalg.norm(action))
        self.motion_control_effort += float(np.linalg.norm(action[:6]))
        self.done_values.append(terminal)
        self.goal_values.append(evaluate_common_goals(self.environment))
        self.grasp_values.append(self.bowl_grasped())
        self.grasp_relative_positions.append(
            self.eef_position() - self.bowl_position()
        )
        return observation, float(reward), terminal, info

    def servo(
        self,
        *,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        gripper: float,
        budget: int,
        max_translation_action: float,
        position_tolerance_m: float,
        orientation_tolerance_rad: float = 0.10,
        pad_to_budget: bool = True,
        stop_on_goal_or_terminal: bool = False,
    ) -> dict[str, Any]:
        from robosuite.utils import transform_utils as transform

        target_position = np.asarray(target_position, dtype=np.float64)
        target_orientation = np.asarray(target_orientation, dtype=np.float64)
        if target_position.shape != (3,) or target_orientation.shape != (3, 3):
            raise ValueError("Servo target has an invalid shape")
        if budget < 1 or not 0.0 < max_translation_action <= 1.0:
            raise ValueError("Servo budget/action bound is invalid")
        history_start = len(self.actions)
        start_relative_position = self.eef_position() - self.bowl_position()
        active_steps = 0
        max_position_error = 0.0
        stopped_on_goal = False
        stopped_on_terminal = False
        for _ in range(budget):
            position_error = target_position - self.eef_position()
            rotation_error = target_orientation @ self.eef_orientation().T
            axis_angle = transform.quat2axisangle(transform.mat2quat(rotation_error))
            position_norm = float(np.linalg.norm(position_error))
            orientation_norm = float(np.linalg.norm(axis_angle))
            max_position_error = max(max_position_error, position_norm)
            action = np.zeros(7, dtype=np.float32)
            needs_motion = bool(
                position_norm > position_tolerance_m
                or orientation_norm > orientation_tolerance_rad
            )
            if not needs_motion and not pad_to_budget:
                break
            if needs_motion:
                active_steps += 1
                action[:3] = np.clip(
                    position_error / 0.035,
                    -max_translation_action,
                    max_translation_action,
                )
                action[3:6] = np.clip(axis_angle / 0.50, -1.0, 1.0)
            action[6] = float(gripper)
            self.step(action)
            if stop_on_goal_or_terminal:
                stopped_on_goal = bool(any(self.goal_values[-1].values()))
                stopped_on_terminal = bool(self.done_values[-1])
                if stopped_on_goal or stopped_on_terminal:
                    break
        final_position_error = float(
            np.linalg.norm(target_position - self.eef_position())
        )
        final_rotation = target_orientation @ self.eef_orientation().T
        final_orientation_error = float(
            np.linalg.norm(transform.quat2axisangle(transform.mat2quat(final_rotation)))
        )
        phase_grasps = self.grasp_values[history_start:]
        longest_grasp_dropout = 0
        current_grasp_dropout = 0
        for grasped in phase_grasps:
            if grasped:
                current_grasp_dropout = 0
            else:
                current_grasp_dropout += 1
                longest_grasp_dropout = max(
                    longest_grasp_dropout, current_grasp_dropout
                )
        phase_relative_positions = self.grasp_relative_positions[history_start:]
        max_relative_pose_deviation = max(
            (
                float(np.linalg.norm(position - start_relative_position))
                for position in phase_relative_positions
            ),
            default=0.0,
        )
        return {
            "budgeted_action_steps": int(budget),
            "executed_action_steps": int(len(self.actions) - history_start),
            "padded_to_budget": bool(pad_to_budget),
            **(
                {
                    "stop_on_goal_or_terminal": True,
                    "stopped_on_goal": stopped_on_goal,
                    "stopped_on_terminal": stopped_on_terminal,
                }
                if stop_on_goal_or_terminal
                else {}
            ),
            "active_action_steps": int(active_steps),
            "max_position_error_m": max_position_error,
            "final_position_error_m": final_position_error,
            "final_orientation_error_rad": final_orientation_error,
            "grasp_preserved_every_step": bool(
                all(phase_grasps)
            ),
            "grasp_false_step_count": int(
                sum(not grasped for grasped in phase_grasps)
            ),
            "max_consecutive_grasp_dropout_steps": int(longest_grasp_dropout),
            "max_grasp_relative_pose_deviation_m": max_relative_pose_deviation,
            "no_goal_every_step": bool(
                not any(
                    any(goals.values())
                    for goals in self.goal_values[history_start:]
                )
            ),
            "nonterminal_every_step": bool(
                not any(self.done_values[history_start:])
            ),
            "pass": bool(
                final_position_error <= position_tolerance_m
                and final_orientation_error <= orientation_tolerance_rad
            ),
        }

    def replay_until(
        self,
        demo: DemoTrace,
        condition: Callable[[], bool],
    ) -> dict[str, Any]:
        first_satisfied_frame = None
        for frame_index, action in zip(demo.frame_indices, demo.actions, strict=True):
            _, _, done, _ = self.step(action)
            if done:
                raise RuntimeError(
                    f"Demonstration episode {demo.episode_index} terminated "
                    f"before its construction condition at frame {int(frame_index)}"
                )
            if condition():
                first_satisfied_frame = int(frame_index)
                break
        if first_satisfied_frame is None:
            raise RuntimeError(
                f"Demonstration episode {demo.episode_index} did not reach construction condition"
            )
        executed = np.stack(self.actions[-(first_satisfied_frame + 1) :])
        return {
            "episode_index": demo.episode_index,
            "task_index": demo.task_index,
            "source_action_sha256": demo.action_sha256,
            "last_frame_index": first_satisfied_frame,
            "executed_action_count": int(len(executed)),
            "executed_action_sha256": _action_sha256(executed),
        }


def _action_sha256(actions: list[np.ndarray] | np.ndarray) -> str:
    values = np.asarray(actions, dtype=np.float32)
    digest = sha256()
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
    digest.update(values.astype("<f4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def action_phase_suffix(
    demo: DemoTrace,
    *,
    maximum_pregrasp_lead_frames: int,
    minimum_anchor_prefix_frames: int,
    gripper_close_threshold: float,
) -> tuple[DemoTrace, dict[str, Any]]:
    """Derive a fixed pre-grasp continuation without using object-state labels."""

    if demo.goal not in GOALS:
        raise ValueError(f"Unknown phase-slice goal: {demo.goal}")
    maximum_lead = int(maximum_pregrasp_lead_frames)
    minimum_prefix = int(minimum_anchor_prefix_frames)
    threshold = float(gripper_close_threshold)
    if maximum_lead < 1 or minimum_prefix < 1 or not np.isfinite(threshold):
        raise ValueError("Action-phase slice parameters are invalid")
    gripper = np.asarray(demo.actions[:, 6], dtype=np.float64)
    close_candidates = np.flatnonzero(
        (gripper[1:] > threshold) & (gripper[:-1] <= threshold)
    ) + 1
    if not len(close_candidates):
        raise ValueError(
            f"Episode {demo.episode_index} has no gripper-close transition"
        )
    close_index = int(close_candidates[0])
    if close_index <= minimum_prefix:
        raise ValueError(
            f"Episode {demo.episode_index} has only {close_index} frames before "
            f"gripper close; more than {minimum_prefix} are required"
        )
    suffix_start = max(minimum_prefix, close_index - maximum_lead)
    realized_lead = close_index - suffix_start
    suffix_actions = demo.actions[suffix_start:].copy()
    suffix = DemoTrace(
        goal=demo.goal,
        episode_index=demo.episode_index,
        task_index=demo.task_index,
        frame_indices=demo.frame_indices[suffix_start:].copy(),
        states=demo.states[suffix_start:].copy(),
        actions=suffix_actions,
        action_sha256=_action_sha256(suffix_actions),
    )
    return suffix, {
        "first_gripper_close_index": close_index,
        "first_gripper_close_frame": int(demo.frame_indices[close_index]),
        "maximum_pregrasp_lead_frames": maximum_lead,
        "realized_pregrasp_lead_frames": realized_lead,
        "minimum_anchor_prefix_frames": minimum_prefix,
        "suffix_start_index": suffix_start,
        "suffix_start_frame": int(demo.frame_indices[suffix_start]),
        "anchor_after_frame": int(demo.frame_indices[suffix_start - 1]),
        "prefix_action_count": suffix_start,
        "prefix_action_sha256": _action_sha256(demo.actions[:suffix_start]),
        "suffix_action_count": int(len(suffix.actions)),
        "suffix_action_sha256": suffix.action_sha256,
    }


def _transport_validation_limits(config: dict[str, Any]) -> dict[str, float | int]:
    validation = config["validation"]
    return {
        "max_grasp_dropout_steps": int(
            validation["max_transient_grasp_dropout_steps"]
        ),
        "max_relative_pose_deviation_m": float(
            validation["max_transport_relative_pose_deviation_m"]
        ),
    }


def replay_grasped_recovery_route(
    controller: PolicyFreeController,
    *,
    root_transit_positions: np.ndarray,
    lifted_eef_position: np.ndarray,
    target_orientation: np.ndarray,
    phase_budgets: dict[str, int],
    intermediate_tolerance_m: float,
    final_tolerance_m: float,
    max_translation_action: float,
    candidate_id: str,
    max_grasp_dropout_steps: int,
    max_relative_pose_deviation_m: float,
) -> dict[str, Any]:
    plan = grasped_root_recovery_plan(
        root_transit_positions,
        lifted_eef_position,
        phase_budgets=phase_budgets,
    )
    action_start = len(controller.actions)
    route_start_relative_position = (
        controller.eef_position() - controller.bowl_position()
    )
    phase_records: list[dict[str, Any]] = []
    for recovery_phase in plan:
        result = controller.servo(
            target_position=recovery_phase["target_position"],
            target_orientation=np.asarray(target_orientation),
            gripper=1.0,
            budget=int(recovery_phase["budget"]),
            max_translation_action=float(max_translation_action),
            position_tolerance_m=float(
                intermediate_tolerance_m
                if recovery_phase["intermediate"]
                else final_tolerance_m
            ),
        )
        _validate_grasped_transport_phase(
            controller,
            result,
            candidate_id=candidate_id,
            phase=f"setdown_{recovery_phase['phase']}",
            max_grasp_dropout_steps=max_grasp_dropout_steps,
            max_relative_pose_deviation_m=max_relative_pose_deviation_m,
        )
        phase_records.append(
            {
                "phase": recovery_phase["phase"],
                "target_position": recovery_phase["target_position"].tolist(),
                "intermediate": recovery_phase["intermediate"],
                "result": result,
            }
        )
    route_actions = np.stack(controller.actions[action_start:])
    expected_count = sum(int(phase["budget"]) for phase in plan)
    if len(route_actions) != expected_count:
        raise RuntimeError(
            f"Recovery-route action count mismatch for {candidate_id}: "
            f"{len(route_actions)} != {expected_count}"
        )
    results = [record["result"] for record in phase_records]
    route_grasps = controller.grasp_values[action_start:]
    longest_route_dropout = 0
    current_route_dropout = 0
    for grasped in route_grasps:
        if grasped:
            current_route_dropout = 0
        else:
            current_route_dropout += 1
            longest_route_dropout = max(
                longest_route_dropout, current_route_dropout
            )
    route_relative_positions = controller.grasp_relative_positions[action_start:]
    max_route_relative_deviation = max(
        (
            float(np.linalg.norm(position - route_start_relative_position))
            for position in route_relative_positions
        ),
        default=0.0,
    )
    return {
        "mode": "feedback_reverse_clearance_route",
        "pass": True,
        "budgeted_action_steps": expected_count,
        "active_action_steps": int(
            sum(int(result["active_action_steps"]) for result in results)
        ),
        "action_sha256": _action_sha256(route_actions),
        "final_position_error_m": float(results[-1]["final_position_error_m"]),
        "final_orientation_error_rad": float(
            results[-1]["final_orientation_error_rad"]
        ),
        "grasp_preserved_every_step": bool(
            all(route_grasps)
        ),
        "grasp_false_step_count": int(
            sum(not grasped for grasped in route_grasps)
        ),
        "max_consecutive_grasp_dropout_steps": int(longest_route_dropout),
        "max_grasp_relative_pose_deviation_m": max_route_relative_deviation,
        "no_goal_every_step": bool(
            all(result["no_goal_every_step"] for result in results)
        ),
        "nonterminal_every_step": bool(
            all(result["nonterminal_every_step"] for result in results)
        ),
        "phases": phase_records,
    }


def _workspace_check(point: np.ndarray, bounds: dict[str, list[float]]) -> None:
    point = np.asarray(point, dtype=np.float64)
    for index, axis in enumerate(("x", "y", "z")):
        low, high = (float(value) for value in bounds[axis])
        if not low <= point[index] <= high:
            raise ValueError(
                f"Constructed root target {point.tolist()} violates {axis} bounds {low, high}"
            )


def grasped_root_transit_plan(
    start_position: np.ndarray,
    target_position: np.ndarray,
    *,
    clearance_margin_m: float,
    workspace_bounds: dict[str, list[float]],
    phase_budgets: dict[str, int],
) -> tuple[dict[str, Any], ...]:
    start = np.asarray(start_position, dtype=np.float64)
    target = np.asarray(target_position, dtype=np.float64)
    if (
        start.shape != (3,)
        or target.shape != (3,)
        or not np.isfinite(start).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("Grasped-root transit endpoints must be finite 3-D points")
    margin = float(clearance_margin_m)
    if not np.isfinite(margin) or margin <= 0.0:
        raise ValueError("Grasped-root clearance margin must be positive and finite")
    names = ("clearance_lift", "clearance_transit", "target_descent")
    if set(phase_budgets) != set(names) or any(
        int(phase_budgets[name]) < 1 for name in names
    ):
        raise ValueError("Grasped-root transit phase budgets are incomplete")
    z_upper = float(workspace_bounds["z"][1])
    clearance_z = min(z_upper, max(float(start[2]), float(target[2])) + margin)
    if clearance_z < max(float(start[2]), float(target[2])):
        raise ValueError("Grasped-root endpoints exceed the clearance workspace")
    plan = (
        {
            "phase": "clearance_lift",
            "target_position": np.asarray([start[0], start[1], clearance_z]),
            "budget": int(phase_budgets["clearance_lift"]),
        },
        {
            "phase": "clearance_transit",
            "target_position": np.asarray([target[0], target[1], clearance_z]),
            "budget": int(phase_budgets["clearance_transit"]),
        },
        {
            "phase": "target_descent",
            "target_position": target.copy(),
            "budget": int(phase_budgets["target_descent"]),
        },
    )
    for phase in plan:
        _workspace_check(phase["target_position"], workspace_bounds)
    return plan


def build_action_phase_proposal_bank(
    environment,
    *,
    layout: str,
    proposals: tuple[DemoTrace, ...],
    config: dict[str, Any],
) -> tuple[ActionPhaseProposal, ...]:
    """Replay only source prefixes needed to obtain deterministic EEF anchors."""

    if layout not in LAYOUT_INIT_STATE_IDS:
        raise ValueError(f"Unknown action-phase layout: {layout}")
    if not proposals:
        raise ValueError("Action-phase proposal bank is empty")
    goal = proposals[0].goal
    if goal not in GOALS or any(proposal.goal != goal for proposal in proposals):
        raise ValueError("Action-phase proposal bank mixes goals")
    phase_cfg = config["action_phase_oracle"]
    controller = PolicyFreeController(environment)
    entries: list[ActionPhaseProposal] = []
    for proposal_index, demo in enumerate(proposals):
        suffix, slice_metadata = action_phase_suffix(
            demo,
            maximum_pregrasp_lead_frames=int(
                phase_cfg["maximum_pregrasp_lead_frames"]
            ),
            minimum_anchor_prefix_frames=int(
                phase_cfg["minimum_anchor_prefix_frames"]
            ),
            gripper_close_threshold=float(
                phase_cfg["gripper_close_threshold"]
            ),
        )
        controller.reset_layout(
            LAYOUT_INIT_STATE_IDS[layout],
            int(config["environment"]["reset_seed"]),
        )
        initial_bowl_position = controller.bowl_position()
        for action in demo.actions[: slice_metadata["prefix_action_count"]]:
            _, _, done, _ = controller.step(action)
            if done:
                raise RuntimeError(
                    f"Episode {demo.episode_index} terminated before its "
                    f"{goal} action-phase anchor in layout {layout}"
                )
        goals = evaluate_common_goals(environment)
        bowl_drift = float(
            np.linalg.norm(controller.bowl_position() - initial_bowl_position)
        )
        if any(goals.values()) or controller.bowl_grasped():
            raise RuntimeError(
                f"Episode {demo.episode_index} has an invalid {goal} action-phase "
                f"anchor in layout {layout}: goals={goals}, "
                f"grasped={controller.bowl_grasped()}"
            )
        if bowl_drift > float(phase_cfg["bowl_drift_tolerance_m"]):
            raise RuntimeError(
                f"Episode {demo.episode_index} moved the bowl before its "
                f"{goal} action-phase anchor in layout {layout}: {bowl_drift:.6f} m"
            )
        anchor_position = controller.eef_position()
        anchor_orientation = controller.eef_orientation()
        metadata = {
            "proposal_index": proposal_index,
            "goal": demo.goal,
            "episode_index": demo.episode_index,
            "task_index": demo.task_index,
            "source_frame_count": int(len(demo.actions)),
            "source_action_sha256": demo.action_sha256,
            "execution_mode": phase_cfg["execution_mode"],
            "anchor_rule": phase_cfg["anchor_rule"],
            "maximum_pregrasp_lead_frames": int(
                phase_cfg["maximum_pregrasp_lead_frames"]
            ),
            "minimum_anchor_prefix_frames": int(
                phase_cfg["minimum_anchor_prefix_frames"]
            ),
            "gripper_close_threshold": float(
                phase_cfg["gripper_close_threshold"]
            ),
            "layout": layout,
            "init_state_id": LAYOUT_INIT_STATE_IDS[layout],
            **slice_metadata,
            "reference_drawer_joint": controller.top_drawer_position(),
            "reference_bowl_drift_m": bowl_drift,
            "anchor_eef_position": anchor_position.tolist(),
            "anchor_eef_orientation": anchor_orientation.tolist(),
            "reference_goals": goals,
            "reference_nonterminal": True,
            "reference_bowl_grasped": False,
        }
        entries.append(
            ActionPhaseProposal(
                source=demo,
                suffix=suffix,
                anchor_position=anchor_position,
                anchor_orientation=anchor_orientation,
                metadata=metadata,
            )
        )
    expected_indices = list(range(len(entries)))
    if [entry.metadata["proposal_index"] for entry in entries] != expected_indices:
        raise RuntimeError("Action-phase proposal bank order changed")
    return tuple(entries)


def build_landmark_registered_action_phase_proposal_bank(
    environment,
    *,
    target_layout: str,
    proposals: tuple[DemoTrace, ...],
    config: dict[str, Any],
) -> tuple[ActionPhaseProposal, ...]:
    """Register one canonical layout's phase anchors to a target bowl landmark.

    Only translation is transferred.  The source-action slice and orientation
    come from the declared reference layout, while the anchor position preserves
    its reference EEF-to-bowl offset at the target layout's bowl position.
    """

    registration = config["action_phase_oracle"].get("registration", {})
    expected = {
        "type": "translation_only",
        "reference_layout": "A",
        "landmark": BOWL_NAME,
    }
    if any(registration.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "Landmark-registered action phases require the locked layout-A "
            "bowl-translation contract"
        )
    reference_layout = str(registration["reference_layout"])
    if target_layout not in LAYOUT_INIT_STATE_IDS:
        raise ValueError(f"Unknown registered target layout: {target_layout}")
    reference_bank = build_action_phase_proposal_bank(
        environment,
        layout=reference_layout,
        proposals=proposals,
        config=config,
    )
    controller = PolicyFreeController(environment)
    controller.reset_layout(
        LAYOUT_INIT_STATE_IDS[reference_layout],
        int(config["environment"]["reset_seed"]),
    )
    reference_landmark = controller.bowl_position()
    controller.reset_layout(
        LAYOUT_INIT_STATE_IDS[target_layout],
        int(config["environment"]["reset_seed"]),
    )
    target_landmark = controller.bowl_position()
    entries = []
    for reference in reference_bank:
        anchor = landmark_registered_point(
            reference.anchor_position,
            reference_landmark,
            target_landmark,
        )
        target_neutral_metadata = {
            key: value
            for key, value in reference.metadata.items()
            if key
            not in {
                "layout",
                "init_state_id",
                "anchor_eef_position",
                "anchor_eef_orientation",
                "reference_drawer_joint",
                "reference_bowl_drift_m",
                "reference_goals",
                "reference_nonterminal",
                "reference_bowl_grasped",
            }
        }
        metadata = {
            **target_neutral_metadata,
            "execution_mode": config["action_phase_oracle"]["execution_mode"],
            "anchor_rule": config["action_phase_oracle"]["anchor_rule"],
            "layout": target_layout,
            "init_state_id": LAYOUT_INIT_STATE_IDS[target_layout],
            "anchor_eef_position": anchor.tolist(),
            "anchor_eef_orientation": reference.anchor_orientation.tolist(),
            "canonical_reference": reference.metadata,
            "landmark_registration": {
                "type": registration["type"],
                "landmark": registration["landmark"],
                "reference_layout": reference_layout,
                "target_layout": target_layout,
                "reference_landmark_position": reference_landmark.tolist(),
                "target_landmark_position": target_landmark.tolist(),
                "reference_anchor_position": reference.anchor_position.tolist(),
                "translation_m": (target_landmark - reference_landmark).tolist(),
                "translation_norm_m": float(
                    np.linalg.norm(target_landmark - reference_landmark)
                ),
                "orientation_transform": "none",
                "target_landmark_tolerance_m": float(
                    registration["target_landmark_tolerance_m"]
                ),
            },
        }
        root_binding = config["action_phase_oracle"].get(
            "root_landmark_binding"
        )
        if root_binding is not None:
            metadata["root_landmark_binding"] = root_binding
            metadata["root_landmark_tolerance_m"] = float(
                config["oracle"]["normalized_bowl_position_tolerance_m"]
                if root_binding == "normalized_bowl_translation_v1"
                else registration["target_landmark_tolerance_m"]
            )
        entries.append(
            ActionPhaseProposal(
                source=reference.source,
                suffix=reference.suffix,
                anchor_position=anchor,
                anchor_orientation=reference.anchor_orientation.copy(),
                metadata=metadata,
            )
        )
    return tuple(entries)


def registered_root_execution_anchor(
    proposal: ActionPhaseProposal,
    actual_bowl_position: np.ndarray,
    *,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Bind a nominal registered anchor to the bowl in the normalized root."""

    registration = proposal.metadata.get("landmark_registration")
    if not isinstance(registration, dict):
        raise ValueError("Registered execution anchor has no landmark contract")
    expected = np.asarray(
        registration.get("target_landmark_position"), dtype=np.float64
    )
    actual = np.asarray(actual_bowl_position, dtype=np.float64)
    nominal_anchor = np.asarray(proposal.anchor_position, dtype=np.float64)
    if (
        expected.shape != (3,)
        or actual.shape != (3,)
        or nominal_anchor.shape != (3,)
        or not np.isfinite(expected).all()
        or not np.isfinite(actual).all()
        or not np.isfinite(nominal_anchor).all()
    ):
        raise ValueError("Registered execution anchor contains invalid vectors")
    binding = proposal.metadata.get(
        "root_landmark_binding", "nominal_target_landmark_v1"
    )
    residual = actual - expected
    residual_norm = float(np.linalg.norm(residual))
    if binding == "nominal_target_landmark_v1":
        tolerance = float(registration["target_landmark_tolerance_m"])
        executed_anchor = nominal_anchor.copy()
    elif binding == "normalized_bowl_translation_v1":
        configured_tolerance = float(
            config["oracle"]["normalized_bowl_position_tolerance_m"]
        )
        declared_tolerance = proposal.metadata.get(
            "root_landmark_tolerance_m"
        )
        if declared_tolerance is None or not np.isclose(
            float(declared_tolerance),
            configured_tolerance,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(
                "Registered normalized-root tolerance changed"
            )
        tolerance = configured_tolerance
        executed_anchor = nominal_anchor + residual
    else:
        raise ValueError(f"Unknown root-landmark binding mode: {binding}")
    if (
        not np.isfinite(tolerance)
        or tolerance <= 0.0
        or residual_norm > tolerance
    ):
        raise RuntimeError(
            "Registered normalized-root landmark exceeds its binding "
            f"tolerance: residual={residual_norm:.9f} m, "
            f"tolerance={tolerance:.9f} m, mode={binding}"
        )
    return executed_anchor, {
        "mode": binding,
        "expected_target_landmark_position": expected.tolist(),
        "observed_normalized_bowl_position": actual.tolist(),
        "normalized_bowl_residual_m": residual.tolist(),
        "normalized_bowl_residual_norm_m": residual_norm,
        "tolerance_m": tolerance,
        "nominal_anchor_position": nominal_anchor.tolist(),
        "executed_anchor_position": executed_anchor.tolist(),
    }


def run_registered_grasp_acquisition(
    controller: PolicyFreeController,
    *,
    spec: StageACandidateSpec,
    proposal: ActionPhaseProposal,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Acquire the bowl after drawer opening without replaying a placement tail."""

    construction_cfg = config["construction"]
    if (
        spec.drawer_aperture != "open"
        or spec.possession != "grasped"
        or proposal.source.episode_index
        != int(construction_cfg["registered_grasp_acquisition_episode_index"])
        or proposal.source.task_index
        != int(config["oracle_proposals"]["cabinet"]["task_index"])
        or proposal.metadata.get("execution_mode")
        != "action_intrinsic_pregrasp_bowl_registered_v1"
        or proposal.metadata.get("anchor_rule")
        != "canonical_layout_a_pregrasp_anchor_translated_by_bowl_landmark"
        or proposal.metadata.get("layout") != spec.layout
        or proposal.metadata.get("init_state_id") != spec.init_state_id
        or proposal.metadata.get("landmark_registration", {}).get("target_layout")
        != spec.layout
        or bool(construction_cfg["registered_grasp_bridge_pad_to_budget"])
    ):
        raise ValueError(
            f"Registered grasp-acquisition contract changed for {spec.candidate_id}"
        )
    if (
        controller.top_drawer_position()
        > float(construction_cfg["open_drawer_threshold"])
        or controller.bowl_grasped()
        or any(evaluate_common_goals(controller.environment).values())
    ):
        raise RuntimeError(
            f"Registered grasp acquisition received an invalid root for "
            f"{spec.candidate_id}"
        )
    registration = proposal.metadata["landmark_registration"]
    initial_bowl_position = controller.bowl_position()
    expected_landmark = np.asarray(
        registration["target_landmark_position"], dtype=np.float64
    )
    if np.linalg.norm(initial_bowl_position - expected_landmark) > float(
        registration["target_landmark_tolerance_m"]
    ):
        raise RuntimeError(
            f"Registered grasp landmark changed for {spec.candidate_id}"
        )

    phase_cfg = config["action_phase_oracle"]
    bridge_start = len(controller.actions)
    route = grasped_root_transit_plan(
        controller.eef_position(),
        proposal.anchor_position,
        clearance_margin_m=float(phase_cfg["clearance_margin_m"]),
        workspace_bounds=phase_cfg["workspace_bounds_m"],
        phase_budgets=phase_cfg["bridge_phase_budgets"],
    )
    bridge_phases = []
    for phase in route:
        intermediate = phase["phase"] != "target_descent"
        result = controller.servo(
            target_position=phase["target_position"],
            target_orientation=proposal.anchor_orientation,
            gripper=float(phase_cfg["gripper_action"]),
            budget=int(phase["budget"]),
            max_translation_action=float(phase_cfg["max_translation_action"]),
            position_tolerance_m=float(
                phase_cfg[
                    "intermediate_tolerance_m"
                    if intermediate
                    else "final_tolerance_m"
                ]
            ),
            orientation_tolerance_rad=float(
                phase_cfg["orientation_tolerance_rad"]
            ),
            pad_to_budget=False,
        )
        _validate_servo_phase(
            result,
            candidate_id=spec.candidate_id,
            phase=f"registered_grasp_bridge_{phase['phase']}",
        )
        bridge_phases.append(
            {
                "phase": phase["phase"],
                "target_position": np.asarray(
                    phase["target_position"]
                ).tolist(),
                "intermediate": intermediate,
                "result": result,
            }
        )
    bridge_stop = len(controller.actions)
    bridge_bowl_drift = float(
        np.linalg.norm(controller.bowl_position() - initial_bowl_position)
    )
    if (
        bridge_stop <= bridge_start
        or any(controller.grasp_values[bridge_start:bridge_stop])
        or any(controller.done_values[bridge_start:bridge_stop])
        or any(
            any(item.values())
            for item in controller.goal_values[bridge_start:bridge_stop]
        )
        or bridge_bowl_drift > float(phase_cfg["bowl_drift_tolerance_m"])
        or controller.top_drawer_position()
        > float(construction_cfg["open_drawer_threshold"])
    ):
        raise RuntimeError(
            f"Registered grasp bridge changed the task state for "
            f"{spec.candidate_id}"
        )

    required_streak = int(
        construction_cfg["construction_grasp_stability_steps"]
    )
    acquisition_start = len(controller.actions)
    grasp_streak = 0
    acquired_at_frame = None
    for frame_index, action in zip(
        proposal.suffix.frame_indices,
        proposal.suffix.actions,
        strict=True,
    ):
        _, _, done, _ = controller.step(action)
        if done or any(evaluate_common_goals(controller.environment).values()):
            raise RuntimeError(
                f"Registered grasp acquisition crossed a terminal or goal for "
                f"{spec.candidate_id}"
            )
        grasp_streak = grasp_streak + 1 if controller.bowl_grasped() else 0
        if grasp_streak >= required_streak:
            acquired_at_frame = int(frame_index)
            break
    acquisition_stop = len(controller.actions)
    if acquired_at_frame is None or not controller.bowl_grasped():
        raise RuntimeError(
            f"Registered continuation did not acquire the bowl for "
            f"{spec.candidate_id}"
        )
    bridge_actions = np.stack(controller.actions[bridge_start:bridge_stop])
    acquisition_actions = np.stack(
        controller.actions[acquisition_start:acquisition_stop]
    )
    return {
        "mode": "registered_cabinet_phase_until_stable_grasp_v1",
        "source_episode_index": proposal.source.episode_index,
        "source_task_index": proposal.source.task_index,
        "source_action_sha256": proposal.source.action_sha256,
        "phase_proposal": proposal.metadata,
        "required_stable_grasp_steps": required_streak,
        "acquired_at_source_frame": acquired_at_frame,
        "bridge": {
            "mode": "early_stop_three_leg_clearance_route",
            "phases": bridge_phases,
            "budgeted_action_steps": int(
                sum(int(item["budget"]) for item in route)
            ),
            "executed_action_steps": int(len(bridge_actions)),
            "active_action_steps": int(
                sum(
                    int(item["result"]["active_action_steps"])
                    for item in bridge_phases
                )
            ),
            "action_sha256": _action_sha256(bridge_actions),
            "bowl_drift_m": bridge_bowl_drift,
            "drawer_aperture_preserved": True,
            "goal_ever": False,
            "done_ever": False,
            "grasp_ever": False,
        },
        "executed_source_action_steps": int(len(acquisition_actions)),
        "executed_source_action_sha256": _action_sha256(
            acquisition_actions
        ),
        "final_eef_minus_bowl_m": (
            controller.eef_position() - controller.bowl_position()
        ).tolist(),
        "final_drawer_joint": controller.top_drawer_position(),
        "final_goals": evaluate_common_goals(controller.environment),
        "final_grasped": controller.bowl_grasped(),
    }


def grasped_root_recovery_plan(
    root_transit_positions: np.ndarray,
    lifted_eef_position: np.ndarray,
    *,
    phase_budgets: dict[str, int],
) -> tuple[dict[str, Any], ...]:
    positions = np.asarray(root_transit_positions, dtype=np.float64)
    lifted = np.asarray(lifted_eef_position, dtype=np.float64)
    if (
        positions.shape != (3, 3)
        or lifted.shape != (3,)
        or not np.isfinite(positions).all()
        or not np.isfinite(lifted).all()
    ):
        raise ValueError(
            "Grasped-root recovery requires three finite transit positions"
        )
    names = ("clearance_lift", "clearance_transit", "target_descent")
    if set(phase_budgets) != set(names) or any(
        int(phase_budgets[name]) < 1 for name in names
    ):
        raise ValueError("Grasped-root recovery phase budgets are incomplete")
    return (
        {
            "phase": "reverse_target_descent",
            "target_position": positions[1].copy(),
            "budget": int(phase_budgets["target_descent"]),
            "intermediate": True,
        },
        {
            "phase": "reverse_clearance_transit",
            "target_position": positions[0].copy(),
            "budget": int(phase_budgets["clearance_transit"]),
            "intermediate": True,
        },
        {
            "phase": "reverse_clearance_lift",
            "target_position": lifted.copy(),
            "budget": int(phase_budgets["clearance_lift"]),
            "intermediate": False,
        },
    )


def _support_geometry(
    controller: PolicyFreeController,
    spec: StageACandidateSpec,
    config: dict[str, Any],
    recovery_center: np.ndarray,
) -> dict[str, Any]:
    drawer_goal = controller.site_position(DRAWER_GOAL_SITE)
    cabinet_goal = controller.site_position(CABINET_GOAL_SITE)
    handle_name, drawer_anchor = controller.top_drawer_handle_position()
    anchor = drawer_anchor if spec.transit_locus == "drawer_side" else cabinet_goal
    offset = np.asarray(
        config["construction"]["demonstration_near_offsets_m"][spec.transit_locus],
        dtype=np.float64,
    )
    scripted_waypoint = anchor + offset
    balanced = recovery_balanced_goal_axis_point(
        scripted_waypoint,
        drawer_goal,
        cabinet_goal,
        recovery_center,
        maximum_angle_degrees=float(
            config["construction"]["maximum_transverse_rotation_degrees"]
        ),
        target_recovery_mismatch=float(
            config["construction"]["target_planned_recovery_mismatch"]
        ),
    )
    near_point = scripted_waypoint
    low_point = np.asarray(balanced["point"])
    if float(balanced["pair_separation_m"]) < float(
        config["construction"]["minimum_support_pair_separation_m"]
    ):
        raise ValueError(
            "Recovery-balanced support pair is below minimum separation: "
            f"{balanced['pair_separation_m']}"
        )
    planned_point = (
        near_point
        if spec.support_stratum == "demonstration_near"
        else low_point
    )
    for point in (near_point, low_point):
        _workspace_check(point, config["construction"]["workspace_bounds_m"])
    return {
        "drawer_goal_position": drawer_goal,
        "cabinet_goal_position": cabinet_goal,
        "drawer_handle_geom": handle_name,
        "drawer_anchor_position": drawer_anchor,
        "locus_anchor_position": anchor,
        "scripted_waypoint": scripted_waypoint,
        "demonstration_near_point": near_point,
        "transverse_low_support_point": low_point,
        "support_construction": "recovery_balanced_goal_axis_rotation",
        "maximum_transverse_rotation_degrees": float(
            config["construction"]["maximum_transverse_rotation_degrees"]
        ),
        "selected_transverse_rotation_degrees": float(
            balanced["selected_angle_degrees"]
        ),
        "target_planned_recovery_mismatch": float(
            config["construction"]["target_planned_recovery_mismatch"]
        ),
        "selected_planned_recovery_mismatch": float(
            balanced["planned_recovery_mismatch"]
        ),
        "planned_controlled_point": planned_point,
        "support_pair_separation_m": float(balanced["pair_separation_m"]),
        "recovery_center_controlled_point": np.asarray(recovery_center),
        "planned_recovery_distance_m": float(
            np.linalg.norm(planned_point - recovery_center)
        ),
        "planned_goal_distances_m": goal_distances(
            planned_point,
            drawer_goal=drawer_goal,
            cabinet_goal=cabinet_goal,
        ),
    }


def _drawer_aperture_label(
    drawer_joint: float, construction_cfg: dict[str, Any]
) -> str:
    if abs(drawer_joint) <= float(construction_cfg["closed_drawer_tolerance"]):
        return "closed"
    if drawer_joint <= float(construction_cfg["open_drawer_threshold"]):
        return "open"
    return "transition"


def _physical_transit_locus(
    controller: PolicyFreeController, controlled_point: np.ndarray
) -> str:
    _, drawer_anchor = controller.top_drawer_handle_position()
    cabinet_anchor = controller.site_position(CABINET_GOAL_SITE)
    return (
        "drawer_side"
        if np.linalg.norm(controlled_point - drawer_anchor)
        <= np.linalg.norm(controlled_point - cabinet_anchor)
        else "cabinet_side"
    )


def _support_state(
    controller: PolicyFreeController,
    *,
    layout: str,
    eef_motion: np.ndarray,
    bowl_motion: np.ndarray,
    action_motion: np.ndarray,
    config: dict[str, Any],
    category_override: dict[str, str] | None = None,
) -> dict[str, Any]:
    eef_position = controller.eef_position()
    bowl_position = controller.bowl_position()
    grasped = controller.bowl_grasped()
    controlled_point = bowl_position if grasped else eef_position
    controlled_motion = bowl_motion if grasped else eef_motion
    drawer_goal = controller.site_position(DRAWER_GOAL_SITE)
    cabinet_goal = controller.site_position(CABINET_GOAL_SITE)
    nearest_anchor_locus = _physical_transit_locus(controller, controlled_point)
    categories = {
        "layout": layout,
        "drawer_aperture": _drawer_aperture_label(
            controller.top_drawer_position(), config["construction"]
        ),
        "possession": "grasped" if grasped else "on_table",
        "transit_locus": nearest_anchor_locus,
        "motion_event": (
            "stationary"
            if np.linalg.norm(controlled_motion)
            <= float(config["support_metric"]["motion_event_threshold_m"])
            else "moving"
        ),
    }
    if category_override:
        categories.update(category_override)
    return {
        **categories,
        "nearest_anchor_locus": nearest_anchor_locus,
        "eef_position": eef_position.tolist(),
        "eef_orientation": controller.eef_orientation().tolist(),
        "robot_joint_positions": controller.joint_positions().tolist(),
        "bowl_position": bowl_position.tolist(),
        "drawer_joint": controller.top_drawer_position(),
        "eef_motion": np.asarray(eef_motion, dtype=np.float64).tolist(),
        "bowl_motion": np.asarray(bowl_motion, dtype=np.float64).tolist(),
        "action_motion": np.asarray(action_motion, dtype=np.float64).tolist(),
        "grasp_relative_position": (
            (eef_position - bowl_position).tolist()
            if grasped
            else [0.0, 0.0, 0.0]
        ),
        "drawer_goal_distance": float(np.linalg.norm(controlled_point - drawer_goal)),
        "cabinet_goal_distance": float(
            np.linalg.norm(controlled_point - cabinet_goal)
        ),
    }


def build_support_reference_bank(
    environment,
    demos: dict[str, DemoTrace],
    config: dict[str, Any],
) -> SupportReferenceBank:
    entries: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for layout, init_state_id in LAYOUT_INIT_STATE_IDS.items():
        for goal, demo in sorted(demos.items()):
            if goal != demo.goal:
                raise ValueError(
                    f"Support reference role/goal mismatch: {goal} != {demo.goal}"
                )
            controller = PolicyFreeController(environment)
            controller.reset_layout(
                init_state_id, int(config["environment"]["reset_seed"])
            )
            initial = _support_state(
                controller,
                layout=layout,
                eef_motion=np.zeros(3),
                bowl_motion=np.zeros(3),
                action_motion=np.zeros(6),
                config=config,
                category_override={"transit_locus": f"{goal}_side"},
            )
            initial.update(
                {
                    "reference_id": (
                        f"layout-{layout.lower()}__goal-{goal}__"
                        f"episode-{demo.episode_index}__frame-reset"
                    ),
                    "goal": goal,
                    "demo_episode_index": demo.episode_index,
                    "demo_task_index": demo.task_index,
                    "demo_action_sha256": demo.action_sha256,
                    "frame_index": -1,
                }
            )
            entries.append(initial)
            initial_goals = evaluate_common_goals(environment)
            if any(initial_goals.values()):
                raise RuntimeError(
                    f"Support reference {goal}/{layout} starts at a goal: "
                    f"{initial_goals}"
                )
            first_goal_frame = None
            other_goal = "cabinet" if goal == "drawer" else "drawer"
            for frame_index, action in zip(
                demo.frame_indices, demo.actions, strict=True
            ):
                before_eef = controller.eef_position()
                before_bowl = controller.bowl_position()
                _, _, done, _ = controller.step(action)
                state = _support_state(
                    controller,
                    layout=layout,
                    eef_motion=controller.eef_position() - before_eef,
                    bowl_motion=controller.bowl_position() - before_bowl,
                    action_motion=np.asarray(action[:6]),
                    config=config,
                    category_override={"transit_locus": f"{goal}_side"},
                )
                state.update(
                    {
                        "reference_id": (
                            f"layout-{layout.lower()}__goal-{goal}__"
                            f"episode-{demo.episode_index}__frame-{int(frame_index):04d}"
                        ),
                        "goal": goal,
                        "demo_episode_index": demo.episode_index,
                        "demo_task_index": demo.task_index,
                        "demo_action_sha256": demo.action_sha256,
                        "frame_index": int(frame_index),
                    }
                )
                entries.append(state)
                goals = evaluate_common_goals(environment)
                if goals[other_goal]:
                    raise RuntimeError(
                        f"Support reference {goal}/{layout} reached wrong goal "
                        f"at frame {int(frame_index)}: {goals}"
                    )
                if goals[goal]:
                    first_goal_frame = int(frame_index)
                    break
                if done:
                    raise RuntimeError(
                        f"Support reference {goal}/{layout} terminated before its goal"
                    )
            if first_goal_frame is None:
                raise RuntimeError(
                    f"Support reference {goal}/{layout} never reached its goal"
                )
            coverage.append(
                {
                    "layout": layout,
                    "goal": goal,
                    "demo_episode_index": demo.episode_index,
                    "demo_action_sha256": demo.action_sha256,
                    "first_goal_frame": first_goal_frame,
                }
            )
    provenance = {
        "schema_version": 1,
        "scope": "two_locked_training_demonstrations_replayed_in_both_stage_a_layouts",
        "policy_loaded": False,
        "entry_count": len(entries),
        "coverage": coverage,
        "support_metric_config": config["support_metric"],
    }
    bank_hash = canonical_sha256({"provenance": provenance, "entries": entries})
    return SupportReferenceBank(
        entries=tuple(entries), sha256=bank_hash, provenance=provenance
    )


def measure_candidate_support(
    controller: PolicyFreeController,
    *,
    spec: StageACandidateSpec,
    bank: SupportReferenceBank,
    stable_start_eef: np.ndarray,
    stable_end_eef: np.ndarray,
    stable_start_bowl: np.ndarray,
    stable_end_bowl: np.ndarray,
    stability_steps: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    if stability_steps < 1 or not controller.actions:
        raise ValueError("Candidate support measurement requires stability actions")
    physical_query = _support_state(
        controller,
        layout=spec.layout,
        eef_motion=(stable_end_eef - stable_start_eef) / stability_steps,
        bowl_motion=(stable_end_bowl - stable_start_bowl) / stability_steps,
        action_motion=np.asarray(controller.actions[-1][:6]),
        config=config,
    )
    query = dict(physical_query)
    query.update(
        {
            "drawer_aperture": spec.drawer_aperture,
            "possession": spec.possession,
            "transit_locus": spec.transit_locus,
        }
    )
    support_cfg = config["support_metric"]
    result = measure_joint_support(
        query,
        bank.entries,
        scales=support_cfg["scales"],
        categorical_mismatch_penalty=float(
            support_cfg["categorical_mismatch_penalty"]
        ),
    )
    result.update(
        {
            "reference_bank_sha256": bank.sha256,
            "reference_scope": bank.provenance["scope"],
            "reference_coverage": bank.provenance["coverage"],
            "physical_query_categories": {
                key: physical_query[key]
                for key in (
                    "layout",
                    "drawer_aperture",
                    "possession",
                    "transit_locus",
                    "motion_event",
                )
            },
            "factor_category_matches": {
                "drawer_aperture": (
                    physical_query["drawer_aperture"] == spec.drawer_aperture
                ),
                "possession": physical_query["possession"] == spec.possession,
            },
            "locus_semantics": {
                "locked_route_locus": spec.transit_locus,
                "nearest_anchor_locus": physical_query["nearest_anchor_locus"],
                "nearest_anchor_matches_locked_route": (
                    physical_query["nearest_anchor_locus"]
                    == spec.transit_locus
                ),
            },
        }
    )
    return result


def construct_candidate(
    environment,
    spec: StageACandidateSpec,
    demos: dict[str, DemoTrace],
    config: dict[str, Any],
    support_reference_bank: SupportReferenceBank | None = None,
    registered_grasp_acquisition: ActionPhaseProposal | None = None,
) -> ConstructedCandidate:
    construction_cfg = config["construction"]
    acquisition_mode = construction_cfg.get(
        "open_grasped_acquisition_mode", "joint_drawer_construction_trace"
    )
    if acquisition_mode not in {
        "joint_drawer_construction_trace",
        "registered_cabinet_phase_v1",
    }:
        raise ValueError(
            f"Unsupported open-grasped acquisition mode: {acquisition_mode}"
        )
    uses_registered_acquisition = bool(
        spec.drawer_aperture == "open"
        and spec.possession == "grasped"
        and acquisition_mode == "registered_cabinet_phase_v1"
    )
    if (registered_grasp_acquisition is not None) != uses_registered_acquisition:
        raise ValueError(
            f"Registered grasp acquisition does not match {spec.candidate_id}"
        )
    controller = PolicyFreeController(environment)
    controller.reset_layout(
        spec.init_state_id, int(config["environment"]["reset_seed"])
    )
    initial_bowl_position = controller.bowl_position()
    initial_eef_position = controller.eef_position()
    initial_eef_orientation = controller.eef_orientation()
    initial_joint_positions = controller.joint_positions()
    prefix: dict[str, Any] | None = None
    grasp_acquisition: dict[str, Any] | None = None
    required_grasp_streak = int(
        construction_cfg["construction_grasp_stability_steps"]
    )
    if required_grasp_streak < 1:
        raise ValueError("Construction grasp-stability window must be positive")
    grasp_streak = 0

    def stable_grasp_condition(*, require_open_drawer: bool) -> bool:
        nonlocal grasp_streak
        aperture_ready = (
            not require_open_drawer
            or controller.top_drawer_position()
            <= float(construction_cfg["open_drawer_threshold"])
        )
        if aperture_ready and controller.bowl_grasped():
            grasp_streak += 1
        else:
            grasp_streak = 0
        return grasp_streak >= required_grasp_streak

    if (
        spec.drawer_aperture == "open"
        and spec.possession == "grasped"
        and acquisition_mode == "registered_cabinet_phase_v1"
    ):
        prefix = controller.replay_until(
            demos["drawer_construction"],
            condition=lambda: controller.top_drawer_position()
            <= float(construction_cfg["open_drawer_threshold"]),
        )
        if controller.bowl_grasped():
            raise RuntimeError(
                f"Drawer-only prefix unexpectedly grasped the bowl for "
                f"{spec.candidate_id}"
            )
        grasp_acquisition = run_registered_grasp_acquisition(
            controller,
            spec=spec,
            proposal=registered_grasp_acquisition,
            config=config,
        )
    elif spec.drawer_aperture == "open":
        prefix = controller.replay_until(
            demos["drawer_construction"],
            condition=lambda: (
                controller.top_drawer_position()
                <= float(construction_cfg["open_drawer_threshold"])
                and (
                    spec.possession != "grasped"
                    or stable_grasp_condition(require_open_drawer=True)
                )
            ),
        )
    elif spec.possession == "grasped":
        prefix = controller.replay_until(
            demos["grasp_construction"],
            condition=lambda: stable_grasp_condition(require_open_drawer=False),
        )

    if prefix is not None:
        prefix["required_consecutive_grasp_steps"] = (
            required_grasp_streak
            if spec.possession == "grasped"
            and grasp_acquisition is None
            else 0
        )

    goals_after_prefix = evaluate_common_goals(environment)
    prefix_goal_ever = any(
        any(goals.values()) for goals in controller.goal_values
    )
    if goals_after_prefix != {"drawer": False, "cabinet": False} or prefix_goal_ever:
        raise RuntimeError(
            f"Construction prefix reached a goal for {spec.candidate_id}: "
            f"final={goals_after_prefix}, transient={prefix_goal_ever}"
        )
    if spec.possession == "grasped" and not controller.bowl_grasped():
        raise RuntimeError(f"Construction did not grasp the bowl for {spec.candidate_id}")

    orientation = controller.eef_orientation()
    recovery_waypoints = None
    if spec.possession == "grasped":
        grasp_eef_position = controller.eef_position()
        recovery_waypoints = {
            "grasp_eef_position": grasp_eef_position,
            "lifted_eef_position": grasp_eef_position
            + np.asarray([0.0, 0.0, float(construction_cfg["safe_lift_m"])]),
            "grasp_orientation": orientation.copy(),
            "grasp_bowl_position": controller.bowl_position(),
        }
    grasp_offset = (
        controller.eef_position() - controller.bowl_position()
        if spec.possession == "grasped"
        else np.zeros(3, dtype=np.float64)
    )
    recovery_center = (
        np.asarray(recovery_waypoints["lifted_eef_position"]) - grasp_offset
        if recovery_waypoints is not None
        else initial_eef_position
    )
    support = _support_geometry(
        controller, spec, config, recovery_center=recovery_center
    )
    eef_target = support["planned_controlled_point"] + grasp_offset
    _workspace_check(eef_target, construction_cfg["workspace_bounds_m"])
    gripper = 1.0 if spec.possession == "grasped" else -1.0
    max_action = float(
        construction_cfg[
            "max_translation_action_grasped"
            if spec.possession == "grasped"
            else "max_translation_action_on_table"
        ]
    )

    safe_lift = controller.servo(
        target_position=controller.eef_position()
        + np.asarray([0.0, 0.0, float(construction_cfg["safe_lift_m"])]),
        target_orientation=orientation,
        gripper=gripper,
        budget=int(construction_cfg["safe_lift_budget"]),
        max_translation_action=max_action,
        position_tolerance_m=float(construction_cfg["root_tolerance_m"]),
        pad_to_budget=grasp_acquisition is None,
    )
    if spec.possession == "grasped":
        _validate_grasped_transport_phase(
            controller,
            safe_lift,
            candidate_id=spec.candidate_id,
            phase="construction_safe_lift",
            **_transport_validation_limits(config),
        )
    else:
        _validate_servo_phase(
            safe_lift,
            candidate_id=spec.candidate_id,
            phase="construction_safe_lift",
        )
    root_servo_action_start = len(controller.actions)
    if spec.possession == "grasped":
        transit_plan = grasped_root_transit_plan(
            controller.eef_position(),
            eef_target,
            clearance_margin_m=float(
                construction_cfg["grasped_root_clearance_margin_m"]
            ),
            workspace_bounds=construction_cfg["workspace_bounds_m"],
            phase_budgets=construction_cfg["grasped_root_transit_budgets"],
        )
        transit_records = []
        for transit_phase in transit_plan:
            is_intermediate = transit_phase["phase"] != "target_descent"
            result = controller.servo(
                target_position=transit_phase["target_position"],
                target_orientation=orientation,
                gripper=gripper,
                budget=transit_phase["budget"],
                max_translation_action=max_action,
                position_tolerance_m=float(
                    construction_cfg[
                        "grasped_root_waypoint_tolerance_m"
                        if is_intermediate
                        else "root_tolerance_m"
                    ]
                ),
            )
            _validate_grasped_transport_phase(
                controller,
                result,
                candidate_id=spec.candidate_id,
                phase=f"construction_{transit_phase['phase']}",
                **_transport_validation_limits(config),
            )
            transit_records.append(
                {
                    "phase": transit_phase["phase"],
                    "target_position": transit_phase["target_position"].tolist(),
                    "result": result,
                }
            )
        root_servo = {
            "mode": "three_leg_clearance_route",
            "clearance_margin_m": float(
                construction_cfg["grasped_root_clearance_margin_m"]
            ),
            "intermediate_waypoint_tolerance_m": float(
                construction_cfg["grasped_root_waypoint_tolerance_m"]
            ),
            "phases": transit_records,
        }
        if recovery_waypoints is None:
            raise RuntimeError("Grasped root lost its recovery-waypoint contract")
        recovery_waypoints["root_transit_positions"] = np.stack(
            [phase["target_position"] for phase in transit_plan]
        )
    else:
        root_servo = controller.servo(
            target_position=eef_target,
            target_orientation=orientation,
            gripper=gripper,
            budget=int(construction_cfg["root_servo_budget"]),
            max_translation_action=max_action,
            position_tolerance_m=float(construction_cfg["root_tolerance_m"]),
        )
        _validate_servo_phase(
            root_servo,
            candidate_id=spec.candidate_id,
            phase="construction_root_servo",
        )
    root_servo_actions = np.stack(controller.actions[root_servo_action_start:])
    if len(root_servo_actions) != int(construction_cfg["root_servo_budget"]):
        raise RuntimeError(
            f"Root transit action count mismatch for {spec.candidate_id}: "
            f"{len(root_servo_actions)} != {construction_cfg['root_servo_budget']}"
        )

    final_timestep = int(construction_cfg["root_final_timestep"])
    stability_steps = int(construction_cfg["stability_steps"])
    padding_steps = final_timestep - stability_steps - int(controller.problem.timestep)
    if padding_steps < 0:
        raise RuntimeError(
            f"Construction exceeded normalized timestep for {spec.candidate_id}: "
            f"pre_padding_timestep={controller.problem.timestep}, "
            f"final_timestep={final_timestep}, "
            f"reserved_stability_steps={stability_steps}, "
            f"padding_steps={padding_steps}"
        )
    padding = controller.servo(
        target_position=eef_target,
        target_orientation=orientation,
        gripper=gripper,
        budget=max(padding_steps, 1),
        max_translation_action=max_action,
        position_tolerance_m=float(construction_cfg["root_tolerance_m"]),
    )
    if padding_steps == 0:
        raise RuntimeError("Stage A root timestep leaves no explicit padding action")
    if spec.possession == "grasped":
        _validate_grasped_transport_phase(
            controller,
            padding,
            candidate_id=spec.candidate_id,
            phase="construction_padding",
            **_transport_validation_limits(config),
        )
    else:
        _validate_servo_phase(
            padding,
            candidate_id=spec.candidate_id,
            phase="construction_padding",
        )

    stable_start_bowl = controller.bowl_position()
    stable_start_eef = controller.eef_position()
    grasp_checks = []
    for _ in range(stability_steps):
        stability = controller.servo(
            target_position=eef_target,
            target_orientation=orientation,
            gripper=gripper,
            budget=1,
            max_translation_action=max_action,
            position_tolerance_m=float(construction_cfg["root_tolerance_m"]),
        )
        if spec.possession == "grasped":
            _validate_grasped_transport_phase(
                controller,
                stability,
                candidate_id=spec.candidate_id,
                phase="construction_stability",
                **_transport_validation_limits(config),
            )
        else:
            _validate_servo_phase(
                stability,
                candidate_id=spec.candidate_id,
                phase="construction_stability",
            )
        grasp_checks.append(controller.bowl_grasped())
    stable_end_bowl = controller.bowl_position()
    stable_end_eef = controller.eef_position()
    if int(controller.problem.timestep) != final_timestep:
        raise RuntimeError(
            f"Root timestep normalization failed for {spec.candidate_id}: "
            f"{controller.problem.timestep} != {final_timestep}"
        )

    snapshot = capture_libero_state(environment)
    realized_point = (
        stable_end_bowl
        if spec.possession == "grasped"
        else stable_end_eef
    )
    root_geometry = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in support.items()
    }
    root_geometry.update(
        {
            "controlled_point_kind": (
                "bowl_position" if spec.possession == "grasped" else "eef_position"
            ),
            "realized_controlled_point": realized_point.tolist(),
            "realized_goal_distances_m": goal_distances(
                realized_point,
                drawer_goal=np.asarray(support["drawer_goal_position"]),
                cabinet_goal=np.asarray(support["cabinet_goal_position"]),
            ),
            "object_goal_distances_m": goal_distances(
                stable_end_bowl,
                drawer_goal=np.asarray(support["drawer_goal_position"]),
                cabinet_goal=np.asarray(support["cabinet_goal_position"]),
            ),
            "realized_recovery_distance_m": float(
                np.linalg.norm(
                    realized_point
                    - np.asarray(support["recovery_center_controlled_point"])
                )
            ),
            "eef_position": stable_end_eef.tolist(),
            "bowl_position": stable_end_bowl.tolist(),
            "eef_minus_bowl": (stable_end_eef - stable_end_bowl).tolist(),
            "top_drawer_joint": controller.top_drawer_position(),
            "scripted_path_distance_m": float(
                np.linalg.norm(
                    realized_point - np.asarray(support["scripted_waypoint"])
                )
            ),
        }
    )
    root_validation = _validate_root(
        controller=controller,
        spec=spec,
        snapshot=snapshot,
        initial_bowl_position=initial_bowl_position,
        stable_start_bowl=stable_start_bowl,
        stable_end_bowl=stable_end_bowl,
        stable_start_eef=stable_start_eef,
        stable_end_eef=stable_end_eef,
        grasp_checks=grasp_checks,
        config=config,
    )
    support_measurement = (
        measure_candidate_support(
            controller,
            spec=spec,
            bank=support_reference_bank,
            stable_start_eef=stable_start_eef,
            stable_end_eef=stable_end_eef,
            stable_start_bowl=stable_start_bowl,
            stable_end_bowl=stable_end_bowl,
            stability_steps=stability_steps,
            config=config,
        )
        if support_reference_bank is not None
        else None
    )
    construction_actions = np.stack(controller.actions)
    construction = {
        "mode": "current_process_policy_independent_script",
        "prefix": prefix,
        "grasp_acquisition": grasp_acquisition,
        "safe_lift": safe_lift,
        "root_servo": root_servo,
        "root_transit_action_count": int(len(root_servo_actions)),
        "root_transit_action_sha256": _action_sha256(root_servo_actions),
        "padding": padding,
        "action_count": int(len(construction_actions)),
        "action_sha256": _action_sha256(construction_actions),
        "eef_path_length_m": controller.eef_path_length_m,
        "control_effort": controller.control_effort,
        "motion_control_effort": controller.motion_control_effort,
        "final_timestep": int(controller.problem.timestep),
        "recovery_waypoints": (
            {
                key: value.tolist()
                for key, value in recovery_waypoints.items()
            }
            if recovery_waypoints is not None
            else None
        ),
    }
    return ConstructedCandidate(
        snapshot=snapshot,
        initial_bowl_position=initial_bowl_position,
        initial_eef_position=initial_eef_position,
        initial_eef_orientation=initial_eef_orientation,
        initial_joint_positions=initial_joint_positions,
        recovery_waypoints=recovery_waypoints,
        support_measurement=support_measurement,
        construction=construction,
        root_validation=root_validation,
        root_geometry=root_geometry,
    )


def _validate_root(
    *,
    controller: PolicyFreeController,
    spec: StageACandidateSpec,
    snapshot: LiberoStateSnapshot,
    initial_bowl_position: np.ndarray,
    stable_start_bowl: np.ndarray,
    stable_end_bowl: np.ndarray,
    stable_start_eef: np.ndarray,
    stable_end_eef: np.ndarray,
    grasp_checks: list[bool],
    config: dict[str, Any],
) -> dict[str, Any]:
    construction_cfg = config["construction"]
    validation_cfg = config["validation"]
    goals = evaluate_common_goals(controller.environment)
    drawer_position = controller.top_drawer_position()
    drawer_match = (
        abs(drawer_position) <= float(construction_cfg["closed_drawer_tolerance"])
        if spec.drawer_aperture == "closed"
        else drawer_position <= float(construction_cfg["open_drawer_threshold"])
    )
    grasped = BOWL_NAME in snapshot.grasped_objects
    possession_match = grasped == (spec.possession == "grasped")
    bowl_position_drift = float(np.linalg.norm(stable_end_bowl - stable_start_bowl))
    relative_pose_drift = float(
        np.linalg.norm(
            (stable_end_eef - stable_end_bowl)
            - (stable_start_eef - stable_start_bowl)
        )
    )
    min_contact_distance = min(
        (float(contact["distance"]) for contact in snapshot.contacts), default=0.0
    )
    deep_contacts = [
        contact
        for contact in snapshot.contacts
        if float(contact["distance"])
        < -float(validation_cfg["penetration_limit_m"])
    ]
    robot_tokens = ("robot", "gripper", "panda", "finger", "hand")
    robot_fixture_contacts = [
        contact
        for contact in snapshot.contacts
        if (
            any(token in str(contact["geom1"]).lower() for token in robot_tokens)
            and "wooden_cabinet" in str(contact["geom2"]).lower()
        )
        or (
            any(token in str(contact["geom2"]).lower() for token in robot_tokens)
            and "wooden_cabinet" in str(contact["geom1"]).lower()
        )
    ]
    stability_pass = (
        relative_pose_drift
        <= float(validation_cfg["stable_relative_pose_drift_m"])
        if spec.possession == "grasped"
        else bowl_position_drift
        <= float(validation_cfg["stable_position_drift_m"])
    )
    grasp_stability_pass = (
        all(grasp_checks) if spec.possession == "grasped" else not any(grasp_checks)
    )
    on_table_displacement = float(
        np.linalg.norm(stable_end_bowl - initial_bowl_position)
    )
    done = bool(getattr(controller.problem, "done", False))
    construction_done_ever = any(controller.done_values)
    construction_goal_ever = any(
        any(values.values()) for values in controller.goal_values
    )
    passed = bool(
        goals == {"drawer": False, "cabinet": False}
        and drawer_match
        and possession_match
        and stability_pass
        and grasp_stability_pass
        and not deep_contacts
        and not robot_fixture_contacts
        and not done
        and not construction_done_ever
        and not construction_goal_ever
        and not snapshot.success
    )
    result = {
        "pass": passed,
        "goals": goals,
        "drawer_aperture_match": drawer_match,
        "possession_match": possession_match,
        "stability_pass": stability_pass,
        "grasp_stability_pass": grasp_stability_pass,
        "bowl_position_drift_m": bowl_position_drift,
        "eef_bowl_relative_pose_drift_m": relative_pose_drift,
        "on_table_displacement_from_initial_m": on_table_displacement,
        "minimum_contact_distance_m": min_contact_distance,
        "deep_contact_count": len(deep_contacts),
        "robot_fixture_contact_count": len(robot_fixture_contacts),
        "contact_count": len(snapshot.contacts),
        "done": done,
        "construction_done_ever": construction_done_ever,
        "construction_goal_ever": construction_goal_ever,
        "native_success": snapshot.success,
    }
    if not passed:
        raise RuntimeError(f"Root validation failed for {spec.candidate_id}: {result}")
    return result


def certify_computational_state(
    environment,
    snapshot: LiberoStateSnapshot,
    *,
    possession: str,
    probe_actions: list[list[float]],
) -> dict[str, Any]:
    round_trip = validate_libero_round_trip(environment, snapshot)
    gripper = 1.0 if possession == "grasped" else -1.0
    actions = []
    if not probe_actions:
        raise ValueError("Stage A certificate requires at least one probe action")
    for values in probe_actions:
        if len(values) != 6:
            raise ValueError("Stage A certificate actions must have six non-gripper values")
        action = np.asarray([*values, gripper], dtype=np.float32)
        if not np.isfinite(action).all():
            raise ValueError("Stage A certificate actions must be finite")
        actions.append(action)

    def run_once() -> dict[str, Any]:
        controller = PolicyFreeController(environment)
        observation = None
        rewards = []
        done_values = []
        for action in actions:
            observation, reward, done, _ = controller.step(action)
            rewards.append(reward)
            done_values.append(done)
        final = capture_libero_state(environment)
        return {
            "snapshot": final,
            "mujoco_state": final.mujoco_state,
            "observation": observation,
            "rewards": rewards,
            "done": done_values,
            "goals": evaluate_common_goals(environment),
        }

    restore_libero_state(environment, snapshot)
    first = run_once()
    restore_libero_state(environment, snapshot)
    second = run_once()
    differences = nested_field_max_abs_differences(
        first["observation"], second["observation"]
    )
    state_difference = float(
        np.max(
            np.abs(first["mujoco_state"] - second["mujoco_state"]), initial=0.0
        )
    )
    first_full_state_sha256 = snapshot_sha256(first["snapshot"])
    second_full_state_sha256 = snapshot_sha256(second["snapshot"])
    full_state_match = first_full_state_sha256 == second_full_state_sha256
    passed = bool(
        round_trip
        and certificate_within_tolerance(state_difference, differences)
        and full_state_match
        and first["rewards"] == second["rewards"]
        and first["done"] == second["done"]
        and first["goals"] == second["goals"]
        and len(actions) == len(probe_actions)
    )
    restore_libero_state(environment, snapshot)
    return {
        "pass": passed,
        "probe_action_count": len(actions),
        "probe_action_sha256": _action_sha256(actions),
        "round_trip": round_trip,
        "max_abs_mujoco_state_diff": state_difference,
        "first_full_state_sha256": first_full_state_sha256,
        "second_full_state_sha256": second_full_state_sha256,
        "full_state_match": full_state_match,
        "observation_field_max_abs_diff": differences,
        "max_abs_observation_diff": max(differences.values(), default=0.0),
        "rewards_match": first["rewards"] == second["rewards"],
        "done_match": first["done"] == second["done"],
        "goals_match": first["goals"] == second["goals"],
        "tolerances": {
            "mujoco_state_atol": MUJOCO_STATE_ATOL,
            "numeric_observation_atol": NUMERIC_OBSERVATION_ATOL,
            "pixel_observation_atol": PIXEL_OBSERVATION_ATOL,
        },
    }


def _validate_servo_phase(
    result: dict[str, Any], *, candidate_id: str, phase: str
) -> None:
    if not bool(result.get("pass", False)):
        raise RuntimeError(
            f"Servo phase {phase} failed for {candidate_id}: {result}"
        )
    if result.get("nonterminal_every_step") is not True:
        raise RuntimeError(
            f"Servo phase {phase} became terminal for {candidate_id}: {result}"
        )
    if result.get("no_goal_every_step") is not True:
        raise RuntimeError(
            f"Servo phase {phase} crossed a goal for {candidate_id}: {result}"
        )


def _validate_grasped_transport_phase(
    controller: PolicyFreeController,
    result: dict[str, Any],
    *,
    candidate_id: str,
    phase: str,
    max_grasp_dropout_steps: int = 0,
    max_relative_pose_deviation_m: float = 0.012,
) -> None:
    _validate_servo_phase(result, candidate_id=candidate_id, phase=phase)
    if result.get("grasp_preserved_every_step") is not True:
        longest_dropout = int(
            result.get("max_consecutive_grasp_dropout_steps", 10**9)
        )
        relative_deviation = float(
            result.get("max_grasp_relative_pose_deviation_m", np.inf)
        )
        if (
            longest_dropout > max_grasp_dropout_steps
            or relative_deviation > max_relative_pose_deviation_m
        ):
            raise RuntimeError(
                f"Servo phase {phase} transiently lost the bowl for "
                f"{candidate_id}: {result}"
            )
    if not controller.bowl_grasped():
        raise RuntimeError(
            f"Servo phase {phase} lost the bowl for {candidate_id}: {result}"
        )


def replay_goal_proposal(
    environment,
    *,
    goal: str,
    demo: DemoTrace,
    controller: PolicyFreeController | None = None,
) -> tuple[PolicyFreeController, dict[str, Any]]:
    if goal != demo.goal:
        raise ValueError(f"Oracle/demo goal mismatch: {goal} != {demo.goal}")
    if controller is None:
        controller = PolicyFreeController(environment)
    goal_ever = False
    first_goal_frame = None
    wrong_goal_ever = False
    unexpected_done_before_goal = False
    other_goal = "cabinet" if goal == "drawer" else "drawer"
    for frame_index, action in zip(demo.frame_indices, demo.actions, strict=True):
        _, _, done, _ = controller.step(action)
        goals = evaluate_common_goals(environment)
        if goals[goal] and first_goal_frame is None:
            first_goal_frame = int(frame_index)
        goal_ever = goal_ever or goals[goal]
        wrong_goal_ever = wrong_goal_ever or goals[other_goal]
        if done and not goal_ever:
            unexpected_done_before_goal = True
            break
        if goal_ever:
            break
    final_goals = evaluate_common_goals(environment)
    return controller, {
        "goal_ever_achieved": goal_ever,
        "first_goal_demo_frame": first_goal_frame,
        "wrong_goal_ever_achieved": wrong_goal_ever,
        "unexpected_done_before_goal": unexpected_done_before_goal,
        "final_goals": final_goals,
        "pass": bool(
            goal_ever
            and final_goals[goal]
            and not wrong_goal_ever
            and not unexpected_done_before_goal
        ),
    }


def run_goal_oracle(
    environment,
    snapshot: LiberoStateSnapshot,
    *,
    spec: StageACandidateSpec,
    goal: str,
    demo: DemoTrace,
    initial_bowl_position: np.ndarray,
    initial_eef_position: np.ndarray,
    initial_eef_orientation: np.ndarray,
    initial_joint_positions: np.ndarray,
    recovery_waypoints: dict[str, np.ndarray] | None,
    config: dict[str, Any],
    raise_on_failure: bool = True,
    return_prepared_root: bool = False,
    normalization_only: bool = False,
) -> dict[str, Any]:
    if goal != demo.goal:
        raise ValueError(f"Oracle/demo goal mismatch: {goal} != {demo.goal}")
    restore_libero_state(environment, snapshot)
    controller = PolicyFreeController(environment)
    oracle_cfg = config["oracle"]
    phases: dict[str, Any] = {}
    root_goals = evaluate_common_goals(environment)
    if any(root_goals.values()):
        raise RuntimeError(
            f"Oracle root already satisfies a goal for {spec.candidate_id}: "
            f"{root_goals}"
        )
    if spec.possession == "grasped":
        if not controller.bowl_grasped():
            raise RuntimeError(f"Oracle root lost possession for {spec.candidate_id}")
        if recovery_waypoints is None:
            raise RuntimeError(
                f"Grasped root has no recovery waypoints for {spec.candidate_id}"
            )
        expected_reverse_actions = int(oracle_cfg["setdown_reverse_root_budget"])
        phase_budgets = config["construction"]["grasped_root_transit_budgets"]
        if sum(int(value) for value in phase_budgets.values()) != expected_reverse_actions:
            raise RuntimeError(
                f"Recovery route budget mismatch for {spec.candidate_id}"
            )
        root_transit_positions = np.asarray(
            recovery_waypoints.get("root_transit_positions")
        )
        orientation = np.asarray(recovery_waypoints["grasp_orientation"])
        phases["setdown_reverse_root"] = replay_grasped_recovery_route(
            controller,
            root_transit_positions=root_transit_positions,
            lifted_eef_position=np.asarray(
                recovery_waypoints["lifted_eef_position"]
            ),
            target_orientation=orientation,
            phase_budgets=phase_budgets,
            intermediate_tolerance_m=float(
                config["construction"]["grasped_root_waypoint_tolerance_m"]
            ),
            final_tolerance_m=float(oracle_cfg["servo_tolerance_m"]),
            max_translation_action=float(
                oracle_cfg["setdown_retrace_max_translation_action"]
            ),
            candidate_id=spec.candidate_id,
            **_transport_validation_limits(config),
        )
        _validate_grasped_transport_phase(
            controller,
            phases["setdown_reverse_root"],
            candidate_id=spec.candidate_id,
            phase="setdown_reverse_root",
            **_transport_validation_limits(config),
        )
        phases["setdown_return_lifted"] = controller.servo(
            target_position=np.asarray(recovery_waypoints["lifted_eef_position"]),
            target_orientation=orientation,
            gripper=1.0,
            budget=int(oracle_cfg["setdown_return_lifted_budget"]),
            max_translation_action=float(
                oracle_cfg["setdown_retrace_max_translation_action"]
            ),
            position_tolerance_m=float(oracle_cfg["servo_tolerance_m"]),
        )
        _validate_grasped_transport_phase(
            controller,
            phases["setdown_return_lifted"],
            candidate_id=spec.candidate_id,
            phase="setdown_return_lifted",
            **_transport_validation_limits(config),
        )
        phases["setdown_return_grasp"] = controller.servo(
            target_position=np.asarray(recovery_waypoints["grasp_eef_position"]),
            target_orientation=orientation,
            gripper=1.0,
            budget=int(oracle_cfg["setdown_return_grasp_budget"]),
            max_translation_action=float(
                oracle_cfg["setdown_retrace_max_translation_action"]
            ),
            position_tolerance_m=float(oracle_cfg["servo_tolerance_m"]),
        )
        _validate_grasped_transport_phase(
            controller,
            phases["setdown_return_grasp"],
            candidate_id=spec.candidate_id,
            phase="setdown_return_grasp",
            **_transport_validation_limits(config),
        )
        offset = controller.eef_position() - controller.bowl_position()
        table_safe_target = (
            np.asarray(initial_bowl_position)
            + offset
            + np.asarray(
                [0.0, 0.0, float(oracle_cfg["setdown_safe_height_m"])]
            )
        )
        phases["setdown_table_safe"] = controller.servo(
            target_position=table_safe_target,
            target_orientation=orientation,
            gripper=1.0,
            budget=int(oracle_cfg["setdown_table_safe_budget"]),
            max_translation_action=float(
                oracle_cfg["setdown_table_max_translation_action"]
            ),
            position_tolerance_m=float(oracle_cfg["servo_tolerance_m"]),
        )
        _validate_grasped_transport_phase(
            controller,
            phases["setdown_table_safe"],
            candidate_id=spec.candidate_id,
            phase="setdown_table_safe",
            **_transport_validation_limits(config),
        )
        phases["setdown_descend"] = controller.servo(
            target_position=np.asarray(initial_bowl_position)
            + offset
            + np.asarray(
                [0.0, 0.0, float(oracle_cfg["setdown_release_height_m"])]
            ),
            target_orientation=orientation,
            gripper=1.0,
            budget=int(oracle_cfg["setdown_descend_budget"]),
            max_translation_action=float(
                oracle_cfg["setdown_descend_max_translation_action"]
            ),
            position_tolerance_m=float(oracle_cfg["servo_tolerance_m"]),
        )
        _validate_grasped_transport_phase(
            controller,
            phases["setdown_descend"],
            candidate_id=spec.candidate_id,
            phase="setdown_descend",
            **_transport_validation_limits(config),
        )
        release_target = controller.eef_position()
        phases["setdown_release"] = controller.servo(
            target_position=release_target,
            target_orientation=orientation,
            gripper=-1.0,
            budget=int(oracle_cfg["setdown_release_budget"]),
            max_translation_action=float(oracle_cfg["max_translation_action"]),
            position_tolerance_m=float(oracle_cfg["servo_tolerance_m"]),
        )
        _validate_servo_phase(
            phases["setdown_release"],
            candidate_id=spec.candidate_id,
            phase="setdown_release",
        )
        phases["setdown_retreat"] = controller.servo(
            target_position=controller.eef_position()
            + np.asarray([0.0, 0.0, 0.08]),
            target_orientation=orientation,
            gripper=-1.0,
            budget=int(oracle_cfg["setdown_retreat_budget"]),
            max_translation_action=float(oracle_cfg["max_translation_action"]),
            position_tolerance_m=float(oracle_cfg["servo_tolerance_m"]),
        )
        _validate_servo_phase(
            phases["setdown_retreat"],
            candidate_id=spec.candidate_id,
            phase="setdown_retreat",
        )
        if controller.bowl_grasped():
            raise RuntimeError(f"Setdown did not release the bowl for {spec.candidate_id}")
        setdown_bowl_error = float(
            np.linalg.norm(controller.bowl_position() - initial_bowl_position)
        )
        if setdown_bowl_error > float(
            oracle_cfg["normalized_bowl_position_tolerance_m"]
        ):
            raise RuntimeError(
                f"Setdown misplaced the bowl for {spec.candidate_id}: "
                f"{setdown_bowl_error:.6f} m"
            )

    phases["home"] = controller.servo(
        target_position=np.asarray(initial_eef_position),
        target_orientation=np.asarray(initial_eef_orientation),
        gripper=-1.0,
        budget=int(oracle_cfg["home_budget"]),
        max_translation_action=float(oracle_cfg["max_translation_action"]),
        position_tolerance_m=float(oracle_cfg["servo_tolerance_m"]),
    )
    _validate_servo_phase(
        phases["home"],
        candidate_id=f"{spec.candidate_id}/{goal}",
        phase="home",
    )
    phases["home"]["joint_max_abs_error_from_layout_reset"] = float(
        np.max(
            np.abs(controller.joint_positions() - np.asarray(initial_joint_positions)),
            initial=0.0,
        )
    )

    normalization_goal_ever = any(
        any(values.values()) for values in controller.goal_values
    )
    normalization_done_ever = any(controller.done_values)
    normalized_goals = evaluate_common_goals(environment)
    normalized_bowl_position_error_m = float(
        np.linalg.norm(controller.bowl_position() - initial_bowl_position)
    )
    if normalization_goal_ever or normalization_done_ever or any(
        normalized_goals.values()
    ):
        raise RuntimeError(
            f"Oracle normalization crossed a terminal goal for "
            f"{spec.candidate_id}/{goal}: transient_goal={normalization_goal_ever}, "
            f"transient_done={normalization_done_ever}, final={normalized_goals}"
        )
    if normalized_bowl_position_error_m > float(
        oracle_cfg["normalized_bowl_position_tolerance_m"]
    ):
        raise RuntimeError(
            f"Oracle normalization bowl error for {spec.candidate_id}/{goal}: "
            f"{normalized_bowl_position_error_m:.6f} m"
        )

    normalized_snapshot = capture_libero_state(environment)
    normalization_action_count = len(controller.actions)
    normalization_action_sha256 = _action_sha256(controller.actions)
    prepared_root = PreparedOracleRoot(
        snapshot=normalized_snapshot,
        phases=phases,
        actions=tuple(action.copy() for action in controller.actions),
        eef_path_length_m=controller.eef_path_length_m,
        control_effort=controller.control_effort,
        motion_control_effort=controller.motion_control_effort,
        active_servo_steps=int(
            sum(
                int(phase.get("active_action_steps", 0))
                for phase in phases.values()
            )
        ),
        done_count=int(sum(controller.done_values)),
        normalization_goal_ever=normalization_goal_ever,
        normalization_done_ever=normalization_done_ever,
        normalized_goals=normalized_goals,
        normalized_bowl_position_error_m=normalized_bowl_position_error_m,
    )
    if normalization_only:
        normalization_actions = np.stack(controller.actions)
        result = {
            "normalization_goal_ever": normalization_goal_ever,
            "normalization_done_ever": normalization_done_ever,
            "normalized_goals": normalized_goals,
            "normalized_bowl_position_error_m": (
                normalized_bowl_position_error_m
            ),
            "normalized_state_sha256": snapshot_sha256(normalized_snapshot),
            "normalization_action_steps": normalization_action_count,
            "normalization_action_sha256": normalization_action_sha256,
            "normalization_action_path_length_m": (
                controller.eef_path_length_m
            ),
            "normalization_control_effort": controller.control_effort,
            "normalization_motion_control_effort": (
                controller.motion_control_effort
            ),
            "normalization_active_servo_steps": (
                prepared_root.active_servo_steps
            ),
            "normalization_action_trace_sha256": _action_sha256(
                normalization_actions
            ),
            "source_proposal_replayed": False,
        }
        restore_libero_state(environment, snapshot)
        if return_prepared_root:
            result["_prepared_oracle_root"] = prepared_root
        return result
    controller, proposal = replay_goal_proposal(
        environment,
        goal=goal,
        demo=demo,
        controller=controller,
    )
    goal_ever = proposal["goal_ever_achieved"]
    first_goal_frame = proposal["first_goal_demo_frame"]
    wrong_goal_ever = proposal["wrong_goal_ever_achieved"]
    unexpected_done_before_goal = proposal["unexpected_done_before_goal"]
    final_goals = proposal["final_goals"]
    all_actions = np.stack(controller.actions)
    expected_budget = int(oracle_cfg["home_budget"]) + len(demo.actions)
    if spec.possession == "grasped":
        expected_budget += sum(
            int(oracle_cfg[key])
            for key in (
                "setdown_reverse_root_budget",
                "setdown_return_lifted_budget",
                "setdown_return_grasp_budget",
                "setdown_table_safe_budget",
                "setdown_descend_budget",
                "setdown_release_budget",
                "setdown_retreat_budget",
            )
        )
    executed_within_budget = len(all_actions) <= expected_budget
    passed = bool(
        goal_ever
        and final_goals[goal]
        and not wrong_goal_ever
        and not unexpected_done_before_goal
        and executed_within_budget
    )
    result = {
        "pass": passed,
        "goal": goal,
        "goal_ever_achieved": goal_ever,
        "first_goal_demo_frame": first_goal_frame,
        "wrong_goal_ever_achieved": wrong_goal_ever,
        "unexpected_done_before_goal": unexpected_done_before_goal,
        "normalization_goal_ever": normalization_goal_ever,
        "normalization_done_ever": normalization_done_ever,
        "normalized_goals": normalized_goals,
        "normalized_bowl_position_error_m": normalized_bowl_position_error_m,
        "normalized_state_sha256": snapshot_sha256(normalized_snapshot),
        "normalization_action_steps": normalization_action_count,
        "normalization_action_sha256": normalization_action_sha256,
        "final_goals": final_goals,
        "demo_episode_index": demo.episode_index,
        "demo_task_index": demo.task_index,
        "demo_action_sha256": demo.action_sha256,
        "phases": phases,
        "cost": {
            "budgeted_action_steps": expected_budget,
            "expected_budgeted_action_steps": expected_budget,
            "executed_action_steps": int(len(all_actions)),
            "unused_action_budget": int(expected_budget - len(all_actions)),
            "executed_within_budget": executed_within_budget,
            "active_servo_steps": int(
                sum(
                    int(phase.get("active_action_steps", 0))
                    for phase in phases.values()
                )
            ),
            "demonstration_action_steps": int(len(demo.actions)),
            "executed_demonstration_action_steps": int(
                len(all_actions) - (expected_budget - len(demo.actions))
            ),
            "eef_path_length_m": controller.eef_path_length_m,
            "control_effort": controller.control_effort,
            "motion_control_effort": controller.motion_control_effort,
            "action_sha256": _action_sha256(all_actions),
        },
        "done_count": int(sum(controller.done_values)),
    }
    restore_libero_state(environment, snapshot)
    if return_prepared_root:
        result["_prepared_oracle_root"] = prepared_root
    if not passed and raise_on_failure:
        raise RuntimeError(f"{goal} oracle failed for {spec.candidate_id}: {result}")
    return result


def run_goal_oracle_from_prepared_root(
    environment,
    root_snapshot: LiberoStateSnapshot,
    prepared: PreparedOracleRoot,
    *,
    spec: StageACandidateSpec,
    goal: str,
    demo: DemoTrace,
    config: dict[str, Any],
) -> dict[str, Any]:
    restore_libero_state(environment, prepared.snapshot)
    try:
        if evaluate_common_goals(environment) != prepared.normalized_goals:
            raise RuntimeError(
                f"Prepared normalized goals changed for {spec.candidate_id}/{goal}"
            )
        controller, proposal = replay_goal_proposal(
            environment,
            goal=goal,
            demo=demo,
        )
        all_actions = np.stack([*prepared.actions, *controller.actions])
        oracle_cfg = config["oracle"]
        expected_budget = int(oracle_cfg["home_budget"]) + len(demo.actions)
        if spec.possession == "grasped":
            expected_budget += sum(
                int(oracle_cfg[key])
                for key in (
                    "setdown_reverse_root_budget",
                    "setdown_return_lifted_budget",
                    "setdown_return_grasp_budget",
                    "setdown_table_safe_budget",
                    "setdown_descend_budget",
                    "setdown_release_budget",
                    "setdown_retreat_budget",
                )
            )
        executed_within_budget = len(all_actions) <= expected_budget
        passed = bool(proposal["pass"] and executed_within_budget)
        result = {
            "pass": passed,
            "goal": goal,
            "goal_ever_achieved": proposal["goal_ever_achieved"],
            "first_goal_demo_frame": proposal["first_goal_demo_frame"],
            "wrong_goal_ever_achieved": proposal[
                "wrong_goal_ever_achieved"
            ],
            "unexpected_done_before_goal": proposal[
                "unexpected_done_before_goal"
            ],
            "normalization_goal_ever": prepared.normalization_goal_ever,
            "normalization_done_ever": prepared.normalization_done_ever,
            "normalized_goals": prepared.normalized_goals,
            "normalized_bowl_position_error_m": (
                prepared.normalized_bowl_position_error_m
            ),
            "normalized_state_sha256": snapshot_sha256(prepared.snapshot),
            "normalization_action_steps": len(prepared.actions),
            "normalization_action_sha256": _action_sha256(prepared.actions),
            "final_goals": proposal["final_goals"],
            "demo_episode_index": demo.episode_index,
            "demo_task_index": demo.task_index,
            "demo_action_sha256": demo.action_sha256,
            "phases": prepared.phases,
            "cost": {
                "budgeted_action_steps": expected_budget,
                "expected_budgeted_action_steps": expected_budget,
                "executed_action_steps": int(len(all_actions)),
                "unused_action_budget": int(expected_budget - len(all_actions)),
                "executed_within_budget": executed_within_budget,
                "active_servo_steps": prepared.active_servo_steps,
                "demonstration_action_steps": int(len(demo.actions)),
                "executed_demonstration_action_steps": int(
                    len(controller.actions)
                ),
                "eef_path_length_m": (
                    prepared.eef_path_length_m + controller.eef_path_length_m
                ),
                "control_effort": (
                    prepared.control_effort + controller.control_effort
                ),
                "motion_control_effort": (
                    prepared.motion_control_effort
                    + controller.motion_control_effort
                ),
                "action_sha256": _action_sha256(all_actions),
            },
            "done_count": prepared.done_count + int(sum(controller.done_values)),
        }
    finally:
        restore_libero_state(environment, root_snapshot)
    return result


def run_action_phase_oracle_from_prepared_root(
    environment,
    root_snapshot: LiberoStateSnapshot,
    prepared: PreparedOracleRoot,
    *,
    spec: StageACandidateSpec,
    proposal: ActionPhaseProposal,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one goal continuation after a fixed, policy-free EEF bridge."""

    goal = proposal.source.goal
    if goal not in GOALS or proposal.suffix.goal != goal:
        raise ValueError("Action-phase oracle received inconsistent proposal goals")
    if proposal.metadata.get("layout") != spec.layout:
        raise ValueError(
            f"Action-phase anchor layout mismatch for {spec.candidate_id}/{goal}"
        )
    phase_cfg = config["action_phase_oracle"]
    restore_libero_state(environment, prepared.snapshot)
    try:
        if evaluate_common_goals(environment) != prepared.normalized_goals:
            raise RuntimeError(
                f"Prepared normalized goals changed for "
                f"{spec.candidate_id}/{goal}"
            )
        controller = PolicyFreeController(environment)
        initial_bowl_position = controller.bowl_position()
        registration = proposal.metadata.get("landmark_registration")
        execution_anchor = proposal.anchor_position
        root_landmark_registration = None
        if registration is not None:
            if (
                registration.get("type") != "translation_only"
                or registration.get("landmark") != BOWL_NAME
                or registration.get("target_layout") != spec.layout
            ):
                raise ValueError("Action-phase landmark registration changed")
            try:
                execution_anchor, root_landmark_registration = (
                    registered_root_execution_anchor(
                        proposal, initial_bowl_position, config=config
                    )
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Registered target landmark changed for "
                    f"{spec.candidate_id}/{goal}: {exc}"
                ) from exc
        route = grasped_root_transit_plan(
            controller.eef_position(),
            execution_anchor,
            clearance_margin_m=float(phase_cfg["clearance_margin_m"]),
            workspace_bounds=phase_cfg["workspace_bounds_m"],
            phase_budgets=phase_cfg["bridge_phase_budgets"],
        )
        phase_records: list[dict[str, Any]] = []
        for phase in route:
            intermediate = phase["phase"] != "target_descent"
            phase_result = controller.servo(
                target_position=phase["target_position"],
                target_orientation=proposal.anchor_orientation,
                gripper=float(phase_cfg["gripper_action"]),
                budget=int(phase["budget"]),
                max_translation_action=float(
                    phase_cfg["max_translation_action"]
                ),
                position_tolerance_m=float(
                    phase_cfg["intermediate_tolerance_m"]
                    if intermediate
                    else phase_cfg["final_tolerance_m"]
                ),
                orientation_tolerance_rad=float(
                    phase_cfg["orientation_tolerance_rad"]
                ),
            )
            phase_records.append(
                {
                    "phase": phase["phase"],
                    "target_position": np.asarray(
                        phase["target_position"]
                    ).tolist(),
                    "intermediate": intermediate,
                    "result": phase_result,
                }
            )

        bridge_action_count = len(controller.actions)
        bridge_actions = np.stack(controller.actions)
        bridge_goals = evaluate_common_goals(environment)
        bridge_drawer_joint = controller.top_drawer_position()
        bridge_bowl_drift = float(
            np.linalg.norm(
                controller.bowl_position() - initial_bowl_position
            )
        )
        bridge_grasp_ever = any(controller.grasp_values)
        bridge_done_ever = any(controller.done_values)
        bridge_goal_ever = any(
            any(values.values()) for values in controller.goal_values
        )
        aperture_preserved = bool(
            bridge_drawer_joint
            <= float(config["construction"]["open_drawer_threshold"])
            if spec.drawer_aperture == "open"
            else abs(bridge_drawer_joint)
            <= float(config["construction"]["closed_drawer_tolerance"])
        )
        bridge_pass = bool(
            all(record["result"]["pass"] for record in phase_records)
            and not bridge_grasp_ever
            and not bridge_done_ever
            and not bridge_goal_ever
            and not any(bridge_goals.values())
            and aperture_preserved
            and bridge_bowl_drift
            <= float(phase_cfg["bowl_drift_tolerance_m"])
        )
        if bridge_pass:
            _, outcome = replay_goal_proposal(
                environment,
                goal=goal,
                demo=proposal.suffix,
                controller=controller,
            )
        else:
            other_goal = "cabinet" if goal == "drawer" else "drawer"
            outcome = {
                "goal_ever_achieved": bool(
                    bridge_goal_ever or bridge_goals[goal]
                ),
                "first_goal_demo_frame": None,
                "wrong_goal_ever_achieved": bool(
                    bridge_goals[other_goal]
                    or any(
                        values[other_goal]
                        for values in controller.goal_values
                    )
                ),
                "unexpected_done_before_goal": bridge_done_ever,
                "final_goals": bridge_goals,
                "pass": False,
            }
        executed_source_action_steps = (
            len(controller.actions) - bridge_action_count
        )
        proposal_execution_action_steps = len(controller.actions)
        all_actions = np.stack([*prepared.actions, *controller.actions])
        bridge_budget = sum(
            int(value)
            for value in phase_cfg["bridge_phase_budgets"].values()
        )
        expected_budget = (
            len(prepared.actions)
            + bridge_budget
            + len(proposal.suffix.actions)
        )
        executed_within_budget = len(all_actions) <= expected_budget
        bridge = {
            "mode": phase_cfg["bridge_mode"],
            "pass": bridge_pass,
            "phases": phase_records,
            "budgeted_action_steps": bridge_budget,
            "executed_action_steps": bridge_action_count,
            "action_sha256": _action_sha256(bridge_actions),
            "active_action_steps": int(
                sum(
                    int(record["result"]["active_action_steps"])
                    for record in phase_records
                )
            ),
            "bowl_drift_m": bridge_bowl_drift,
            "bowl_grasp_ever": bridge_grasp_ever,
            "done_ever": bridge_done_ever,
            "goal_ever": bridge_goal_ever,
            "final_goals": bridge_goals,
            "final_drawer_joint": bridge_drawer_joint,
            "drawer_aperture_preserved": aperture_preserved,
            "root_landmark_registration": root_landmark_registration,
        }
        passed = bool(bridge_pass and outcome["pass"] and executed_within_budget)
        return {
            "pass": passed,
            "goal": goal,
            "goal_ever_achieved": outcome["goal_ever_achieved"],
            "first_goal_demo_frame": outcome["first_goal_demo_frame"],
            "wrong_goal_ever_achieved": outcome[
                "wrong_goal_ever_achieved"
            ],
            "unexpected_done_before_goal": outcome[
                "unexpected_done_before_goal"
            ],
            "normalization_goal_ever": prepared.normalization_goal_ever,
            "normalization_done_ever": prepared.normalization_done_ever,
            "normalized_goals": prepared.normalized_goals,
            "normalized_bowl_position_error_m": (
                prepared.normalized_bowl_position_error_m
            ),
            "normalized_state_sha256": snapshot_sha256(prepared.snapshot),
            "normalization_action_steps": len(prepared.actions),
            "normalization_action_sha256": _action_sha256(prepared.actions),
            "final_goals": outcome["final_goals"],
            "demo_episode_index": proposal.source.episode_index,
            "demo_task_index": proposal.source.task_index,
            "demo_action_sha256": proposal.source.action_sha256,
            "proposal_execution_mode": phase_cfg["execution_mode"],
            "phase_proposal": proposal.metadata,
            "phases": {**prepared.phases, "action_phase_bridge": bridge},
            "cost": {
                "budgeted_action_steps": expected_budget,
                "expected_budgeted_action_steps": expected_budget,
                "executed_action_steps": int(len(all_actions)),
                "unused_action_budget": int(expected_budget - len(all_actions)),
                "executed_within_budget": executed_within_budget,
                "active_servo_steps": int(
                    prepared.active_servo_steps
                    + bridge["active_action_steps"]
                ),
                "demonstration_action_steps": int(
                    len(proposal.source.actions)
                ),
                "source_suffix_action_steps": int(
                    len(proposal.suffix.actions)
                ),
                "executed_source_action_steps": int(
                    executed_source_action_steps
                ),
                "proposal_execution_action_steps": int(
                    proposal_execution_action_steps
                ),
                "executed_demonstration_action_steps": int(
                    proposal_execution_action_steps
                ),
                "eef_path_length_m": (
                    prepared.eef_path_length_m
                    + controller.eef_path_length_m
                ),
                "control_effort": (
                    prepared.control_effort + controller.control_effort
                ),
                "motion_control_effort": (
                    prepared.motion_control_effort
                    + controller.motion_control_effort
                ),
                "action_sha256": _action_sha256(all_actions),
            },
            "done_count": prepared.done_count
            + int(sum(controller.done_values)),
        }
    finally:
        restore_libero_state(environment, root_snapshot)


def run_goal_oracle_bank(
    environment,
    snapshot: LiberoStateSnapshot,
    *,
    spec: StageACandidateSpec,
    goal: str,
    proposals: tuple[DemoTrace, ...],
    initial_bowl_position: np.ndarray,
    initial_eef_position: np.ndarray,
    initial_eef_orientation: np.ndarray,
    initial_joint_positions: np.ndarray,
    recovery_waypoints: dict[str, np.ndarray] | None,
    config: dict[str, Any],
    action_phase_proposals: tuple[ActionPhaseProposal, ...] | None = None,
    completed_results: dict[int, dict[str, Any]] | None = None,
    result_callback: Callable[[int, dict[str, Any]], None] | None = None,
    allow_exhaustive_failure: bool = False,
) -> dict[str, Any]:
    if not proposals:
        raise ValueError(f"{goal} oracle proposal bank is empty")
    phase_mode = action_phase_proposals is not None
    if phase_mode:
        if len(action_phase_proposals) != len(proposals):
            raise ValueError(f"{goal} action-phase proposal bank is incomplete")
        for source, phase_proposal in zip(
            proposals, action_phase_proposals, strict=True
        ):
            if (
                phase_proposal.source.episode_index != source.episode_index
                or phase_proposal.source.task_index != source.task_index
                or phase_proposal.source.action_sha256 != source.action_sha256
            ):
                raise ValueError("Action-phase/source proposal identity mismatch")
        proposal_execution_mode = config["action_phase_oracle"][
            "execution_mode"
        ]
        proposal_execution_contract = [
            item.metadata for item in action_phase_proposals
        ]
    else:
        proposal_execution_mode = "full_trajectory_replay"
        proposal_execution_contract = {
            "execution_mode": proposal_execution_mode,
            "transformation": "none",
        }
    proposal_execution_contract_sha256 = canonical_sha256(
        proposal_execution_contract
    )
    proposal_metadata = [
        {
            "proposal_index": index,
            "goal": proposal.goal,
            "episode_index": proposal.episode_index,
            "task_index": proposal.task_index,
            "frame_count": int(len(proposal.actions)),
            "action_sha256": proposal.action_sha256,
        }
        for index, proposal in enumerate(proposals)
    ]
    if any(proposal.goal != goal for proposal in proposals):
        raise ValueError(f"{goal} oracle proposal bank contains another goal")
    proposal_bank_sha256 = canonical_sha256(proposal_metadata)
    attempts: list[dict[str, Any]] = []
    all_results: dict[int, dict[str, Any]] = {}
    successful_results: dict[int, dict[str, Any]] = {}
    counterfactual_full_attempt_action_steps = 0
    normalized_state_hashes: set[str] = set()
    normalization_action_hashes: set[str] = set()
    preparation = run_goal_oracle(
        environment,
        snapshot,
        spec=spec,
        goal=goal,
        demo=proposals[0],
        initial_bowl_position=initial_bowl_position,
        initial_eef_position=initial_eef_position,
        initial_eef_orientation=initial_eef_orientation,
        initial_joint_positions=initial_joint_positions,
        recovery_waypoints=recovery_waypoints,
        config=config,
        raise_on_failure=False,
        return_prepared_root=True,
        normalization_only=True,
    )
    prepared = preparation.pop("_prepared_oracle_root")
    normalization_preparation = {
        "execution_mode": "normalization_only",
        "source_proposal_replayed": preparation["source_proposal_replayed"],
        "executed_action_steps": preparation["normalization_action_steps"],
        "action_sha256": preparation["normalization_action_sha256"],
    }
    completed_results = completed_results or {}
    unknown_completed = set(completed_results) - set(range(len(proposals)))
    if unknown_completed:
        raise ValueError(
            f"{goal} checkpoint has unknown proposal indices: "
            f"{sorted(unknown_completed)}"
        )
    for proposal_index, proposal in enumerate(proposals):
        if proposal_index in completed_results:
            result = completed_results[proposal_index]
        elif phase_mode:
            result = run_action_phase_oracle_from_prepared_root(
                environment,
                snapshot,
                prepared,
                spec=spec,
                proposal=action_phase_proposals[proposal_index],
                config=config,
            )
        else:
            result = run_goal_oracle_from_prepared_root(
                environment,
                snapshot,
                prepared,
                spec=spec,
                goal=goal,
                demo=proposal,
                config=config,
            )
        if (
            result.get("goal") != goal
            or result.get("demo_episode_index") != proposal.episode_index
            or result.get("demo_task_index") != proposal.task_index
            or result.get("demo_action_sha256") != proposal.action_sha256
        ):
            raise ValueError(
                f"{goal} checkpoint/result identity mismatch at proposal "
                f"{proposal_index}"
            )
        cost = result["cost"]
        normalized_state_hashes.add(result["normalized_state_sha256"])
        normalization_action_hashes.add(result["normalization_action_sha256"])
        if (
            result["normalized_state_sha256"]
            != snapshot_sha256(prepared.snapshot)
            or result["normalization_action_sha256"]
            != _action_sha256(prepared.actions)
            or len(normalized_state_hashes) != 1
            or len(normalization_action_hashes) != 1
        ):
            restore_libero_state(environment, snapshot)
            raise RuntimeError(
                f"{goal} proposal attempts did not share one normalized root for "
                f"{spec.candidate_id}"
            )
        if proposal_index not in completed_results and result_callback is not None:
            result_callback(proposal_index, result)
        counterfactual_full_attempt_action_steps += int(
            cost["executed_action_steps"]
        )
        attempt = {
                "proposal_index": proposal_index,
                "episode_index": proposal.episode_index,
                "task_index": proposal.task_index,
                "action_sha256": proposal.action_sha256,
                "proposal_execution_mode": proposal_execution_mode,
                "pass": result["pass"],
                "goal_ever_achieved": result["goal_ever_achieved"],
                "first_goal_demo_frame": result["first_goal_demo_frame"],
                "wrong_goal_ever_achieved": result["wrong_goal_ever_achieved"],
                "unexpected_done_before_goal": result[
                    "unexpected_done_before_goal"
                ],
                "normalized_bowl_position_error_m": result[
                    "normalized_bowl_position_error_m"
                ],
                "normalized_state_sha256": result["normalized_state_sha256"],
                "normalization_action_sha256": result[
                    "normalization_action_sha256"
                ],
                "final_goals": result["final_goals"],
                "cost": cost,
            }
        if phase_mode:
            attempt["phase_proposal"] = result["phase_proposal"]
            attempt["action_phase_bridge"] = result["phases"][
                "action_phase_bridge"
            ]
        attempts.append(attempt)
        all_results[proposal_index] = result
        if result["pass"]:
            successful_results[proposal_index] = result
    if prepared is None:
        raise RuntimeError("Oracle proposal bank has no prepared root")
    actual_bank_search_action_steps = len(prepared.actions) + sum(
        int(attempt["cost"]["executed_demonstration_action_steps"])
        for attempt in attempts
    )
    common_ledger = {
        "proposal_bank_sha256": proposal_bank_sha256,
        "proposal_bank": proposal_metadata,
        "proposal_execution_mode": proposal_execution_mode,
        "proposal_execution_contract_sha256": (
            proposal_execution_contract_sha256
        ),
        "proposal_execution_contract": proposal_execution_contract,
        "proposal_attempts": attempts,
        "proposal_attempt_count": len(attempts),
        "proposal_success_count": len(successful_results),
        "proposal_success_fraction": float(
            len(successful_results) / len(attempts)
        ),
        "successful_proposal_indices": sorted(successful_results),
        "proposal_selection_rule": (
            "minimum_executed_steps_then_path_effort_index"
        ),
        "shared_normalized_state_sha256": next(
            iter(normalized_state_hashes)
        ),
        "shared_normalization_action_sha256": next(
            iter(normalization_action_hashes)
        ),
        "shared_normalization_action_steps": len(prepared.actions),
        "shared_normalization_active_servo_steps": (
            prepared.active_servo_steps
        ),
        "total_attempted_action_steps": actual_bank_search_action_steps,
        "counterfactual_full_attempt_action_steps": (
            counterfactual_full_attempt_action_steps
        ),
        "normalization_preparation": normalization_preparation,
        "total_environment_action_steps": int(
            actual_bank_search_action_steps
        ),
    }
    if successful_results:
        selected_index = min(
            successful_results,
            key=lambda index: (
                int(
                    successful_results[index]["cost"][
                        "executed_action_steps"
                    ]
                ),
                float(
                    successful_results[index]["cost"]["eef_path_length_m"]
                ),
                float(
                    successful_results[index]["cost"][
                        "motion_control_effort"
                    ]
                ),
                int(index),
            ),
        )
        result = deepcopy(successful_results[selected_index])
        result.update(
            {
                **common_ledger,
                "selected_proposal_index": selected_index,
            }
        )
        return result
    if allow_exhaustive_failure:
        result = deepcopy(all_results[0])
        result.update(
            {
                **common_ledger,
                "pass": False,
                "goal_ever_achieved": False,
                "demo_episode_index": None,
                "demo_task_index": None,
                "demo_action_sha256": None,
                "selected_proposal_index": None,
                "cost": None,
                "proposal_coverage_status": "exhaustive_failure",
            }
        )
        return result
    restore_libero_state(environment, snapshot)
    raise RuntimeError(
        f"{goal} proposal bank exhausted for {spec.candidate_id}: {attempts}"
    )


def candidate_record(
    *,
    spec: StageACandidateSpec,
    constructed: ConstructedCandidate,
    certificate: dict[str, Any],
    oracles: dict[str, dict[str, Any]],
    contract_sha256: str,
    selection_lock_sha256: str,
    construction_revision: str,
) -> dict[str, Any]:
    if constructed.support_measurement is None:
        raise ValueError("Stage A candidate has no joint support measurement")
    return {
        "schema_version": 1,
        "candidate_id": spec.candidate_id,
        "factors": spec.as_dict(),
        "contract_sha256": contract_sha256,
        "selection_lock_sha256": selection_lock_sha256,
        "construction_revision": construction_revision,
        "policy_loaded": False,
        "state_sha256": snapshot_sha256(constructed.snapshot),
        "construction": constructed.construction,
        "root_validation": constructed.root_validation,
        "root_geometry": constructed.root_geometry,
        "support_measurement": constructed.support_measurement,
        "certificate": certificate,
        "oracles": oracles,
    }


def compact_snapshot_metadata(snapshot: LiberoStateSnapshot) -> dict[str, Any]:
    return {
        "mujoco_state_length": int(snapshot.mujoco_state.size),
        "object_count": len(snapshot.objects),
        "contact_count": len(snapshot.contacts),
        "grasped_objects": list(snapshot.grasped_objects),
        "goal_predicates": list(snapshot.goal_predicates),
        "success": snapshot.success,
        "full_sim_data_fields": sorted(
            snapshot.runtime_state.get("sim_data", {}).keys()
        ),
    }


def dump_snapshot_payload(snapshot: LiberoStateSnapshot) -> tuple[np.ndarray, str]:
    return snapshot.mujoco_state, json.dumps(snapshot.metadata(), sort_keys=True)
