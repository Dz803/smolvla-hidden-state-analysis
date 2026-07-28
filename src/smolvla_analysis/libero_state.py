from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np


FULL_SIM_DATA_FIELDS = (
    "act",
    "ctrl",
    "qacc_warmstart",
    "qfrc_applied",
    "xfrc_applied",
    "mocap_pos",
    "mocap_quat",
    "userdata",
    "eq_active",
    "plugin_state",
)


def _array(value: Any) -> list:
    return np.asarray(value).tolist()


@dataclass(frozen=True)
class LiberoStateSnapshot:
    mujoco_state: np.ndarray
    objects: dict[str, dict[str, Any]]
    goal_predicates: tuple[dict[str, Any], ...]
    contacts: tuple[dict[str, Any], ...]
    grasped_objects: tuple[str, ...]
    success: bool
    runtime_state: dict[str, Any] = field(default_factory=dict)
    numpy_random_state: dict[str, Any] = field(default_factory=dict)

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "mujoco_state_length": int(self.mujoco_state.size),
            "objects": self.objects,
            "goal_predicates": list(self.goal_predicates),
            "contacts": list(self.contacts),
            "grasped_objects": list(self.grasped_objects),
            "success": self.success,
            "runtime_state": self.runtime_state,
            "numpy_random_state": self.numpy_random_state,
        }


def _unwrap_libero_environment(environment):
    if hasattr(environment, "envs"):
        if len(environment.envs) != 1:
            raise ValueError("State capture requires exactly one synchronous LIBERO environment")
        lerobot_env = environment.envs[0]
    else:
        lerobot_env = environment
    control_env = getattr(lerobot_env, "_env", lerobot_env)
    if not hasattr(control_env, "get_sim_state") or not hasattr(control_env, "regenerate_obs_from_state"):
        raise TypeError("LIBERO environment does not expose get_sim_state/regenerate_obs_from_state")
    problem_env = getattr(control_env, "env", None)
    if problem_env is None or not hasattr(problem_env, "sim"):
        raise TypeError("LIBERO control environment does not expose its MuJoCo task environment")
    return lerobot_env, control_env, problem_env


def libero_problem_environment(environment):
    """Return the underlying LIBERO task environment for predicate evaluation."""

    return _unwrap_libero_environment(environment)[2]


def _object_metadata(problem_env) -> tuple[dict[str, dict[str, Any]], tuple[str, ...]]:
    objects: dict[str, dict[str, Any]] = {}
    grasped = []
    movable = set(getattr(problem_env, "objects_dict", {}))
    for name, state in sorted(getattr(problem_env, "object_states_dict", {}).items()):
        entry: dict[str, Any] = {"state_type": getattr(state, "object_state_type", "unknown")}
        geometry = state.get_geom_state()
        entry["position"] = _array(geometry["pos"])
        entry["quaternion"] = _array(geometry["quat"])
        named_joints = {}
        try:
            object_model = problem_env.get_object(name)
            for joint_name in getattr(object_model, "joints", []):
                address = problem_env.sim.model.get_joint_qpos_addr(joint_name)
                values = (
                    problem_env.sim.data.qpos[slice(*address)]
                    if isinstance(address, tuple)
                    else np.asarray([problem_env.sim.data.qpos[address]])
                )
                named_joints[str(joint_name)] = _array(values)
        except (AttributeError, KeyError, TypeError, ValueError):
            named_joints = {}
        if named_joints:
            joints = [value for values in named_joints.values() for value in values]
        else:
            try:
                joints = state.get_joint_state()
            except (AttributeError, IndexError, NotImplementedError, TypeError, ValueError):
                joints = None
        entry["joint_state"] = None if joints is None else _array(joints)
        entry["joint_states"] = named_joints

        is_grasped = False
        if name in movable and hasattr(problem_env, "_check_grasp") and getattr(problem_env, "robots", None):
            try:
                is_grasped = bool(
                    problem_env._check_grasp(
                        gripper=problem_env.robots[0].gripper,
                        object_geoms=problem_env.get_object(name),
                    )
                )
            except (AttributeError, KeyError, TypeError, ValueError):
                is_grasped = False
        entry["grasped"] = is_grasped
        if is_grasped:
            grasped.append(name)
        objects[name] = entry
    return objects, tuple(grasped)


def _goal_metadata(problem_env) -> tuple[dict[str, Any], ...]:
    result = []
    for predicate in getattr(problem_env, "parsed_problem", {}).get("goal_state", []):
        result.append(
            {
                "predicate": [str(item) for item in predicate],
                "satisfied": bool(problem_env._eval_predicate(predicate)),
            }
        )
    return tuple(result)


def _contact_metadata(problem_env) -> tuple[dict[str, Any], ...]:
    sim = problem_env.sim
    result = []
    for index in range(int(sim.data.ncon)):
        contact = sim.data.contact[index]
        result.append(
            {
                "geom1": sim.model.geom_id2name(int(contact.geom1)),
                "geom2": sim.model.geom_id2name(int(contact.geom2)),
                "distance": float(contact.dist),
                "position": _array(contact.pos),
            }
        )
    return tuple(result)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _capture_attributes(value: Any, names: tuple[str, ...]) -> dict[str, Any]:
    return {
        name: _jsonable(deepcopy(getattr(value, name)))
        for name in names
        if hasattr(value, name) and getattr(value, name) is not None
    }


def _capture_buffer(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "last") and hasattr(value, "current"):
        return {
            "kind": "delta",
            "last": _jsonable(value.last),
            "current": _jsonable(value.current),
        }
    if hasattr(value, "buf") and hasattr(value, "ptr"):
        return {
            "kind": "ring",
            "buf": _jsonable(value.buf),
            "ptr": int(value.ptr),
            "size": int(getattr(value, "_size", len(value.buf))),
        }
    return None


_CONTROLLER_ATTRIBUTES = (
    "goal_pos",
    "goal_ori",
    "relative_ori",
    "ori_ref",
    "kp",
    "kd",
    "initial_joint",
    "torques",
)
_INTERPOLATOR_ATTRIBUTES = ("start", "goal", "step")
_ROBOT_BUFFER_ATTRIBUTES = (
    "recent_qpos",
    "recent_actions",
    "recent_torques",
    "recent_ee_forcetorques",
    "recent_ee_pose",
    "recent_ee_vel",
    "recent_ee_acc",
    "recent_ee_vel_buffer",
)


def _capture_runtime_state(environment, lerobot_env, problem_env) -> dict[str, Any]:
    robots = []
    for robot in getattr(problem_env, "robots", []):
        controller = getattr(robot, "controller", None)
        controller_state = _capture_attributes(controller, _CONTROLLER_ATTRIBUTES) if controller else {}
        if controller is not None:
            for name in ("interpolator_pos", "interpolator_ori"):
                interpolator = getattr(controller, name, None)
                if interpolator is not None:
                    controller_state[name] = _capture_attributes(interpolator, _INTERPOLATOR_ATTRIBUTES)
        buffers = {}
        for name in _ROBOT_BUFFER_ATTRIBUTES:
            captured = _capture_buffer(getattr(robot, name, None))
            if captured is not None:
                buffers[name] = captured
        robot_state = {"controller": controller_state, "buffers": buffers}
        gripper = getattr(robot, "gripper", None)
        if gripper is not None and hasattr(gripper, "current_action"):
            robot_state["gripper_current_action"] = _jsonable(deepcopy(gripper.current_action))
        robots.append(robot_state)

    observables = {
        name: _capture_attributes(
            observable,
            ("_time_since_last_sample", "_current_delay", "_sampled"),
        )
        for name, observable in sorted(getattr(problem_env, "_observables", {}).items())
    }
    vector_state = _capture_attributes(
        environment,
        ("_terminations", "_truncations", "_autoreset_envs"),
    )
    gym_rng = getattr(lerobot_env, "np_random", None)
    gym_rng_state = (
        _jsonable(deepcopy(gym_rng.bit_generator.state))
        if gym_rng is not None and hasattr(gym_rng, "bit_generator")
        else None
    )
    sim_data = {}
    for name in FULL_SIM_DATA_FIELDS:
        if not hasattr(problem_env.sim.data, name):
            continue
        values = np.asarray(getattr(problem_env.sim.data, name)).copy()
        sim_data[name] = {
            "shape": list(values.shape),
            "values": _jsonable(values),
        }
    return {
        "lerobot": _capture_attributes(lerobot_env, ("init_state_id",)),
        "environment": _capture_attributes(problem_env, ("timestep", "cur_time", "done")),
        "vector": vector_state,
        "gym_rng_state": gym_rng_state,
        "sim_data": sim_data,
        "robots": robots,
        "observables": observables,
    }


def _capture_numpy_random_state() -> dict[str, Any]:
    state = np.random.get_state()
    return {
        "bit_generator": state[0],
        "keys": state[1].tolist(),
        "position": int(state[2]),
        "has_gauss": int(state[3]),
        "cached_gaussian": float(state[4]),
    }


def _restore_attributes(value: Any, captured: dict[str, Any]) -> None:
    for name, item in captured.items():
        current = getattr(value, name, None)
        if isinstance(current, np.ndarray):
            setattr(value, name, np.asarray(item, dtype=current.dtype))
        elif isinstance(current, np.generic):
            setattr(value, name, type(current)(item))
        else:
            setattr(value, name, deepcopy(item))


def _restore_buffer(value: Any, captured: dict[str, Any]) -> None:
    if captured["kind"] == "delta":
        value.last = np.asarray(captured["last"], dtype=np.asarray(value.last).dtype)
        value.current = np.asarray(captured["current"], dtype=np.asarray(value.current).dtype)
    elif captured["kind"] == "ring":
        value.buf = np.asarray(captured["buf"], dtype=np.asarray(value.buf).dtype)
        value.ptr = int(captured["ptr"])
        value._size = int(captured["size"])
    else:
        raise ValueError(f"Unknown buffer kind: {captured['kind']}")


def _restore_numpy_random_state(captured: dict[str, Any]) -> None:
    if not captured:
        return
    np.random.set_state(
        (
            captured["bit_generator"],
            np.asarray(captured["keys"], dtype=np.uint32),
            int(captured["position"]),
            int(captured["has_gauss"]),
            float(captured["cached_gaussian"]),
        )
    )


def _restore_runtime_state(environment, lerobot_env, problem_env, captured: dict[str, Any]) -> None:
    if not captured:
        return
    _restore_attributes(lerobot_env, captured.get("lerobot", {}))
    _restore_attributes(problem_env, captured.get("environment", {}))
    _restore_attributes(environment, captured.get("vector", {}))

    gym_rng_state = captured.get("gym_rng_state")
    gym_rng = getattr(lerobot_env, "np_random", None)
    if gym_rng_state is not None and gym_rng is not None and hasattr(gym_rng, "bit_generator"):
        gym_rng.bit_generator.state = deepcopy(gym_rng_state)

    for name, item in captured.get("sim_data", {}).items():
        if not hasattr(problem_env.sim.data, name):
            raise AttributeError(f"MuJoCo data field disappeared during restore: {name}")
        current = np.asarray(getattr(problem_env.sim.data, name))
        if isinstance(item, dict) and set(item) == {"shape", "values"}:
            restored = np.asarray(item["values"], dtype=current.dtype).reshape(tuple(item["shape"]))
        else:
            restored = np.asarray(item, dtype=current.dtype)
        if current.shape != restored.shape:
            raise ValueError(
                f"MuJoCo data field shape changed for {name}: {restored.shape} -> {current.shape}"
            )
        current[...] = restored

    for robot, robot_state in zip(getattr(problem_env, "robots", []), captured.get("robots", [])):
        controller = getattr(robot, "controller", None)
        controller_state = robot_state.get("controller", {})
        if controller is not None:
            direct = {
                key: value
                for key, value in controller_state.items()
                if key not in {"interpolator_pos", "interpolator_ori"}
            }
            _restore_attributes(controller, direct)
            for name in ("interpolator_pos", "interpolator_ori"):
                interpolator = getattr(controller, name, None)
                if interpolator is not None and name in controller_state:
                    _restore_attributes(interpolator, controller_state[name])
        gripper = getattr(robot, "gripper", None)
        if gripper is not None and "gripper_current_action" in robot_state:
            current = getattr(gripper, "current_action", np.asarray([]))
            gripper.current_action = np.asarray(
                robot_state["gripper_current_action"], dtype=np.asarray(current).dtype
            )
        for name, buffer_state in robot_state.get("buffers", {}).items():
            buffer = getattr(robot, name, None)
            if buffer is not None:
                _restore_buffer(buffer, buffer_state)

    for name, observable_state in captured.get("observables", {}).items():
        observable = getattr(problem_env, "_observables", {}).get(name)
        if observable is not None:
            _restore_attributes(observable, observable_state)


def capture_libero_state(environment) -> LiberoStateSnapshot:
    lerobot_env, control_env, problem_env = _unwrap_libero_environment(environment)
    objects, grasped = _object_metadata(problem_env)
    return LiberoStateSnapshot(
        mujoco_state=np.asarray(control_env.get_sim_state(), dtype=np.float64).copy(),
        objects=objects,
        goal_predicates=_goal_metadata(problem_env),
        contacts=_contact_metadata(problem_env),
        grasped_objects=grasped,
        success=bool(control_env.check_success()),
        runtime_state=_capture_runtime_state(environment, lerobot_env, problem_env),
        numpy_random_state=_capture_numpy_random_state(),
    )


def restore_libero_state(environment, snapshot: LiberoStateSnapshot):
    lerobot_env, control_env, problem_env = _unwrap_libero_environment(environment)
    raw_observation = control_env.regenerate_obs_from_state(snapshot.mujoco_state.copy())
    _restore_runtime_state(environment, lerobot_env, problem_env, snapshot.runtime_state)
    _restore_numpy_random_state(snapshot.numpy_random_state)
    formatter = getattr(lerobot_env, "_format_raw_obs", None)
    return formatter(raw_observation) if formatter is not None else raw_observation


def validate_libero_round_trip(environment, snapshot: LiberoStateSnapshot, *, atol: float = 1e-10) -> dict[str, Any]:
    restore_libero_state(environment, snapshot)
    restored = capture_libero_state(environment)
    if snapshot.mujoco_state.shape != restored.mujoco_state.shape:
        raise AssertionError(
            f"MuJoCo state shape changed: {snapshot.mujoco_state.shape} -> {restored.mujoco_state.shape}"
        )
    maximum = float(np.max(np.abs(snapshot.mujoco_state - restored.mujoco_state), initial=0.0))
    predicates_match = snapshot.goal_predicates == restored.goal_predicates
    grasp_match = snapshot.grasped_objects == restored.grasped_objects
    runtime_match = snapshot.runtime_state == restored.runtime_state
    numpy_random_match = snapshot.numpy_random_state == restored.numpy_random_state
    if maximum > atol or not predicates_match or not grasp_match or not runtime_match or not numpy_random_match:
        raise AssertionError(
            "LIBERO state round trip failed: "
            f"max_abs_diff={maximum}, predicates_match={predicates_match}, grasp_match={grasp_match}, "
            f"runtime_match={runtime_match}, numpy_random_match={numpy_random_match}"
        )
    return {
        "max_abs_state_diff": maximum,
        "goal_predicates_match": predicates_match,
        "grasped_objects_match": grasp_match,
        "runtime_state_match": runtime_match,
        "numpy_random_state_match": numpy_random_match,
        "object_count": len(restored.objects),
        "contact_count": len(restored.contacts),
    }
