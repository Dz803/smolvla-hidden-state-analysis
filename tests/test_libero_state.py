from types import SimpleNamespace

import numpy as np
import zarr

from smolvla_analysis.libero_state import (
    capture_libero_state,
    restore_libero_state,
    validate_libero_round_trip,
)
from smolvla_analysis.phase2_storage import read_libero_snapshot, write_libero_snapshot


class FakeObjectState:
    object_state_type = "object"

    def __init__(self, problem):
        self.problem = problem

    def get_geom_state(self):
        return {"pos": self.problem.state[:3], "quat": np.asarray([1.0, 0.0, 0.0, 0.0])}

    def get_joint_state(self):
        return [self.problem.state[3]]


class FakeModel:
    @staticmethod
    def geom_id2name(index):
        return {0: "robot0_left_finger", 1: "bowl_geom"}[index]


class FakeContact:
    geom1 = 0
    geom2 = 1
    dist = -0.001
    pos = np.asarray([0.1, 0.2, 0.3])


class FakeDeltaBuffer:
    def __init__(self):
        self.last = np.asarray([0.0, 1.0])
        self.current = np.asarray([2.0, 3.0])


class FakeObservable:
    def __init__(self):
        self._time_since_last_sample = 0.01
        self._current_delay = 0.0
        self._sampled = True


class FakeProblem:
    def __init__(self):
        self.state = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        self.objects_dict = {"bowl": object()}
        self.object_states_dict = {"bowl": FakeObjectState(self)}
        self.parsed_problem = {"goal_state": [("above_threshold", "bowl")]}
        interpolator = SimpleNamespace(
            start=np.asarray([0.1, 0.2, 0.3]),
            goal=np.asarray([0.2, 0.3, 0.4]),
            step=2,
        )
        controller = SimpleNamespace(
            goal_pos=np.asarray([0.2, 0.3, 0.4]),
            kp=np.asarray([150.0, 150.0, 150.0]),
            interpolator_pos=interpolator,
            interpolator_ori=None,
        )
        self.robots = [
            SimpleNamespace(
                gripper=SimpleNamespace(current_action=np.asarray([-0.2, 0.2])),
                controller=controller,
                recent_actions=FakeDeltaBuffer(),
            )
        ]
        self.timestep = 11
        self.cur_time = 0.55
        self.done = False
        self._observables = {"eef_pos": FakeObservable()}
        self.sim = SimpleNamespace(
            model=FakeModel(),
            data=SimpleNamespace(
                ncon=1,
                contact=[FakeContact()],
                ctrl=np.asarray([0.3, -0.4]),
                qacc_warmstart=np.asarray([0.01, -0.02]),
                qfrc_applied=np.asarray([0.0, 0.1]),
                mocap_pos=np.zeros((0, 3)),
            ),
        )

    def _eval_predicate(self, _predicate):
        return self.state[0] > 0

    def _check_grasp(self, gripper, object_geoms):
        del gripper, object_geoms
        return self.state[3] > 0

    def get_object(self, name):
        return self.objects_dict[name]


class FakeControlEnvironment:
    def __init__(self, problem):
        self.env = problem

    def get_sim_state(self):
        return self.env.state.copy()

    def regenerate_obs_from_state(self, state):
        self.env.state = np.asarray(state, dtype=np.float64).copy()
        return {"raw_state": self.env.state.copy()}

    def check_success(self):
        return self.env._eval_predicate(None)


class FakeLeRobotEnvironment:
    def __init__(self):
        self.problem = FakeProblem()
        self._env = FakeControlEnvironment(self.problem)
        self.init_state_id = 3
        self.np_random = np.random.default_rng(123)

    @staticmethod
    def _format_raw_obs(raw):
        return {"formatted_state": raw["raw_state"]}


def test_libero_snapshot_captures_semantics_and_restores_state():
    environment = SimpleNamespace(
        envs=[FakeLeRobotEnvironment()],
        _terminations=np.asarray([False]),
        _truncations=np.asarray([False]),
    )
    snapshot = capture_libero_state(environment)

    assert snapshot.success
    assert snapshot.grasped_objects == ("bowl",)
    assert snapshot.goal_predicates[0]["satisfied"]
    assert snapshot.objects["bowl"]["position"] == [0.1, 0.2, 0.3]
    assert snapshot.contacts[0]["geom1"] == "robot0_left_finger"

    environment.envs[0].problem.state[:] = -1
    environment.envs[0].problem.timestep = 99
    environment.envs[0].problem.robots[0].controller.goal_pos[:] = -2
    environment.envs[0].problem.robots[0].controller.interpolator_pos.step = 9
    environment.envs[0].problem.robots[0].gripper.current_action[:] = 1
    environment.envs[0].problem.robots[0].recent_actions.current[:] = -3
    environment.envs[0].problem._observables["eef_pos"]._sampled = False
    environment.envs[0].problem.sim.data.ctrl[:] = 9
    environment.envs[0].problem.sim.data.qacc_warmstart[:] = 8
    environment.envs[0].init_state_id = 10
    environment._terminations[:] = True
    np.random.random(5)
    observation = restore_libero_state(environment, snapshot)
    np.testing.assert_allclose(observation["formatted_state"], snapshot.mujoco_state)
    np.testing.assert_allclose(
        environment.envs[0].problem.robots[0].controller.goal_pos,
        [0.2, 0.3, 0.4],
    )
    np.testing.assert_allclose(
        environment.envs[0].problem.robots[0].gripper.current_action,
        [-0.2, 0.2],
    )
    assert environment.envs[0].problem.robots[0].controller.interpolator_pos.step == 2
    assert environment.envs[0].problem.timestep == 11
    assert environment.envs[0].init_state_id == 3
    assert not environment._terminations[0]
    np.testing.assert_allclose(environment.envs[0].problem.sim.data.ctrl, [0.3, -0.4])
    np.testing.assert_allclose(
        environment.envs[0].problem.sim.data.qacc_warmstart, [0.01, -0.02]
    )
    assert snapshot.runtime_state["sim_data"]["mocap_pos"]["shape"] == [0, 3]
    assert environment.envs[0].problem.sim.data.mocap_pos.shape == (0, 3)
    result = validate_libero_round_trip(environment, snapshot)
    assert result["max_abs_state_diff"] == 0
    assert result["goal_predicates_match"]
    assert result["grasped_objects_match"]
    assert result["runtime_state_match"]
    assert result["numpy_random_state_match"]


def test_libero_snapshot_storage_is_immutable(tmp_path):
    environment = SimpleNamespace(envs=[FakeLeRobotEnvironment()])
    snapshot = capture_libero_state(environment)
    store = zarr.open_group(str(tmp_path / "states.zarr"), mode="w")
    write_libero_snapshot(store, "step_0000", snapshot)
    np.testing.assert_allclose(store["step_0000/mujoco_state"][:], snapshot.mujoco_state)
    restored = read_libero_snapshot(store, "step_0000")
    np.testing.assert_allclose(restored.mujoco_state, snapshot.mujoco_state)
    assert restored.objects == snapshot.objects
    assert restored.runtime_state == snapshot.runtime_state
    assert restored.numpy_random_state == snapshot.numpy_random_state
