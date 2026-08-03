from __future__ import annotations

import numpy as np
import pytest

from smolvla_analysis.phase3b_normalized_state import (
    compare_normalized_descriptors,
    rotation_distance_radians,
    vector_difference,
)


def _descriptor() -> dict:
    return {
        "bowl_position": [0.0, 0.0, 0.0],
        "bowl_orientation": np.eye(3).tolist(),
        "bowl_joint_qpos": [0.0] * 7,
        "bowl_joint_qvel": [0.0] * 6,
        "eef_position": [0.0, 0.0, 1.0],
        "eef_orientation": np.eye(3).tolist(),
        "eef_bowl_relative_position": [0.0, 0.0, 1.0],
        "robot_joint_positions": [0.0] * 7,
        "robot_joint_velocities": [0.0] * 7,
        "gripper_joint_positions": [0.0, 0.0],
        "gripper_joint_velocities": [0.0, 0.0],
        "drawer_joint": -0.15,
        "drawer_velocity": 0.0,
        "timestep": 1530,
        "bowl_contact_pairs": [["bowl", "table"]],
        "runtime_state_sha256": "runtime",
        "sim_data_sha256": "sim",
    }


def test_rotation_distance_radians_recovers_quarter_turn() -> None:
    quarter_turn = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    assert rotation_distance_radians(np.eye(3), quarter_turn) == pytest.approx(
        np.pi / 2.0
    )


def test_vector_difference_uses_second_minus_first() -> None:
    result = vector_difference([1.0, 2.0], [2.0, 0.0])
    assert result["difference"] == [1.0, -2.0]
    assert result["l2"] == pytest.approx(np.sqrt(5.0))
    assert result["max_abs"] == 2.0


def test_compare_normalized_descriptors_reports_state_axes() -> None:
    near = _descriptor()
    low = _descriptor()
    low["bowl_position"] = [0.001, -0.002, 0.0]
    low["robot_joint_positions"][3] = 0.01
    low["bowl_contact_pairs"] = [["bowl", "finger"]]
    low["runtime_state_sha256"] = "different"

    result = compare_normalized_descriptors(near, low)

    assert result["bowl_position_low_minus_near"]["l2"] == pytest.approx(
        np.sqrt(5.0) / 1000.0
    )
    assert result["robot_joint_positions_low_minus_near"]["max_abs"] == 0.01
    assert result["bowl_contact_pairs_equal"] is False
    assert result["runtime_state_sha256_equal"] is False
    assert result["sim_data_sha256_equal"] is True
