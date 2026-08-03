"""Compact diagnostics for matched LIBERO normalized states."""

from __future__ import annotations

from typing import Any

import numpy as np


def rotation_distance_radians(first: Any, second: Any) -> float:
    """Return the geodesic distance between two finite 3-D rotations."""

    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if (
        first_array.shape != (3, 3)
        or second_array.shape != (3, 3)
        or not np.isfinite(first_array).all()
        or not np.isfinite(second_array).all()
    ):
        raise ValueError("Rotations must be finite 3x3 matrices")
    relative = first_array @ second_array.T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def vector_difference(first: Any, second: Any) -> dict[str, Any]:
    """Summarize ``second - first`` without retaining large raw state arrays."""

    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if (
        first_array.shape != second_array.shape
        or not np.isfinite(first_array).all()
        or not np.isfinite(second_array).all()
    ):
        raise ValueError("Compared vectors must have the same finite shape")
    difference = second_array - first_array
    return {
        "difference": difference.tolist(),
        "l2": float(np.linalg.norm(difference)),
        "rms": float(np.sqrt(np.mean(np.square(difference))))
        if difference.size
        else 0.0,
        "max_abs": float(np.max(np.abs(difference), initial=0.0)),
    }


def compare_normalized_descriptors(
    near: dict[str, Any], low: dict[str, Any]
) -> dict[str, Any]:
    """Compare compact descriptors for a matched near/low normalized pair."""

    required_vectors = (
        "bowl_position",
        "bowl_joint_qpos",
        "bowl_joint_qvel",
        "eef_position",
        "eef_bowl_relative_position",
        "robot_joint_positions",
        "robot_joint_velocities",
        "gripper_joint_positions",
        "gripper_joint_velocities",
    )
    missing = [
        key for key in required_vectors if key not in near or key not in low
    ]
    if missing:
        raise ValueError(f"Normalized descriptors are missing: {missing}")
    comparison = {
        f"{key}_low_minus_near": vector_difference(near[key], low[key])
        for key in required_vectors
    }
    comparison.update(
        {
            "bowl_orientation_distance_rad": rotation_distance_radians(
                near["bowl_orientation"], low["bowl_orientation"]
            ),
            "eef_orientation_distance_rad": rotation_distance_radians(
                near["eef_orientation"], low["eef_orientation"]
            ),
            "drawer_joint_low_minus_near": float(
                low["drawer_joint"] - near["drawer_joint"]
            ),
            "drawer_velocity_low_minus_near": float(
                low["drawer_velocity"] - near["drawer_velocity"]
            ),
            "timestep_low_minus_near": int(
                low["timestep"] - near["timestep"]
            ),
            "bowl_contact_pairs_near": near["bowl_contact_pairs"],
            "bowl_contact_pairs_low": low["bowl_contact_pairs"],
            "bowl_contact_pairs_equal": (
                near["bowl_contact_pairs"] == low["bowl_contact_pairs"]
            ),
            "runtime_state_sha256_equal": (
                near["runtime_state_sha256"]
                == low["runtime_state_sha256"]
            ),
            "sim_data_sha256_equal": (
                near["sim_data_sha256"] == low["sim_data_sha256"]
            ),
        }
    )
    return comparison
