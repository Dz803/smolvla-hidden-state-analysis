from __future__ import annotations

import numpy as np


def orient_archived_camera_for_policy(camera_chw: np.ndarray) -> np.ndarray:
    """Apply LeRobot's LIBERO 180-degree camera transform to a saved CHW frame."""

    camera = np.asarray(camera_chw)
    if camera.ndim != 3 or camera.shape[0] not in {1, 3, 4}:
        raise ValueError(f"Expected a CHW camera frame, got shape {camera.shape}")
    return np.flip(camera, axis=(-2, -1)).copy()
