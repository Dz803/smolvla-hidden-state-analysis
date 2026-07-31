from __future__ import annotations

from typing import Iterable

import numpy as np


def landmark_registered_point(
    reference_point: np.ndarray,
    reference_landmark: np.ndarray,
    target_landmark: np.ndarray,
) -> np.ndarray:
    """Translate a world-frame point while preserving landmark-relative offset."""

    point = np.asarray(reference_point, dtype=np.float64)
    reference = np.asarray(reference_landmark, dtype=np.float64)
    target = np.asarray(target_landmark, dtype=np.float64)
    if any(value.shape != (3,) for value in (point, reference, target)):
        raise ValueError("Landmark registration requires three finite 3-D points")
    if not all(np.isfinite(value).all() for value in (point, reference, target)):
        raise ValueError("Landmark registration requires three finite 3-D points")
    return target + (point - reference)


def first_stable_true_index(values: Iterable[bool], *, streak: int) -> int | None:
    """Return the first index ending a fixed consecutive-true streak."""

    if streak < 1:
        raise ValueError("Stable-event streak must be positive")
    current = 0
    for index, value in enumerate(values):
        current = current + 1 if bool(value) else 0
        if current >= streak:
            return index
    return None
