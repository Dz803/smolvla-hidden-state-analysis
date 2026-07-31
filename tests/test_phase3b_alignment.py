from __future__ import annotations

import numpy as np
import pytest

from smolvla_analysis.phase3b_alignment import (
    first_stable_true_index,
    landmark_registered_point,
)


def test_landmark_registration_preserves_relative_offset() -> None:
    point = np.asarray([1.0, 2.0, 3.0])
    reference = np.asarray([0.5, 1.0, 2.0])
    target = np.asarray([-1.0, 4.0, 2.5])
    registered = landmark_registered_point(point, reference, target)
    np.testing.assert_allclose(registered - target, point - reference)

    with pytest.raises(ValueError, match="finite 3-D"):
        landmark_registered_point(point[:2], reference, target)
    with pytest.raises(ValueError, match="finite 3-D"):
        landmark_registered_point(point, reference, np.asarray([0.0, np.nan, 0.0]))


def test_first_stable_true_index_is_explicit_about_streak() -> None:
    values = [False, True, False, True, True, True, False]
    assert first_stable_true_index(values, streak=1) == 1
    assert first_stable_true_index(values, streak=2) == 4
    assert first_stable_true_index(values, streak=3) == 5
    assert first_stable_true_index(values, streak=4) is None
    with pytest.raises(ValueError, match="positive"):
        first_stable_true_index(values, streak=0)
