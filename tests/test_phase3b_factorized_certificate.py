from __future__ import annotations

import numpy as np
import pytest

from scripts.diagnose_phase3b_factorized_certificate import (
    goal_registered_target,
)


def test_goal_registered_target_preserves_relative_offset() -> None:
    reference_bowl = np.asarray([1.0, 2.0, 3.0])
    reference_goal = np.asarray([0.5, 1.5, 2.5])
    target_goal = np.asarray([-1.0, 4.0, 8.0])

    target = goal_registered_target(
        reference_bowl, reference_goal, target_goal
    )

    assert target.tolist() == pytest.approx([-0.5, 4.5, 8.5])
    assert (target - target_goal).tolist() == pytest.approx(
        (reference_bowl - reference_goal).tolist()
    )


def test_goal_registered_target_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="three finite 3-vectors"):
        goal_registered_target([0.0, 1.0], [0.0] * 3, [0.0] * 3)
