from __future__ import annotations

from smolvla_analysis.libero_state import LiberoStateSnapshot
from smolvla_analysis.phase3b_stage_a import snapshot_sha256
from scripts.diagnose_phase3b_state_swap import _runtime_cross

import numpy as np


def _snapshot(*, state: float, runtime: str) -> LiberoStateSnapshot:
    return LiberoStateSnapshot(
        mujoco_state=np.asarray([state], dtype=np.float64),
        objects={},
        goal_predicates=(),
        contacts=(),
        grasped_objects=(),
        success=False,
        runtime_state={"source": runtime},
        numpy_random_state={},
    )


def test_runtime_cross_keeps_physics_and_swaps_runtime() -> None:
    near = _snapshot(state=1.0, runtime="near")
    low = _snapshot(state=2.0, runtime="low")

    crossed = _runtime_cross(near, low)

    assert crossed.mujoco_state.tolist() == [1.0]
    assert crossed.runtime_state == {"source": "low"}
    assert crossed.objects is near.objects
    assert snapshot_sha256(crossed) not in {
        snapshot_sha256(near),
        snapshot_sha256(low),
    }


def test_runtime_cross_deep_copies_runtime_payload() -> None:
    near = _snapshot(state=1.0, runtime="near")
    low = _snapshot(state=2.0, runtime="low")

    crossed = _runtime_cross(near, low)
    crossed.runtime_state["source"] = "changed"

    assert low.runtime_state == {"source": "low"}
