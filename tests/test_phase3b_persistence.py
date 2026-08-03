from __future__ import annotations

import json
import os

import numpy as np
import zarr

from smolvla_analysis.libero_state import LiberoStateSnapshot
from smolvla_analysis.phase3b_persistence import (
    recover_candidate_transactions,
    stage_candidate_transaction,
)
from smolvla_analysis.phase3b_stage_a import snapshot_sha256


def _snapshot() -> LiberoStateSnapshot:
    return LiberoStateSnapshot(
        mujoco_state=np.asarray([1.0, 2.0, 3.0]),
        objects={},
        goal_predicates=(),
        contacts=(),
        grasped_objects=(),
        success=False,
        runtime_state={"environment": {"timestep": 560}},
        numpy_random_state={},
    )


def test_candidate_transaction_recovers_state_record_interruption(tmp_path) -> None:
    candidate_id = "candidate"
    snapshot = _snapshot()
    record = {
        "candidate_id": candidate_id,
        "state_sha256": snapshot_sha256(snapshot),
    }
    transaction = stage_candidate_transaction(
        tmp_path,
        candidate_id=candidate_id,
        snapshot=snapshot,
        record=record,
    )

    final_root_path = tmp_path / "states.zarr"
    zarr.open_group(str(final_root_path), mode="a")
    os.replace(
        transaction / "state.zarr" / candidate_id,
        final_root_path / candidate_id,
    )
    assert not (tmp_path / "candidates" / f"{candidate_id}.json").exists()

    assert recover_candidate_transactions(tmp_path) == [candidate_id]
    assert json.loads(
        (tmp_path / "candidates" / f"{candidate_id}.json").read_text()
    ) == record
    assert json.loads((transaction / "intent.json").read_text())["status"] == (
        "committed"
    )
    assert recover_candidate_transactions(tmp_path) == []
