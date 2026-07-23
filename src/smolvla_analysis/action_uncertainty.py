from __future__ import annotations

import numpy as np


def chunk_uncertainty(chunks) -> dict[str, object]:
    values = np.asarray(chunks, dtype=float)
    if values.ndim != 3 or values.shape[-1] < 7:
        raise ValueError("chunks must have shape [samples, horizon, action_dim>=7]")
    if values.shape[0] < 2:
        raise ValueError("at least two independent samples are required")
    pairwise = []
    for left in range(values.shape[0]):
        for right in range(left + 1, values.shape[0]):
            pairwise.append(np.linalg.norm(values[left] - values[right]))
    gripper_sign = values[..., 6] >= 0
    agreement = np.maximum(gripper_sign.mean(axis=0), 1 - gripper_sign.mean(axis=0))
    return {
        "translation_variance": float(values[..., :3].var(axis=0).mean()),
        "rotation_variance": float(values[..., 3:6].var(axis=0).mean()),
        "gripper_disagreement": float((1 - agreement).mean()),
        "mean_pairwise_chunk_distance": float(np.mean(pairwise)),
        "variance_over_horizon": values.var(axis=0).mean(axis=1).tolist(),
    }

