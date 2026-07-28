from __future__ import annotations

import json

import numpy as np

from .libero_state import LiberoStateSnapshot
from .phase2_capture import ActionQueryCapture


def write_action_query(
    group,
    query: ActionQueryCapture,
    environment_action_chunk: np.ndarray | None = None,
    *,
    group_key: str | None = None,
) -> int:
    """Write one query without overwriting an existing group.

    ``group_key`` lets callers stage the write under a temporary key and move
    it into place only after every array has been persisted.
    """

    query_group = group.create_group(group_key or query.query_id)
    query_group.attrs.update(
        {
            "schema_version": 1,
            "query_id": query.query_id,
            "flow_noise_seed": query.flow_noise_seed,
            "flow_noise_sha256": query.flow_noise_sha256,
            "token_spans_json": json.dumps([span.as_dict() for span in query.token_spans]),
        }
    )
    arrays = {
        "flow_noise": query.flow_noise,
        "model_action_chunk": query.model_action_chunk,
        "first_action": query.first_action,
        "prefix_embeddings": query.prefix_embeddings,
        "prefix_pad_mask": query.prefix_pad_mask,
        "prefix_attention_segments": query.prefix_attention_segments,
        "action_head_inputs": query.action_head_inputs,
        "denoising_timestep": np.stack([record.timestep for record in query.denoising]),
        "denoising_x_t": np.stack([record.x_t for record in query.denoising]),
        "denoising_velocity": np.stack([record.velocity for record in query.denoising]),
    }
    if environment_action_chunk is not None:
        arrays["environment_action_chunk"] = np.asarray(environment_action_chunk)
    for name, values in arrays.items():
        query_group.create_dataset(name, data=values, overwrite=False)

    activation_group = query_group.create_group("activations")
    grouped: dict[tuple[str, int], list] = {}
    for record in query.activations:
        grouped.setdefault((record.pathway, record.layer_index), []).append(record)
    for (pathway, layer_index), records in grouped.items():
        records.sort(key=lambda record: record.invocation_index)
        array = activation_group.create_dataset(
            f"{pathway}_layer_{layer_index:02d}",
            data=np.stack([record.values for record in records]),
            overwrite=False,
        )
        array.attrs.update(
            {
                "module_name": records[0].module_name,
                "source_dtype": records[0].dtype,
                "invocation_indices": [record.invocation_index for record in records],
                "axes": ["invocation", "batch", "token_or_action_position", "hidden"],
            }
        )

    if query.kv_cache_norms:
        cache_group = query_group.create_group("kv_cache")
        cache_group.attrs["layer_indices"] = [record.layer_index for record in query.kv_cache_norms]
        cache_group.create_dataset(
            "key_token_l2",
            data=np.stack([record.key_token_l2 for record in query.kv_cache_norms]),
            overwrite=False,
        )
        cache_group.create_dataset(
            "value_token_l2",
            data=np.stack([record.value_token_l2 for record in query.kv_cache_norms]),
            overwrite=False,
        )
    query_group.attrs["complete"] = True
    return len(arrays) + len(grouped) + (2 if query.kv_cache_norms else 0)


def write_libero_snapshot(group, step_key: str, snapshot: LiberoStateSnapshot) -> int:
    step_group = group.create_group(step_key)
    step_group.create_dataset("mujoco_state", data=snapshot.mujoco_state, overwrite=False)
    step_group.attrs["metadata_json"] = json.dumps(snapshot.metadata(), sort_keys=True)
    return 1


def read_libero_snapshot(group, step_key: str) -> LiberoStateSnapshot:
    step_group = group[step_key]
    metadata = json.loads(step_group.attrs["metadata_json"])
    return LiberoStateSnapshot(
        mujoco_state=np.asarray(step_group["mujoco_state"][:], dtype=np.float64),
        objects=metadata["objects"],
        goal_predicates=tuple(metadata["goal_predicates"]),
        contacts=tuple(metadata["contacts"]),
        grasped_objects=tuple(metadata["grasped_objects"]),
        success=bool(metadata["success"]),
        runtime_state=metadata.get("runtime_state", {}),
        numpy_random_state=metadata.get("numpy_random_state", {}),
    )
