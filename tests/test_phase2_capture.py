from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zarr
from torch import nn

from smolvla_analysis.phase2_capture import capture_action_query, fixed_flow_noise
from smolvla_analysis.phase2_storage import write_action_query


class ToyVLMWithExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.connector = nn.Identity()

    def get_vlm_model(self):
        return self


class ToyFlow(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vlm_with_expert = ToyVLMWithExpert()
        self.vlm_norm = nn.Identity()
        self.expert_norm = nn.Identity()
        self.action_out_proj = nn.Identity()

    def embed_prefix(self, images, language_mask, state):
        image_embeddings = [self.vlm_with_expert.connector(image) for image in images]
        batch_size = state.shape[0]
        language = torch.zeros(batch_size, language_mask.shape[1], 2)
        state_embedding = state[:, None, :2]
        embeddings = torch.cat([*image_embeddings, language, state_embedding], dim=1)
        image_masks = [torch.ones(batch_size, image.shape[1], dtype=torch.bool) for image in images]
        pad_mask = torch.cat(
            [*image_masks, language_mask.bool(), torch.ones(batch_size, 1, dtype=torch.bool)], dim=1
        )
        segments = torch.zeros_like(pad_mask)
        segments[:, -1] = True
        return embeddings, pad_mask, segments

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        del prefix_pad_masks, past_key_values, timestep
        return self.action_out_proj(self.expert_norm(x_t)) * 0.25

    def sample_actions(self, batch, noise):
        prefix, prefix_mask, _ = self.embed_prefix(
            [batch["camera1"], batch["camera2"]],
            batch["observation.language.attention_mask"],
            batch["observation.state"],
        )
        prefix = self.vlm_norm(prefix)
        cache = {
            0: {
                "key_states": prefix[:, :, None, :],
                "value_states": (prefix + 1)[:, :, None, :],
            }
        }
        x_t = noise
        for step in range(self.config.num_steps):
            timestep = torch.full((x_t.shape[0],), 1.0 - step / self.config.num_steps)
            velocity = self.denoise_step(
                prefix_pad_masks=prefix_mask,
                past_key_values=cache,
                x_t=x_t,
                timestep=timestep,
            )
            x_t = x_t - velocity / self.config.num_steps
        return x_t


class ToyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            chunk_size=3,
            max_action_dim=4,
            num_steps=2,
            image_features=("camera1", "camera2"),
            add_image_special_tokens=False,
        )
        self.model = ToyFlow(self.config)
        self.reset()

    def reset(self):
        self._queues = {"action": deque(maxlen=self.config.chunk_size)}

    def _get_action_chunk(self, batch, noise=None):
        return self.model.sample_actions(batch, noise)[:, :, :2]

    def select_action(self, batch, noise=None):
        if not self._queues["action"]:
            chunk = self._get_action_chunk(batch, noise)
            self._queues["action"].extend(chunk.transpose(0, 1))
        return self._queues["action"].popleft()


def batch():
    return {
        "camera1": torch.ones(1, 2, 2),
        "camera2": torch.full((1, 2, 2), 2.0),
        "observation.language.attention_mask": torch.tensor([[True, True, False]]),
        "observation.state": torch.tensor([[0.5, -0.5]]),
    }


TARGETS = [
    {"module_name": "model.vlm_norm", "pathway": "vlm", "layer_index": 0},
    {"module_name": "model.expert_norm", "pathway": "action_expert", "layer_index": 0},
]


def test_exact_query_capture_preserves_tokens_flow_steps_and_chunk():
    policy = ToyPolicy()
    first_action, query = capture_action_query(
        policy, batch(), TARGETS, query_id="query-0", flow_noise_seed=17
    )

    np.testing.assert_allclose(query.first_action, first_action.numpy())
    np.testing.assert_allclose(query.model_action_chunk[:, 0], query.first_action)
    assert len(policy._queues["action"]) == 2
    assert [span.name for span in query.token_spans] == [
        "camera1",
        "camera2",
        "observation.language",
        "observation.state",
    ]
    assert [(span.start, span.stop) for span in query.token_spans] == [(0, 2), (2, 4), (4, 7), (7, 8)]
    assert query.token_spans[2].valid_tokens_per_batch == (2,)
    assert query.activation_stack("vlm", 0).shape == (1, 1, 8, 2)
    assert query.activation_stack("action_expert", 0).shape == (2, 1, 3, 4)
    assert query.action_head_inputs.shape == (2, 1, 3, 4)
    np.testing.assert_allclose([record.timestep[0] for record in query.denoising], [1.0, 0.5])
    assert query.kv_cache_norms[0].key_token_l2.shape == (1, 8)


def test_fixed_noise_is_reproducible_and_queue_guard_is_enforced():
    policy = ToyPolicy()
    first = fixed_flow_noise(policy, batch(), 5)
    second = fixed_flow_noise(policy, batch(), 5)
    different = fixed_flow_noise(policy, batch(), 6)
    assert torch.equal(first, second)
    assert not torch.equal(first, different)

    capture_action_query(policy, batch(), TARGETS, query_id="query-0", flow_noise_seed=5)
    with pytest.raises(RuntimeError, match="empty action queue"):
        capture_action_query(policy, batch(), TARGETS, query_id="query-1", flow_noise_seed=5)


def test_structured_query_storage_preserves_axes(tmp_path):
    policy = ToyPolicy()
    _, query = capture_action_query(
        policy, batch(), TARGETS, query_id="query-0", flow_noise_seed=11
    )
    store = zarr.open_group(str(tmp_path / "queries.zarr"), mode="w")
    write_action_query(store, query, query.model_action_chunk)

    saved = store["query-0"]
    assert saved["activations/action_expert_layer_00"].shape == (2, 1, 3, 4)
    assert saved["denoising_x_t"].shape == (2, 1, 3, 4)
    assert saved["action_head_inputs"].shape == (2, 1, 3, 4)
    assert saved["kv_cache/key_token_l2"].shape == (1, 1, 8)
    assert saved.attrs["flow_noise_sha256"] == query.flow_noise_sha256
    assert saved.attrs["query_id"] == query.query_id
    assert saved.attrs["complete"] is True


def test_structured_query_storage_supports_transactional_group_key(tmp_path):
    policy = ToyPolicy()
    _, query = capture_action_query(
        policy, batch(), TARGETS, query_id="query-0", flow_noise_seed=11
    )
    store = zarr.open_group(str(tmp_path / "queries.zarr"), mode="w")
    write_action_query(store, query, query.model_action_chunk, group_key=".partial__query-0")
    store.move(".partial__query-0", "query-0")

    assert store["query-0"].attrs["query_id"] == "query-0"
    assert store["query-0"].attrs["complete"] is True
