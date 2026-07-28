from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn

from .activation_hooks import first_tensor


@dataclass(frozen=True)
class TokenSpan:
    name: str
    start: int
    stop: int
    valid_tokens_per_batch: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start": self.start,
            "stop": self.stop,
            "valid_tokens_per_batch": list(self.valid_tokens_per_batch),
        }


@dataclass(frozen=True)
class StructuredActivationRecord:
    module_name: str
    pathway: str
    layer_index: int
    invocation_index: int
    tensor_shape: tuple[int, ...]
    dtype: str
    values: np.ndarray


@dataclass(frozen=True)
class DenoisingRecord:
    invocation_index: int
    timestep: np.ndarray
    x_t: np.ndarray
    velocity: np.ndarray


@dataclass(frozen=True)
class KVCacheNorms:
    layer_index: int
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    key_token_l2: np.ndarray
    value_token_l2: np.ndarray


@dataclass(frozen=True)
class ActionQueryCapture:
    query_id: str
    flow_noise_seed: int
    flow_noise_sha256: str
    flow_noise: np.ndarray
    model_action_chunk: np.ndarray
    first_action: np.ndarray
    prefix_embeddings: np.ndarray
    prefix_pad_mask: np.ndarray
    prefix_attention_segments: np.ndarray
    action_head_inputs: np.ndarray
    token_spans: tuple[TokenSpan, ...]
    activations: tuple[StructuredActivationRecord, ...]
    denoising: tuple[DenoisingRecord, ...]
    kv_cache_norms: tuple[KVCacheNorms, ...]

    def activation_stack(self, pathway: str, layer_index: int) -> np.ndarray:
        selected = [
            record.values
            for record in self.activations
            if record.pathway == pathway and record.layer_index == layer_index
        ]
        if not selected:
            raise KeyError(f"No activations for pathway={pathway!r}, layer={layer_index}")
        return np.stack(selected)


class StructuredActivationCapture:
    """Capture complete token tensors while preserving repeated flow-step calls."""

    def __init__(self, model: nn.Module, targets: list[dict[str, Any]], image_connector: nn.Module):
        modules = dict(model.named_modules())
        self.records: list[StructuredActivationRecord] = []
        self.image_token_lengths: list[int] = []
        self._invocations: dict[str, int] = {}
        self._handles = []
        for target in targets:
            name = target["module_name"]
            if name not in modules:
                raise KeyError(f"Resolved activation module is absent: {name}")
            self._handles.append(modules[name].register_forward_hook(self._hook(target)))
        self._handles.append(image_connector.register_forward_hook(self._image_hook))

    def _hook(self, target: dict[str, Any]):
        def capture(_module, _inputs, output):
            tensor = first_tensor(output).detach()
            name = target["module_name"]
            invocation = self._invocations.get(name, 0)
            self._invocations[name] = invocation + 1
            self.records.append(
                StructuredActivationRecord(
                    module_name=name,
                    pathway=target["pathway"],
                    layer_index=int(target["layer_index"]),
                    invocation_index=invocation,
                    tensor_shape=tuple(tensor.shape),
                    dtype=str(tensor.dtype),
                    values=tensor.to(device="cpu", dtype=torch.float16).numpy(),
                )
            )

        return capture

    def _image_hook(self, _module, _inputs, output):
        tensor = first_tensor(output)
        if tensor.ndim < 3:
            raise ValueError(f"Image connector output must have a token axis, got {tuple(tensor.shape)}")
        self.image_token_lengths.append(int(tensor.shape[-2]))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def fixed_flow_noise(policy, batch: dict[str, Any], seed: int) -> torch.Tensor:
    state = batch["observation.state"]
    batch_size = int(state.shape[0]) if state.ndim > 1 else 1
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    shape = (batch_size, int(policy.config.chunk_size), int(policy.config.max_action_dim))
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(state.device)


def _replace_instance_method(instance, name: str, replacement) -> Iterator[None]:
    had_instance_value = name in instance.__dict__
    previous_instance_value = instance.__dict__.get(name)
    setattr(instance, name, replacement)
    try:
        yield
    finally:
        if had_instance_value:
            setattr(instance, name, previous_instance_value)
        else:
            delattr(instance, name)


@contextmanager
def _intercept_method(instance, name: str, replacement) -> Iterator[None]:
    yield from _replace_instance_method(instance, name, replacement)


def _kv_norms(past_key_values: Any) -> tuple[KVCacheNorms, ...]:
    if not isinstance(past_key_values, dict):
        return ()
    summaries = []
    for layer_index in sorted(past_key_values):
        layer = past_key_values[layer_index]
        if not isinstance(layer, dict) or "key_states" not in layer or "value_states" not in layer:
            continue
        key = layer["key_states"].detach().float()
        value = layer["value_states"].detach().float()
        summaries.append(
            KVCacheNorms(
                layer_index=int(layer_index),
                key_shape=tuple(key.shape),
                value_shape=tuple(value.shape),
                key_token_l2=torch.linalg.vector_norm(key.flatten(2), dim=-1).cpu().numpy(),
                value_token_l2=torch.linalg.vector_norm(value.flatten(2), dim=-1).cpu().numpy(),
            )
        )
    return tuple(summaries)


@contextmanager
def _capture_module_inputs(module: nn.Module, destination: list[np.ndarray]) -> Iterator[None]:
    def capture(_module, inputs):
        tensor = first_tensor(inputs).detach()
        destination.append(tensor.to(device="cpu", dtype=torch.float16).numpy())

    handle = module.register_forward_pre_hook(capture)
    try:
        yield
    finally:
        handle.remove()


def _token_spans(
    policy,
    batch: dict[str, Any],
    image_token_lengths: list[int],
    prefix_pad_mask: np.ndarray,
) -> tuple[TokenSpan, ...]:
    image_keys = [key for key in policy.config.image_features if key in batch]
    if len(image_keys) != len(image_token_lengths):
        raise RuntimeError(
            f"Image-token capture mismatch: keys={image_keys}, token_lengths={image_token_lengths}"
        )
    batch_size, prefix_length = prefix_pad_mask.shape
    spans = []
    cursor = 0
    for key, token_length in zip(image_keys, image_token_lengths, strict=True):
        if policy.config.add_image_special_tokens:
            spans.append(TokenSpan(f"{key}.start", cursor, cursor + 2, (2,) * batch_size))
            cursor += 2
        image_mask = batch.get(f"{key}_padding_mask")
        valid = (
            tuple(int(value) * token_length for value in image_mask.detach().cpu().bool().tolist())
            if image_mask is not None
            else (token_length,) * batch_size
        )
        spans.append(TokenSpan(key, cursor, cursor + token_length, valid))
        cursor += token_length
        if policy.config.add_image_special_tokens:
            spans.append(TokenSpan(f"{key}.end", cursor, cursor + 1, (1,) * batch_size))
            cursor += 1

    language_mask = batch["observation.language.attention_mask"].detach().cpu().bool().numpy()
    language_length = int(language_mask.shape[1])
    spans.append(
        TokenSpan(
            "observation.language",
            cursor,
            cursor + language_length,
            tuple(int(row.sum()) for row in language_mask),
        )
    )
    cursor += language_length
    spans.append(TokenSpan("observation.state", cursor, cursor + 1, (1,) * batch_size))
    cursor += 1
    if cursor < prefix_length:
        valid = tuple(int(row[cursor:].sum()) for row in prefix_pad_mask)
        spans.append(TokenSpan("padding", cursor, prefix_length, valid))
        cursor = prefix_length
    if cursor != prefix_length:
        raise RuntimeError(f"Token spans cover {cursor} tokens, but prefix contains {prefix_length}")
    return tuple(spans)


def capture_action_query(
    policy,
    batch: dict[str, Any],
    targets: list[dict[str, Any]],
    *,
    query_id: str,
    flow_noise_seed: int,
) -> tuple[torch.Tensor, ActionQueryCapture]:
    """Run and capture the exact forward that fills an empty SmolVLA action queue."""

    action_queue = policy._queues.get("action")
    if action_queue is None or len(action_queue) != 0:
        raise RuntimeError("capture_action_query requires an empty action queue")

    flow_model = policy.model
    image_connector = flow_model.vlm_with_expert.get_vlm_model().connector
    noise = fixed_flow_noise(policy, batch, flow_noise_seed)
    chunks: list[torch.Tensor] = []
    prefix_results: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    denoising: list[DenoisingRecord] = []
    action_head_inputs: list[np.ndarray] = []
    cache_summaries: list[tuple[KVCacheNorms, ...]] = []

    original_get_chunk = policy._get_action_chunk
    original_embed_prefix = flow_model.embed_prefix
    original_denoise_step = flow_model.denoise_step

    def get_chunk(*args, **kwargs):
        chunk = original_get_chunk(*args, **kwargs)
        chunks.append(chunk.detach().cpu())
        return chunk

    def embed_prefix(*args, **kwargs):
        result = original_embed_prefix(*args, **kwargs)
        prefix_results.append(tuple(item.detach().cpu() for item in result))
        return result

    def denoise_step(*args, **kwargs):
        x_t = kwargs.get("x_t", args[2] if len(args) > 2 else None)
        timestep = kwargs.get("timestep", args[3] if len(args) > 3 else None)
        past_key_values = kwargs.get("past_key_values", args[1] if len(args) > 1 else None)
        velocity = original_denoise_step(*args, **kwargs)
        denoising.append(
            DenoisingRecord(
                invocation_index=len(denoising),
                timestep=timestep.detach().float().cpu().numpy(),
                x_t=x_t.detach().to(device="cpu", dtype=torch.float16).numpy(),
                velocity=velocity.detach().to(device="cpu", dtype=torch.float16).numpy(),
            )
        )
        if not cache_summaries:
            cache_summaries.append(_kv_norms(past_key_values))
        return velocity

    capture = StructuredActivationCapture(policy, targets, image_connector)
    with ExitStack() as stack:
        stack.enter_context(capture)
        stack.enter_context(_intercept_method(policy, "_get_action_chunk", get_chunk))
        stack.enter_context(_intercept_method(flow_model, "embed_prefix", embed_prefix))
        stack.enter_context(_intercept_method(flow_model, "denoise_step", denoise_step))
        stack.enter_context(_capture_module_inputs(flow_model.action_out_proj, action_head_inputs))
        with torch.inference_mode():
            first_action = policy.select_action(batch, noise=noise)

    if len(chunks) != 1 or len(prefix_results) != 1:
        raise RuntimeError(
            f"Expected one action chunk and one prefix, got chunks={len(chunks)}, prefixes={len(prefix_results)}"
        )
    expected_steps = int(policy.config.num_steps)
    if len(denoising) != expected_steps:
        raise RuntimeError(f"Expected {expected_steps} denoising calls, got {len(denoising)}")
    if len(action_head_inputs) != expected_steps:
        raise RuntimeError(f"Expected {expected_steps} action-head inputs, got {len(action_head_inputs)}")

    prefix_embeddings, prefix_pad_mask, prefix_attention_segments = prefix_results[0]
    prefix_pad_array = prefix_pad_mask.bool().numpy()
    spans = _token_spans(policy, batch, capture.image_token_lengths, prefix_pad_array)
    noise_cpu = noise.detach().float().cpu().numpy()
    query = ActionQueryCapture(
        query_id=query_id,
        flow_noise_seed=int(flow_noise_seed),
        flow_noise_sha256=sha256(np.ascontiguousarray(noise_cpu).tobytes()).hexdigest(),
        flow_noise=noise_cpu,
        model_action_chunk=chunks[0].float().numpy(),
        first_action=first_action.detach().float().cpu().numpy(),
        prefix_embeddings=prefix_embeddings.to(dtype=torch.float16).numpy(),
        prefix_pad_mask=prefix_pad_array,
        prefix_attention_segments=prefix_attention_segments.bool().numpy(),
        action_head_inputs=np.stack(action_head_inputs),
        token_spans=spans,
        activations=tuple(capture.records),
        denoising=tuple(denoising),
        kv_cache_norms=cache_summaries[0] if cache_summaries else (),
    )
    return first_action, query
