from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from torch import nn


def first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return first_tensor(item)
            except TypeError:
                continue
    if isinstance(value, dict):
        for item in value.values():
            try:
                return first_tensor(item)
            except TypeError:
                continue
    raise TypeError(f"Hook output contains no tensor: {type(value).__name__}")


@dataclass(frozen=True)
class ActivationRecord:
    module_name: str
    pathway: str
    layer_index: int
    tensor_shape: tuple[int, ...]
    dtype: str
    pooling_method: str
    pooled: np.ndarray
    l2_norm: float
    mean: float
    std: float


class ActivationCapture:
    """Read-only forward hooks that immediately pool and transfer activations to CPU."""

    def __init__(self, model: nn.Module, targets: list[dict[str, Any]]):
        modules = dict(model.named_modules())
        self.records: list[ActivationRecord] = []
        self.handles = []
        for target in targets:
            name = target["module_name"]
            if name not in modules:
                raise KeyError(f"Resolved activation module is absent: {name}")
            self.handles.append(modules[name].register_forward_hook(self._hook(target)))

    def _hook(self, target: dict[str, Any]) -> Callable:
        def capture(_module, _inputs, output):
            tensor = first_tensor(output).detach()
            cpu = tensor.float().cpu()
            reduce_dims = tuple(range(max(cpu.ndim - 1, 0)))
            pooled = cpu.mean(dim=reduce_dims) if reduce_dims else cpu
            self.records.append(
                ActivationRecord(
                    module_name=target["module_name"],
                    pathway=target["pathway"],
                    layer_index=int(target["layer_index"]),
                    tensor_shape=tuple(tensor.shape),
                    dtype=str(tensor.dtype),
                    pooling_method="mean_all_non_feature_dimensions",
                    pooled=pooled.numpy().astype(np.float16, copy=False),
                    l2_norm=float(torch.linalg.vector_norm(cpu)),
                    mean=float(cpu.mean()),
                    std=float(cpu.std(unbiased=False)),
                )
            )

        return capture

    def clear(self) -> None:
        self.records.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def assert_hook_equivalence(
    inference: Callable[[], torch.Tensor], capture_factory: Callable[[], ActivationCapture], *, atol=1e-5, rtol=1e-5
) -> dict[str, float | int]:
    torch.manual_seed(0)
    before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    baseline = inference().detach().cpu()
    torch.manual_seed(0)
    with capture_factory() as capture:
        hooked = inference().detach().cpu()
        record_count = len(capture.records)
    after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    if not torch.allclose(baseline, hooked, atol=atol, rtol=rtol):
        maximum = float((baseline - hooked).abs().max())
        raise AssertionError(f"Hooks changed model output; max_abs_diff={maximum}")
    growth = max(0, after - before)
    return {
        "max_abs_diff": float((baseline - hooked).abs().max()),
        "records": record_count,
        "persistent_gpu_memory_growth_bytes": growth,
    }

