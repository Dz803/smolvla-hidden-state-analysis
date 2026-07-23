from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any

from torch import nn


LAYER_PATTERN = re.compile(r"^(.*(?:text_model\.layers|lm_expert\.layers))\.(\d+)$")


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_before_freeze": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def freeze_policy(model: nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def candidate_layer_groups(model: nn.Module) -> dict[str, list[tuple[int, str]]]:
    groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for name, module in model.named_modules():
        match = LAYER_PATTERN.match(name)
        if match:
            groups[match.group(1)].append((int(match.group(2)), name))
    return {key: sorted(value) for key, value in groups.items()}


def resolve_pathways(model: nn.Module, positions: list[float]) -> list[dict[str, Any]]:
    groups = candidate_layer_groups(model)
    module_names = dict(model.named_modules())
    vlm_groups = {key: value for key, value in groups.items() if "lm_expert" not in key}
    expert_groups = {key: value for key, value in groups.items() if "lm_expert" in key}
    if not vlm_groups or not expert_groups:
        raise RuntimeError(f"Could not identify both pathways from named_modules(); groups={list(groups)}")
    vlm = max(vlm_groups.values(), key=len)
    expert = max(expert_groups.values(), key=len)
    resolved: list[dict[str, Any]] = []
    for pathway, layers in (("vlm", vlm), ("action_expert", expert)):
        for position in positions:
            offset = min(len(layers) - 1, max(0, math.ceil(position * len(layers)) - 1))
            layer_index, block_name = layers[offset]
            # SmolVLA's joint VLM/expert forward calls transformer submodules
            # directly, so a full-block hook never fires. The post-attention
            # norm is executed once per layer and exposes each pathway's
            # normalized residual stream before the MLP.
            norm_name = f"{block_name}.post_attention_layernorm"
            module_name = norm_name if norm_name in module_names else block_name
            resolved.append(
                {
                    "pathway": pathway,
                    "relative_position": position,
                    "layer_index": layer_index,
                    "block_name": block_name,
                    "module_name": module_name,
                    "capture_point": "post_attention_layernorm_output" if module_name == norm_name else "block_output",
                }
            )
    return resolved
