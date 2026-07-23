#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from smolvla_analysis.activation_hooks import ActivationCapture, assert_hook_equivalence
from smolvla_analysis.config import load_config
from smolvla_analysis.model_inspection import candidate_layer_groups, freeze_policy, parameter_counts, resolve_pathways


PROJECT = Path(__file__).resolve().parents[1]


def synthetic_batch(preprocessor):
    observation = {
        # The serialized normalizer and current LIBERO adapter are 8-D even though
        # this checkpoint's config.json incorrectly declares 6-D state features.
        "observation.state": torch.zeros(8),
        "observation.images.camera1": torch.zeros(3, 256, 256),
        "observation.images.camera2": torch.zeros(3, 256, 256),
        "task": "pick up the object and place it in the target",
    }
    return preprocessor(observation)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/smoke.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT / "reports/model_modules.json")
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = PROJECT / config["model"]["local_path"]

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy_config = PreTrainedConfig.from_pretrained(checkpoint)
    policy_config.device = config["model"]["device"]
    policy_config.n_action_steps = config["model"]["n_action_steps"]
    # The checkpoint contains the complete VLM. Initializing from architecture avoids
    # downloading the separate base checkpoint before these frozen weights are loaded.
    policy_config.load_vlm_weights = False
    processor_assets = PROJECT / "checkpoints/smolvlm_processor"
    policy_config.vlm_model_name = str(processor_assets)
    policy = SmolVLAPolicy.from_pretrained(checkpoint, config=policy_config).to(policy_config.device)
    freeze_policy(policy)
    counts = parameter_counts(policy)
    groups = candidate_layer_groups(policy)
    targets = resolve_pathways(policy, config["activations"]["relative_layer_positions"])
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=checkpoint,
        preprocessor_overrides={
            "device_processor": {"device": policy_config.device},
            "tokenizer_processor": {"tokenizer_name": str(processor_assets)},
        },
    )
    batch = synthetic_batch(preprocessor)

    def infer():
        policy.reset()
        torch.manual_seed(config["project"]["seed"])
        return policy.predict_action_chunk(batch)

    with torch.inference_mode():
        action = postprocessor(infer())
        equivalence = assert_hook_equivalence(infer, lambda: ActivationCapture(policy, targets), atol=1e-5, rtol=1e-5)
    if action.shape[-1] != 7 or not torch.isfinite(action).all():
        raise RuntimeError(f"Invalid action output: shape={tuple(action.shape)} finite={torch.isfinite(action).all()}")
    result = {
        "checkpoint": str(checkpoint.resolve()), "frozen": not any(parameter.requires_grad for parameter in policy.parameters()),
        "state_contract": {
            "config_declared_dim": policy_config.input_features["observation.state"].shape[0],
            "normalizer_and_runtime_dim": batch["observation.state"].shape[-1],
            "resolution": "use serialized 8-D normalization statistics and official LIBERO 8-D state",
        },
        "parameter_counts": counts, "candidate_layer_groups": groups, "resolved_targets": targets,
        "input_shapes": {key: list(value.shape) if hasattr(value, "shape") else str(type(value)) for key, value in batch.items()},
        "action_chunk_shape": list(action.shape), "action_dtype": str(action.dtype), "actions_finite": True,
        "hook_equivalence": equivalence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
