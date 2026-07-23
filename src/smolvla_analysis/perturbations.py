from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn.functional as functional


CONDITIONS = {"clean", "wrist_camera_mask", "main_camera_blur", "instruction_paraphrase"}


def _paraphrase(instruction: str) -> str:
    text = instruction.replace("pick up", "grasp", 1)
    text = text.replace("and place it", "and then put it", 1)
    return text


def apply_perturbation(observation: dict, condition: str, parameters: dict) -> dict:
    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported perturbation condition: {condition}")
    if condition == "clean":
        return observation
    result = deepcopy(observation)
    if condition == "wrist_camera_mask":
        key = "observation.images.image2"
        if key not in result:
            raise KeyError(f"Missing wrist-camera observation: {key}")
        value = float(parameters.get("mask_value", 0))
        result[key] = torch.full_like(result[key], value)
    elif condition == "main_camera_blur":
        key = "observation.images.image"
        if key not in result:
            raise KeyError(f"Missing main-camera observation: {key}")
        kernel = int(parameters.get("kernel_size", 7))
        if kernel < 3 or kernel % 2 == 0:
            raise ValueError("main_camera_blur.kernel_size must be an odd integer >= 3")
        result[key] = functional.avg_pool2d(result[key], kernel, stride=1, padding=kernel // 2)
    elif condition == "instruction_paraphrase":
        if int(parameters.get("version", 1)) != 1:
            raise ValueError("Only instruction paraphrase version 1 is implemented")
        task = result.get("task")
        if isinstance(task, list):
            result["task"] = [_paraphrase(item) for item in task]
        elif isinstance(task, str):
            result["task"] = _paraphrase(task)
        else:
            raise TypeError("Instruction paraphrase requires task as a string or list of strings")
    return result
