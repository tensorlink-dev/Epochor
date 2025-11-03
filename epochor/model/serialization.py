"""Utility helpers mimicking a subset of the Hugging Face API."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Type

import torch

try:
    from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
except ImportError:  # pragma: no cover - optional dependency
    load_safetensors = None
    save_safetensors = None


def _config_to_dict(config: Any) -> Dict[str, Any]:
    if config is None:
        return {}
    if hasattr(config, "to_dict"):
        return config.to_dict()
    if hasattr(config, "__dict__"):
        return {
            key: value
            for key, value in config.__dict__.items()
            if not key.startswith("_") and not callable(value)
        }
    raise TypeError("Config object must implement to_dict or expose attributes via __dict__.")


def save_hf(
    *,
    model: torch.nn.Module,
    config: Any,
    save_directory: str,
    safe: bool = False,
) -> None:
    os.makedirs(save_directory, exist_ok=True)

    with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(_config_to_dict(config), handle, indent=2)

    weights_name = "model.safetensors" if safe else "pytorch_model.bin"
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    weights_path = os.path.join(save_directory, weights_name)

    if safe:
        if save_safetensors is None:
            raise RuntimeError("safetensors is required for safe serialization but is not installed")
        save_safetensors(state_dict, weights_path)
    else:
        torch.save(state_dict, weights_path)


def _load_config(config_cls: Type, config_path: str) -> Any:
    if not os.path.exists(config_path):
        return config_cls()
    with open(config_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if hasattr(config_cls, "from_dict"):
        return config_cls.from_dict(data)
    return config_cls(**data)


def load_hf(
    *,
    model_name_or_path: str,
    model_cls: Type[torch.nn.Module],
    config_cls: Type,
    safe: bool = False,
    map_location: str | torch.device = "cpu",
) -> torch.nn.Module:
    config = _load_config(config_cls, os.path.join(model_name_or_path, "config.json"))
    model = model_cls(config=config)

    weights_name = "model.safetensors" if safe else "pytorch_model.bin"
    weights_path = os.path.join(model_name_or_path, weights_name)
    if safe:
        if load_safetensors is None:
            raise RuntimeError("safetensors is required for safe deserialization but is not installed")
        state = load_safetensors(weights_path, device=map_location)
    else:
        state = torch.load(weights_path, map_location=map_location)
    model.load_state_dict(state, strict=False)
    return model
