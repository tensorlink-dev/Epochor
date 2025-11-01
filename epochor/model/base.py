"""Core abstractions shared by miner time-series models."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn


class TemporalModelOutput(dict):
    """Lightweight container mirroring Hugging Face style outputs."""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - mirrors dict semantics
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        if key.startswith("_"):
            super().__setattr__(key, value)
        else:
            self[key] = value


class BaseTemporalModel(nn.Module, ABC):
    """Model-agnostic base class for temporal miners."""

    def __init__(self, config: Optional[Any] = None):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self, *args: Any, **kwargs: Any
    ) -> Union[TemporalModelOutput, Dict[str, torch.Tensor], torch.Tensor]:
        """Sub-classes must implement their forward pass."""

    def _to_output(
        self, out: Union[TemporalModelOutput, Dict[str, torch.Tensor], torch.Tensor]
    ) -> TemporalModelOutput:
        if isinstance(out, TemporalModelOutput):
            return out
        if isinstance(out, dict):
            return TemporalModelOutput(out)
        return TemporalModelOutput({"predictions": out})

    def save_pretrained(self, save_directory: str, **_: Any) -> None:
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        if self.config is not None and hasattr(self.config, "to_dict"):
            import json

            with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as handle:
                json.dump(self.config.to_dict(), handle, indent=2)

    @classmethod
    def from_pretrained(
        cls, load_directory: str, config: Optional[Any] = None, **kwargs: Any
    ) -> "BaseTemporalModel":
        model = cls(config=config, **kwargs)
        state = torch.load(os.path.join(load_directory, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(state, strict=False)
        return model


class AutoregressiveGenerationMixin:
    """Adds a generic autoregressive ``generate`` loop to compatible models."""

    supports_generation: bool = True

    def init_generation_state(self, inputs: torch.Tensor, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError

    def generation_step(
        self, step_input: torch.Tensor, state: Dict[str, Any], **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError

    def select_next_token(
        self,
        step_outputs: torch.Tensor,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **_: Any,
    ) -> torch.Tensor:
        logits = step_outputs.squeeze(1)
        if do_sample:
            if temperature != 1.0:
                logits = logits / temperature
            if top_k is not None and top_k > 0:
                top_k = min(top_k, logits.size(-1))
                vals, idx = torch.topk(logits, top_k)
                mask = torch.full_like(logits, float("-inf"))
                mask.scatter_(1, idx, vals)
                logits = mask
            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        return torch.argmax(logits, dim=-1, keepdim=True)

    @torch.no_grad()
    def generate(
        self,
        inputs: torch.Tensor,
        *,
        max_length: int = 32,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not getattr(self, "supports_generation", False):
            raise RuntimeError("This model does not support generation.")

        generated = inputs
        state = self.init_generation_state(inputs, **kwargs)

        for _ in range(max_length):
            step_input = generated[:, -1:, ...]
            step_out, state = self.generation_step(step_input, state, **kwargs)
            next_token = self.select_next_token(
                step_out,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                **kwargs,
            )
            if next_token.dim() == 2:
                next_token = next_token.unsqueeze(-1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated


__all__ = [
    "AutoregressiveGenerationMixin",
    "BaseTemporalModel",
    "TemporalModelOutput",
]
