"""Public-facing contract that miners must implement for validator-driven training."""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn


class MinerSubmissionProtocol:
    """Interface validators expect miner submissions to implement.

    Validators drive the training loop and call back into the miner to
    construct the model, build an optimizer, and execute a single training
    step. Submissions should be deterministic under the provided configuration
    and seed so validators can reproduce results.
    """

    def build_model(self, cfg: Dict[str, Any]) -> nn.Module:
        """Return the model to train under validator supervision."""

        raise NotImplementedError

    def build_optimizer(self, model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
        """Create and return the optimizer to use during training."""

        raise NotImplementedError

    def train_step(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        step_idx: int,
        cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute one validator-provided batch and return training metrics."""

        raise NotImplementedError


__all__ = ["MinerSubmissionProtocol"]
