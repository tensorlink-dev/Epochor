"""Public-facing contract that miners must implement for validator-driven training."""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn


class MinerSubmissionProtocol:
    """Interface validators expect miner submissions to implement.

    Validators drive the training loop and call back into the miner to
    construct the model, build an optimizer, and execute a single training
    step. Submissions must be deterministic under the provided configuration
    and seed so validators can reproduce results.
    """

    def build_model(self, cfg: Dict[str, Any]) -> nn.Module:
        """Construct and return the model to train under validator supervision.

        Expectations:
        - The implementation must be deterministic given ``cfg`` (and any
          externally provided seed) so the validator can reproduce results.
        - The returned model must accept context inputs shaped like the first
          ``cfg['context_length']`` timesteps of ``batch["x"]`` from the
          validator-provided dataloaders.
        - The forward pass must produce outputs whose shape matches the final
          ``cfg['prediction_length']`` timesteps of ``batch["x"]`` (the implied
          target segment).
        """

        raise NotImplementedError

    def build_optimizer(self, model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
        """Create and return the optimizer to use during training.

        Expectations:
        - The optimizer must operate on the parameters of ``model``.
        - Any hyperparameters should be derived from ``cfg`` to remain
          deterministic and reproducible.
        """

        raise NotImplementedError

    def train_step(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        step_idx: int,
        cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute one validator-provided batch and return training metrics.

        Expectations:
        - The validator provides ``batch['x']`` containing a concatenated
          sequence of length ``context_length + prediction_length`` on the
          chosen device. Submissions must reshape/split this sequence to derive
          their targets (e.g., first context_length timesteps as input and the
          remaining prediction_length timesteps as targets).
        - Implementations must perform a full training update: set the model to
          training mode, run forward + loss + backward, and step the optimizer.
        - The returned mapping **must** include ``"loss"`` as a scalar float
          value. Additional metrics are allowed but ignored for ranking.
        - Submissions may perform multiple internal gradient steps per call,
          but this is discouraged; validators may impose wall-clock limits in
          the future.
        """

        raise NotImplementedError


__all__ = ["MinerSubmissionProtocol"]
