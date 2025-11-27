"""Public-facing contract that miners must implement for validator-driven training."""
from __future__ import annotations

from typing import Any, Dict, Sequence

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

    def process_data(self, batch: Dict[str, torch.Tensor], cfg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Transform validator-provided batches into model-ready tensors.

        Expectations:
        - Receives the validator-provided ``batch`` containing at least ``batch["x"]``
          with the concatenated context + prediction sequence.
        - Returns a mapping containing at minimum ``"inputs"`` (model inputs) and
          ``"targets"`` (expected outputs) as tensors. Targets **must** include a
          quantile axis matching the requested quantiles (default 9). Additional
          derived tensors may be included to support custom training logic.
        - Implementations may reshape, normalize, or otherwise transform the data
          but must remain deterministic under ``cfg`` and any externally provided
          seed so that the validator can reproduce behavior.
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

    def forecast(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        cfg: Dict[str, Any],
        *,
        prediction_length: int | None = None,
        quantiles: Sequence[float] | None = None,
    ) -> torch.Tensor:
        """Run inference to predict ``prediction_length`` steps and quantiles.

        Expectations:
        - ``inputs`` should match the processed model inputs (typically the
          context segment returned by :meth:`process_data`).
        - ``prediction_length`` must match ``cfg['prediction_length']`` unless
          otherwise specified; outputs **must** span exactly this many future
          steps.
        - ``quantiles`` defaults to nine quantiles if not provided. The returned
          tensor must include a quantile axis whose length matches the provided
          quantiles (e.g., ``len(quantiles) == 9``).
        - The returned tensor must be deterministic under ``cfg`` and any
          externally provided seed.
        - Implementations may apply custom decoding or sampling strategies but
          must avoid leaking evaluation data and should keep runtime modest.

        By default, this calls ``model.forecast`` when available to preserve
        quantile-aware behavior, otherwise it falls back to a direct forward
        pass (without quantile handling) for backwards compatibility.
        """

        if hasattr(model, "forecast"):
            return model.forecast(
                inputs=inputs,
                prediction_length=prediction_length,
                quantiles=quantiles,
            )
        return model(inputs)


__all__ = ["MinerSubmissionProtocol"]
