"""Validator-owned dataloaders and evaluation helpers for miner submissions."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Sequence, Tuple

import torch

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from epochor.training.validator_contract import MinerSubmissionProtocol

from epochor.validation.validation import score_time_series_model

Batch = Mapping[str, torch.Tensor]
DEFAULT_QUANTILES = [0.1 * i for i in range(1, 10)]


def split_context_and_target(sequence: torch.Tensor, cfg: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a concatenated sequence into context and target segments.

    Expects ``sequence`` to have shape ``(batch, total_length, ...)`` where
    ``total_length`` is ``context_length + prediction_length``. The context is
    returned as the first ``context_length`` timesteps and the target as the
    following ``prediction_length`` timesteps.
    """

    if sequence.dim() < 2:
        raise ValueError("Expected sequence tensor with shape (batch, time, ...)")

    try:
        context_length = int(cfg["context_length"])
        prediction_length = int(cfg["prediction_length"])
    except KeyError as exc:  # pragma: no cover - configuration validation
        raise KeyError("cfg must include 'context_length' and 'prediction_length'") from exc

    if context_length <= 0 or prediction_length <= 0:
        raise ValueError("context_length and prediction_length must be positive")

    required = context_length + prediction_length
    if sequence.shape[1] < required:
        raise ValueError(
            f"Sequence length {sequence.shape[1]} is shorter than required {required}"
        )

    context = sequence[:, :context_length]
    target = sequence[:, context_length : context_length + prediction_length]
    return context, target


def _resolve_quantiles(cfg: Dict[str, Any]) -> Sequence[float]:
    """Return quantile levels, defaulting to nine evenly spaced values."""

    quantiles = cfg.get("quantiles")
    if quantiles is None:
        quantiles = DEFAULT_QUANTILES
    if not isinstance(quantiles, (list, tuple)):
        raise TypeError("quantiles must be a list or tuple of floats")
    resolved = [float(q) for q in quantiles]
    if len(resolved) != 9:
        raise ValueError(f"quantiles must include exactly 9 entries (received {len(resolved)})")
    for q in resolved:
        if not 0 < q < 1:
            raise ValueError("quantiles must satisfy 0 < q < 1")
    return resolved


def _ensure_batches(batches: Sequence[Mapping[str, torch.Tensor]]) -> Sequence[Mapping[str, torch.Tensor]]:
    """Validate that batches are mappings of tensors."""

    for idx, batch in enumerate(batches):
        if not isinstance(batch, Mapping):
            raise TypeError(f"Batch #{idx} must be a mapping")
        for key, value in batch.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"Batch #{idx} entry '{key}' must be a torch.Tensor (received {type(value)!r})"
                )
    return batches


def _materialize_batches(batches: Sequence[Mapping[str, torch.Tensor]]) -> Iterable[Mapping[str, torch.Tensor]]:
    """Clone and yield batches to avoid in-place mutations by miner code."""

    validated = _ensure_batches(batches)
    for batch in validated:
        yield {key: tensor.clone() for key, tensor in batch.items()}


def _load_batches_from_path(path: Path) -> Sequence[Mapping[str, torch.Tensor]]:
    """Load serialized batches from ``path``."""

    data = torch.load(path)
    if not isinstance(data, Sequence):
        raise TypeError(f"Expected a sequence of batches at {path}")
    return _ensure_batches(data)


def make_train_loader(cfg: Dict[str, Any]) -> Iterable[Batch]:
    """Factory for validator-owned training batches."""

    train_path = cfg.get("train_batches_path")
    if train_path is None:
        raise ValueError("cfg must include 'train_batches_path'")
    batches = _load_batches_from_path(Path(train_path))
    return _materialize_batches(batches)


def make_val_loader(cfg: Dict[str, Any]) -> Iterable[Batch]:
    """Factory for validator-owned validation batches."""

    val_path = cfg.get("val_batches_path")
    if val_path is None:
        raise ValueError("cfg must include 'val_batches_path'")
    batches = _load_batches_from_path(Path(val_path))
    return _materialize_batches(batches)


def evaluate_fn(
    submission: "MinerSubmissionProtocol",  # quoted to avoid import cycle
    model: torch.nn.Module,
    val_loader: Iterable[Batch],
    device: torch.device,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Validator-owned evaluation function for miner submissions."""

    model.eval()
    metrics: Dict[str, Any] = {}
    total_loss = 0.0
    count = 0
    quantiles = _resolve_quantiles(cfg)
    quantile_count = len(quantiles)
    prediction_length = int(cfg["prediction_length"])
    with torch.no_grad():
        for batch in val_loader:
            batch_on_device = {key: tensor.to(device) for key, tensor in batch.items()}
            processed = submission.process_data(batch_on_device, cfg)
            if not isinstance(processed, Mapping):
                raise TypeError("process_data must return a mapping during evaluation")
            inputs = processed.get("inputs")
            targets = processed.get("targets")
            if inputs is None or targets is None:
                context, target = split_context_and_target(batch_on_device["x"], cfg)
                inputs = context if inputs is None else inputs
                targets = target if targets is None else targets
            if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
                raise TypeError("process_data inputs/targets must be tensors during evaluation")
            if targets.shape[1] != prediction_length:
                raise ValueError(
                    f"targets must span prediction_length={prediction_length} timesteps (got {targets.shape[1]})"
                )
            if targets.shape[-1] != quantile_count:
                raise ValueError(
                    f"targets last dimension must match quantile count {quantile_count} (got {targets.shape[-1]})"
                )
            try:
                preds = submission.forecast(
                    model,
                    inputs,
                    cfg,
                    prediction_length=prediction_length,
                    quantiles=quantiles,
                )
            except (NotImplementedError, AttributeError):
                if hasattr(model, "forecast"):
                    preds = model.forecast(
                        inputs=inputs,
                        prediction_length=prediction_length,
                        quantiles=quantiles,
                    )
                else:
                    preds = model(inputs)
            if preds.shape != targets.shape:
                raise ValueError(
                    f"Validation forward shape {tuple(preds.shape)} does not match target {tuple(targets.shape)}"
                )
            loss = torch.nn.functional.mse_loss(preds, targets)
            total_loss += float(loss.item())
            count += 1
    metrics["val_loss"] = total_loss / max(count, 1)

    eval_samples_path = cfg.get("eval_samples_path")
    eval_tasks_path = cfg.get("eval_tasks_path")
    if eval_samples_path is not None and eval_tasks_path is not None:
        samples = torch.load(Path(eval_samples_path))
        eval_tasks = torch.load(Path(eval_tasks_path))
        score, score_details = score_time_series_model(
            model,
            samples,
            eval_tasks,
            str(device),
            int(cfg.get("evaluation_seed", 0)),
        )
        metrics.update({"evaluation_score": score, "score_details": score_details})

    return metrics


__all__ = ["make_train_loader", "make_val_loader", "evaluate_fn", "split_context_and_target"]
