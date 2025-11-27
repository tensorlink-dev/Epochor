"""Validator-owned dataloaders and evaluation helpers for miner submissions."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch

from epochor.validation.validation import score_time_series_model

Batch = Mapping[str, torch.Tensor]


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
    with torch.no_grad():
        for batch in val_loader:
            batch_on_device = {key: tensor.to(device) for key, tensor in batch.items()}
            preds = model(batch_on_device["x"])
            loss = torch.nn.functional.mse_loss(preds, batch_on_device["y"])
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


__all__ = ["make_train_loader", "make_val_loader", "evaluate_fn"]
