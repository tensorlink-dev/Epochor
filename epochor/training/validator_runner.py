"""Validator-owned training loop utilities."""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, MutableMapping, Optional

import torch
from torch import nn

from .validator_contract import MinerSubmissionProtocol

Batch = Mapping[str, torch.Tensor]
TrainLoaderFactory = Callable[[Dict[str, Any]], Iterable[Batch]]
ValLoaderFactory = Callable[[Dict[str, Any]], Iterable[Batch]]
EvaluateFn = Callable[[nn.Module, Iterable[Batch], torch.device, Dict[str, Any]], Dict[str, Any]]

MAX_TRAIN_STEPS = 2_000


@dataclass
class TrainingSummary:
    """Structured return value from :func:`run_training`."""

    train_metrics: Dict[str, Any]
    val_metrics: Dict[str, Any]
    num_steps: int
    device: str
    model: nn.Module


def _resolve_device(preferred: Optional[Any] = None) -> torch.device:
    if preferred is not None:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_batch_to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: tensor.to(device) for key, tensor in batch.items()}


def load_miner_module(submission_path: str) -> MinerSubmissionProtocol:
    """Dynamically load a miner submission from a python file."""

    spec = importlib.util.spec_from_file_location("miner_submission", submission_path)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib safeguard
        raise ImportError(f"Unable to create spec for submission: {submission_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[assignment]
    submission = _fetch_submission_from_module(module)
    if not isinstance(submission, MinerSubmissionProtocol):
        raise TypeError("Submission must implement MinerSubmissionProtocol")
    return submission


def _fetch_submission_from_module(module: ModuleType) -> MinerSubmissionProtocol:
    if not hasattr(module, "get_submission"):
        raise AttributeError("Submission module must define get_submission()")
    submission = module.get_submission()  # type: ignore[attr-defined]
    if submission is None:
        raise ValueError("get_submission() returned None")
    return submission


def run_training(
    submission: MinerSubmissionProtocol,
    cfg: Dict[str, Any],
    *,
    train_loader_factory: TrainLoaderFactory,
    val_loader_factory: ValLoaderFactory,
    evaluate_fn: EvaluateFn,
    max_train_steps: Optional[int] = None,
    preferred_device: Optional[Any] = None,
    grad_clip_norm: Optional[float] = None,
    max_memory_bytes: Optional[int] = None,
) -> TrainingSummary:
    """Execute the validator-owned training loop for a miner submission."""

    device = _resolve_device(preferred_device)
    max_steps_cfg = cfg.get("max_steps")
    hard_cap = MAX_TRAIN_STEPS if max_train_steps is None else min(MAX_TRAIN_STEPS, int(max_train_steps))
    if max_steps_cfg is not None:
        hard_cap = min(hard_cap, int(max_steps_cfg))
    if hard_cap <= 0:
        raise ValueError("Training must run for at least one step")

    model = submission.build_model(cfg).to(device)
    optimizer = submission.build_optimizer(model, cfg)

    max_epochs = cfg.get("max_epochs")
    if max_epochs is not None:
        max_epochs = int(max_epochs)
        if max_epochs <= 0:
            raise ValueError("max_epochs must be positive when provided")
    epochs_to_run = max_epochs or 1

    train_metrics: Optional[Dict[str, Any]] = None
    num_steps = 0

    for epoch_idx in range(epochs_to_run):
        for batch in _iterate_batches(train_loader_factory, cfg):
            batch_on_device = _move_batch_to_device(batch, device)
            before_mem = _capture_allocated(device)
            metrics = submission.train_step(model, batch_on_device, optimizer, num_steps, cfg)
            if not isinstance(metrics, MutableMapping):
                raise TypeError("train_step must return a mapping of metrics")
            if "loss" not in metrics:
                raise ValueError("train_step metrics must include a 'loss' entry")
            train_metrics = dict(metrics)
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            if max_memory_bytes is not None:
                after_mem = _capture_allocated(device)
                if after_mem is not None and before_mem is not None and (after_mem - before_mem) > max_memory_bytes:
                    raise RuntimeError("train_step exceeded allowed memory delta")
            num_steps += 1
            if num_steps >= hard_cap:
                break
        if num_steps >= hard_cap:
            break

    if train_metrics is None:
        raise RuntimeError("No training steps were executed")

    model.eval()
    with torch.no_grad():
        val_metrics = evaluate_fn(
            model,
            _iterate_batches(val_loader_factory, cfg),
            device,
            cfg,
        )
        if not isinstance(val_metrics, MutableMapping):
            raise TypeError("evaluate_fn must return a mapping of metrics")

    return TrainingSummary(
        train_metrics=dict(train_metrics),
        val_metrics=dict(val_metrics),
        num_steps=num_steps,
        device=str(device),
        model=model,
    )


def _capture_allocated(device: torch.device) -> Optional[int]:
    if device.type != "cuda":
        return None
    try:  # pragma: no cover - depends on GPU availability
        torch.cuda.synchronize(device)
        return torch.cuda.memory_allocated(device)
    except RuntimeError:  # pragma: no cover - handle CUDA driver absence
        return None


def _iterate_batches(factory: Callable[[Dict[str, Any]], Iterable[Batch]], cfg: Dict[str, Any]) -> Iterator[Batch]:
    iterable = factory(cfg)
    iterator = iter(iterable)
    try:
        first = next(iterator)
    except StopIteration:
        raise RuntimeError("Batch loader produced no data")
    yield first
    for batch in iterator:
        yield batch


__all__ = [
    "MAX_TRAIN_STEPS",
    "TrainingSummary",
    "load_miner_module",
    "run_training",
]
