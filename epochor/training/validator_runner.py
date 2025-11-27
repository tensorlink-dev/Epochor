"""Validator-owned training loop utilities."""
from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
import uuid
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, MutableMapping, Optional

import torch
from torch import nn

from .validator_contract import MinerSubmissionProtocol
from epochor.utils.hf_io import save_as_safetensors

Batch = Mapping[str, torch.Tensor]
TrainLoaderFactory = Callable[[Dict[str, Any]], Iterable[Batch]]
ValLoaderFactory = Callable[[Dict[str, Any]], Iterable[Batch]]
EvaluateFn = Callable[
    [MinerSubmissionProtocol, nn.Module, Iterable[Batch], torch.device, Dict[str, Any]],
    Dict[str, Any],
]
BenchmarkLoaderFactory = Callable[[Dict[str, Any]], Iterable[Batch]]

MAX_TRAIN_STEPS = 2_000


@dataclass
class TrainingSummary:
    """Structured return value from :func:`run_training`."""

    train_metrics: Dict[str, Any]
    val_metrics: Dict[str, Any]
    num_steps: int
    device: str
    model: nn.Module
    submission_id: str
    run_id: str
    artifact_path: str | None = None
    artifact_uri: str | None = None


def _resolve_run_ids(
    cfg: Mapping[str, Any], submission_id: Optional[str] = None, run_id: Optional[str] = None
) -> tuple[str, str]:
    """Return stable submission and run identifiers.

    ``submission_id`` may be provided explicitly or via ``cfg['submission_id']``;
    otherwise, a new UUID4 string is generated. ``run_id`` is always unique per
    invocation unless explicitly provided or present in ``cfg``.
    """

    resolved_submission_id = str(
        submission_id
        or cfg.get("submission_id")
        or cfg.get("submission_uuid")
        or uuid.uuid4()
    )
    resolved_run_id = str(run_id or cfg.get("run_id") or uuid.uuid4())
    return resolved_submission_id, resolved_run_id


def _resolve_device(preferred: Optional[Any] = None) -> torch.device:
    """Return the requested device, defaulting to CUDA when available."""

    if preferred is not None:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_batch_to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move all tensors in ``batch`` to ``device``."""

    return {key: tensor.to(device) for key, tensor in batch.items()}


def _count_params(model: nn.Module) -> int:
    """Return the total number of parameters in ``model``."""

    return sum(p.numel() for p in model.parameters())


def _resolve_prediction_length(cfg: Mapping[str, Any]) -> int:
    """Return the required prediction length from configuration."""

    try:
        prediction_length = int(cfg["prediction_length"])
    except KeyError as exc:
        raise KeyError("cfg must include 'prediction_length'") from exc
    if prediction_length <= 0:
        raise ValueError("prediction_length must be positive")
    return prediction_length


def _resolve_quantiles(cfg: Mapping[str, Any]) -> list[float]:
    """Return quantiles list, defaulting to nine evenly spaced values."""

    quantiles = cfg.get("quantiles")
    if quantiles is None:
        quantiles = [0.1 * i for i in range(1, 10)]
    if not isinstance(quantiles, (list, tuple)):
        raise TypeError("quantiles must be a list or tuple of floats")
    resolved = [float(q) for q in quantiles]
    if len(resolved) != 9:
        raise ValueError(f"quantiles must include exactly 9 entries (received {len(resolved)})")
    for q in resolved:
        if not 0 < q < 1:
            raise ValueError("quantiles must satisfy 0 < q < 1")
    return resolved


def _artifact_filename(submission_id: str, run_id: str, cfg: Mapping[str, Any]) -> str:
    """Return the filename to use for a safetensors artifact."""

    explicit = cfg.get("artifact_filename") or cfg.get("artifact_name")
    if explicit:
        return str(explicit)
    return f"model_{submission_id}_{run_id}.safetensors"


def _upload_file_to_s3(
    local_path: Path,
    *,
    bucket: str,
    key: str,
    endpoint_url: str | None = None,
    region: str | None = None,
    access_key: str | None = None,
    secret_key: str | None = None,
) -> str:
    """Upload ``local_path`` to an S3/R2-compatible bucket and return the URI."""

    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("boto3 is required for S3 uploads; install it or omit S3 config") from exc

    client = boto3.client(
        "s3",
        region_name=region,
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    client.upload_file(str(local_path), bucket, key)
    return f"s3://{bucket}/{key}"


def _persist_model_artifact(
    model: nn.Module, submission_id: str, run_id: str, cfg: Mapping[str, Any]
) -> tuple[str, str | None]:
    """Save the trained ``model`` as safetensors and optionally upload to S3."""

    artifact_root = Path(cfg.get("artifact_dir") or cfg.get("artifacts_dir") or "./artifacts")
    artifact_root.mkdir(parents=True, exist_ok=True)
    filename = _artifact_filename(submission_id, run_id, cfg)
    local_path = artifact_root / filename
    saved_path = save_as_safetensors(model, local_path)

    bucket = cfg.get("s3_bucket")
    if not bucket:
        return str(saved_path), None

    key_prefix = cfg.get("s3_prefix", "validator-runs")
    key_override = cfg.get("s3_key")
    key = key_override or f"{key_prefix}/{submission_id}/{run_id}/{saved_path.name}"
    endpoint_url = cfg.get("s3_endpoint") or cfg.get("s3_endpoint_url")
    region = cfg.get("s3_region")
    access_key = cfg.get("s3_access_key") or os.getenv("S3_ACCESS_KEY") or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = cfg.get("s3_secret_key") or os.getenv("S3_SECRET_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")

    remote_uri = _upload_file_to_s3(
        saved_path,
        bucket=str(bucket),
        key=str(key),
        endpoint_url=endpoint_url,
        region=region,
        access_key=access_key,
        secret_key=secret_key,
    )
    return str(saved_path), remote_uri


def _prepare_inputs_and_targets(
    submission: MinerSubmissionProtocol, batch: Mapping[str, torch.Tensor], cfg: Dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the submission to derive model inputs and targets from a batch."""

    processed = submission.process_data(dict(batch), cfg)
    if not isinstance(processed, Mapping):
        raise TypeError("process_data must return a mapping")
    if "inputs" not in processed:
        raise KeyError("process_data result must include an 'inputs' entry")
    if "targets" not in processed:
        raise KeyError("process_data result must include a 'targets' entry")

    inputs = processed["inputs"]
    targets = processed["targets"]
    if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError("process_data 'inputs' and 'targets' must be tensors")

    prediction_length = _resolve_prediction_length(cfg)
    quantiles = _resolve_quantiles(cfg)
    if targets.shape[1] != prediction_length:
        raise ValueError(
            f"targets must span prediction_length={prediction_length} timesteps (got {targets.shape[1]})"
        )
    if targets.shape[-1] != len(quantiles):
        raise ValueError(
            "targets last dimension must match quantile count "
            f"(expected {len(quantiles)}, got {targets.shape[-1]})"
        )

    return inputs, targets


def _call_forecast(
    submission: MinerSubmissionProtocol,
    model: nn.Module,
    inputs: torch.Tensor,
    cfg: Mapping[str, Any],
    *,
    prediction_length: int,
    quantiles: list[float],
) -> torch.Tensor:
    """Invoke the submission's forecast method (with fallbacks) and return outputs."""

    try:
        return submission.forecast(
            model,
            inputs,
            cfg,
            prediction_length=prediction_length,
            quantiles=quantiles,
        )
    except (NotImplementedError, AttributeError):
        if hasattr(model, "forecast"):
            return model.forecast(
                inputs=inputs,
                prediction_length=prediction_length,
                quantiles=quantiles,
            )
        return model(inputs)


def _validate_model_contract(
    submission: MinerSubmissionProtocol,
    cfg: Dict[str, Any],
    train_loader_factory: TrainLoaderFactory,
    val_loader_factory: ValLoaderFactory,
    device: torch.device,
) -> nn.Module:
    """Build and validate a submission model against validator batch shapes.

    This constructs the model, moves it to ``device``, derives inputs/targets via
    ``submission.process_data``, performs a dummy forward pass, and ensures the
    predicted output matches the derived target shape exactly.
    """

    train_iter = _iterate_batches(train_loader_factory, cfg)
    train_batch = next(train_iter)
    if "x" not in train_batch:
        raise KeyError("Training batches must contain an 'x' entry")

    try:
        batch_on_device = _move_batch_to_device(train_batch, device)
        expected_context, expected_target = _prepare_inputs_and_targets(submission, batch_on_device, cfg)
    except Exception:
        val_iter = _iterate_batches(val_loader_factory, cfg)
        val_batch = next(val_iter)
        if "x" not in val_batch:
            raise KeyError("Validation batches must contain an 'x' entry for contract checks")
        batch_on_device = _move_batch_to_device(val_batch, device)
        expected_context, expected_target = _prepare_inputs_and_targets(submission, batch_on_device, cfg)

    prediction_length = _resolve_prediction_length(cfg)
    quantiles = _resolve_quantiles(cfg)

    model = submission.build_model(cfg).to(device)
    model.eval()

    with torch.no_grad():
        inputs = expected_context.to(device)
        preds = _call_forecast(
            submission,
            model,
            inputs,
            cfg,
            prediction_length=prediction_length,
            quantiles=quantiles,
        )

    expected_shape = expected_target.shape
    if not hasattr(preds, "shape"):
        raise ValueError("Model forward pass must return a tensor-like object with a shape")
    if preds.shape[1] != prediction_length:
        raise ValueError(
            f"Model output must span prediction_length={prediction_length} timesteps (got {preds.shape[1]})"
        )
    if preds.shape[-1] != len(quantiles):
        raise ValueError(
            "Model output quantile dimension must match requested quantiles "
            f"(expected {len(quantiles)}, got {preds.shape[-1]})"
        )
    if preds.shape != expected_shape:
        raise ValueError(
            f"Model output shape {tuple(preds.shape)} does not match expected {tuple(expected_shape)}"
        )

    return model


def load_miner_module(submission_path: str) -> MinerSubmissionProtocol:
    """Dynamically load a miner submission from a python file.

    The module must expose ``get_submission()`` which returns an instance of
    :class:`MinerSubmissionProtocol`.
    """

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
    submission_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> TrainingSummary:
    """Execute the validator-owned training loop for a miner submission."""

    resolved_submission_id, resolved_run_id = _resolve_run_ids(cfg, submission_id, run_id)
    device = _resolve_device(preferred_device)
    max_steps_cfg = cfg.get("max_steps")
    hard_cap = MAX_TRAIN_STEPS if max_train_steps is None else min(MAX_TRAIN_STEPS, int(max_train_steps))
    if max_steps_cfg is not None:
        hard_cap = min(hard_cap, int(max_steps_cfg))
    if hard_cap <= 0:
        raise ValueError("Training must run for at least one step")

    model = _validate_model_contract(submission, cfg, train_loader_factory, val_loader_factory, device)

    max_params = int(cfg.get("max_params", 10_000_000))
    if max_params <= 0:
        raise ValueError("max_params must be positive")
    num_params = _count_params(model)
    if num_params > max_params:
        raise ValueError(
            f"Model has {num_params} parameters which exceeds allowed maximum of {max_params}"
        )

    optimizer = submission.build_optimizer(model, cfg)
    model.train()

    max_epochs = cfg.get("max_epochs")
    if max_epochs is not None:
        max_epochs = int(max_epochs)
        if max_epochs <= 0:
            raise ValueError("max_epochs must be positive when provided")
    epochs_to_run = max_epochs or 1

    train_metrics: Optional[Dict[str, Any]] = None
    num_steps = 0
    artifact_path: Optional[str] = None
    artifact_uri: Optional[str] = None

    for epoch_idx in range(epochs_to_run):
        for batch in _iterate_batches(train_loader_factory, cfg):
            batch_on_device = _move_batch_to_device(batch, device)
            before_mem = _capture_allocated(device)
            # Hook for future wall-clock enforcement could be placed here.
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
            submission,
            model,
            _iterate_batches(val_loader_factory, cfg),
            device,
            cfg,
        )
        if not isinstance(val_metrics, MutableMapping):
            raise TypeError("evaluate_fn must return a mapping of metrics")

    artifact_path, artifact_uri = _persist_model_artifact(
        model, resolved_submission_id, resolved_run_id, cfg
    )

    return TrainingSummary(
        train_metrics=dict(train_metrics),
        val_metrics=dict(val_metrics),
        num_steps=num_steps,
        device=str(device),
        model=model,
        submission_id=resolved_submission_id,
        run_id=resolved_run_id,
        artifact_path=artifact_path,
        artifact_uri=artifact_uri,
    )


def benchmark_submission(
    submission: MinerSubmissionProtocol,
    model: nn.Module,
    cfg: Dict[str, Any],
    *,
    benchmark_loader_factory: BenchmarkLoaderFactory,
    preferred_device: Optional[Any] = None,
) -> list[Dict[str, torch.Tensor]]:
    """Run the trained model in eval mode on unseen data using ``forecast``.

    The benchmark loader is expected to yield batches with ``"x"`` entries whose
    lengths reflect the desired context and prediction lengths. The submission's
    ``process_data`` hook derives inputs/targets; ``forecast`` must return
    tensors matching the target shape and quantile dimension.
    """

    device = _resolve_device(preferred_device)
    prediction_length = _resolve_prediction_length(cfg)
    quantiles = _resolve_quantiles(cfg)
    results: list[Dict[str, torch.Tensor]] = []

    model.eval()
    with torch.no_grad():
        for batch in _iterate_batches(benchmark_loader_factory, cfg):
            batch_on_device = _move_batch_to_device(batch, device)
            inputs, targets = _prepare_inputs_and_targets(submission, batch_on_device, cfg)
            preds = _call_forecast(
                submission,
                model,
                inputs,
                cfg,
                prediction_length=prediction_length,
                quantiles=quantiles,
            )
            if preds.shape[1] != prediction_length:
                raise ValueError(
                    "Benchmark predictions must span the configured prediction length "
                    f"{prediction_length} (got {preds.shape[1]})"
                )
            if preds.shape[-1] != len(quantiles):
                raise ValueError(
                    "Benchmark predictions must include the configured quantile dimension "
                    f"{len(quantiles)} (got {preds.shape[-1]})"
                )
            if preds.shape != targets.shape:
                raise ValueError(
                    f"Benchmark output shape {tuple(preds.shape)} does not match target {tuple(targets.shape)}"
                )
            results.append({"preds": preds.cpu(), "targets": targets.cpu()})

    return results


def _capture_allocated(device: torch.device) -> Optional[int]:
    """Return current CUDA memory allocation for ``device`` if available."""

    if device.type != "cuda":
        return None
    try:  # pragma: no cover - depends on GPU availability
        torch.cuda.synchronize(device)
        return torch.cuda.memory_allocated(device)
    except RuntimeError:  # pragma: no cover - handle CUDA driver absence
        return None


def _iterate_batches(factory: Callable[[Dict[str, Any]], Iterable[Batch]], cfg: Dict[str, Any]) -> Iterator[Batch]:
    """Yield batches from ``factory(cfg)`` ensuring at least one batch exists."""

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
    "benchmark_submission",
]
