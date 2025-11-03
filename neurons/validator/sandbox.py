"""Helpers for executing miner submissions inside an isolated sandbox.

This module currently provides a minimal shim that mimics a sandboxed
execution environment. It is designed so that the validator can treat the
training loop as if it was executed in a separate runtime (e.g. Docker).
As tighter isolation is introduced, the API exposed here should remain
stable, limiting changes required in the evaluation service.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from epochor.training import load_miner_module, run_training
from epochor.validation.validation import score_time_series_model


@dataclass
class SandboxRuntimeConfig:
    """Runtime options that describe how the sandbox should execute."""

    image: str
    timeout_seconds: int
    max_memory_bytes: Optional[int] = None
    max_cpus: Optional[float] = None
    max_gpus: Optional[float] = None


@dataclass
class SandboxExecutionResult:
    """Result returned from :func:`run_submission_in_sandbox`."""

    status: str
    summary_json: Optional[str] = None
    error: Optional[str] = None
    returncode: Optional[int] = None

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.summary_json is not None


def run_submission_in_sandbox(
    snapshot_dir: str,
    *,
    competition_id: int,
    seed: int,
    train_batches: List[Any],
    samples: List[Any],
    eval_tasks: List[Any],
    preferred_device: str,
    runtime: SandboxRuntimeConfig,
) -> SandboxExecutionResult:
    """Execute a miner submission as if it were inside a sandbox.

    Args:
        snapshot_dir: Filesystem path containing the miner submission.
        competition_id: Identifier of the competition for configuration.
        seed: RNG seed used for deterministic behaviour.
        train_batches: Flattened list of training batches.
        samples: Raw samples used to assemble evaluation batches.
        eval_tasks: Tasks that drive validation scoring.
        preferred_device: Device hint supplied by the validator operator.
        runtime: Runtime settings describing the sandbox environment.

    Returns:
        A :class:`SandboxExecutionResult` that either contains a serialized
        :class:`~epochor.training.validator_runner.TrainingSummary` (on
        success) or metadata about the failure.
    """

    submission_path = _find_submission_file(snapshot_dir)
    if submission_path is None:
        return SandboxExecutionResult(
            status="missing_submission",
            error="miner_submission.py not found",
            returncode=127,
        )

    try:
        submission = load_miner_module(submission_path)
    except Exception:
        return SandboxExecutionResult(
            status="load_error",
            error=traceback.format_exc(),
            returncode=126,
        )

    cfg = {
        "competition_id": int(competition_id),
        "seed": seed,
        "max_steps": len(train_batches),
        "max_epochs": 1,
    }

    start_time = time.monotonic()
    try:
        summary = run_training(
            submission,
            cfg,
            train_loader_factory=_make_loader_factory(train_batches),
            val_loader_factory=_make_loader_factory(train_batches),
            evaluate_fn=_make_evaluate_fn(samples, eval_tasks, seed),
            preferred_device=preferred_device,
        )
    except Exception:
        return SandboxExecutionResult(
            status="runtime_error",
            error=traceback.format_exc(),
            returncode=1,
        )

    duration = time.monotonic() - start_time
    if runtime.timeout_seconds > 0 and duration > runtime.timeout_seconds:
        return SandboxExecutionResult(
            status="timeout",
            error=(
                f"Execution exceeded timeout of {runtime.timeout_seconds}s "
                f"(took {duration:.2f}s)"
            ),
            returncode=-1,
        )

    payload = {
        "train_metrics": _json_safe(summary.train_metrics),
        "val_metrics": _json_safe(summary.val_metrics),
        "num_steps": int(summary.num_steps),
        "device": summary.device,
    }

    return SandboxExecutionResult(
        status="ok",
        summary_json=json.dumps(payload),
        returncode=0,
    )


def _find_submission_file(snapshot_dir: str) -> Optional[str]:
    if not os.path.isdir(snapshot_dir):
        return None
    for root, _, files in os.walk(snapshot_dir):
        if "miner_submission.py" in files:
            return os.path.join(root, "miner_submission.py")
    return None


def _make_loader_factory(batches: List[Any]):
    def factory(cfg: Dict[str, Any]):  # noqa: D401 - small closure
        for batch in batches:
            yield batch

    return factory


def _make_evaluate_fn(samples: List[Any], eval_tasks: List[Any], seed: int):
    def evaluate(model, loader, device, cfg):  # noqa: D401 - matching protocol
        for _ in loader:
            pass
        score, score_details = score_time_series_model(
            model,
            samples,
            eval_tasks,
            str(device),
            seed,
        )
        return {"val_loss": score, "score_details": score_details}

    return evaluate


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive fallback
            return str(value)
    if hasattr(value, "tolist") and callable(getattr(value, "tolist")):
        try:
            return value.tolist()
        except Exception:  # pragma: no cover - defensive fallback
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "SandboxExecutionResult",
    "SandboxRuntimeConfig",
    "run_submission_in_sandbox",
]

