"""Adapter that routes validator sandbox requests through the container runner."""
from __future__ import annotations

import json
import os
import shutil
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from epochor.training.sandbox_runner import (
    SandboxError,
    SandboxExecutionError,
    SandboxInvalidOutputError,
    SandboxMissingOutputError,
    SandboxResult,
    SandboxRuntimeNotFound,
    SandboxTimeoutError,
    run_submission_in_sandbox as _runner_run_submission,
)


@dataclass
class SandboxRuntimeConfig:
    """Validator-provided runtime options for sandboxed execution."""

    image: str
    timeout_seconds: int
    max_memory_bytes: Optional[int] = None
    max_cpus: Optional[float] = None
    max_gpus: Optional[float] = None
    runtime: str = "docker"
    network_disabled: bool = True
    read_only_root: bool = True
    pids_limit: Optional[int] = 256
    ulimit_nofile: Optional[int] = 1024
    no_new_privileges: bool = True
    drop_all_caps: bool = True
    seccomp_profile: Optional[str] = None
    additional_runtime_args: Sequence[str] = field(default_factory=tuple)
    extra_env: Mapping[str, str] = field(default_factory=dict)


def run_submission_in_sandbox(
    snapshot_dir: str,
    *,
    competition_id: int,
    seed: int,
    train_batches: Sequence[Mapping[str, torch.Tensor]],
    samples: Sequence[Sequence[Mapping[str, torch.Tensor]]],
    eval_tasks: Sequence[Any],
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

    output_root = os.path.join(snapshot_dir, ".validator_sandbox_output")
    start_time = time.monotonic()
    try:
        try:
            summary = _runner_run_submission(
                submission=submission,
                cfg=cfg,
                train_batches=train_batches,
                samples=samples,
                eval_tasks=eval_tasks,
                seed=seed,
                preferred_device=preferred_device,
            )
        except Exception:
            shutil.rmtree(output_root, ignore_errors=True)
            raise
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

    gpus_arg = str(runtime.max_gpus) if runtime.max_gpus and runtime.max_gpus > 0 else None
    cpus_arg = str(runtime.max_cpus) if runtime.max_cpus and runtime.max_cpus > 0 else None
    memory_arg = (
        f"{int(runtime.max_memory_bytes)}b"
        if runtime.max_memory_bytes and runtime.max_memory_bytes > 0
        else None
    )
    timeout = runtime.timeout_seconds if runtime.timeout_seconds > 0 else None

    staged_samples = []
    for task_batches in samples:
        prepared_batches = []
        for batch in task_batches:
            prepared_batches.append({key: tensor.detach().cpu() for key, tensor in batch.items()})
        staged_samples.append(prepared_batches)

    result = _runner_run_submission(
        submission_dir=snapshot_dir,
        train_batches=list(train_batches),
        val_batches=list(train_batches),
        evaluation_config=evaluation_config,
        output_path=summary_path,
        artifacts_dir=artifacts_dir,
        evaluation_samples=staged_samples,
        evaluation_tasks=list(eval_tasks),
        evaluation_seed=seed,
        runtime=runtime.runtime,
        image=runtime.image or "epochor-sandbox:latest",
        timeout=timeout,
        gpus=gpus_arg,
        cpus=cpus_arg,
        memory=memory_arg,
        extra_env=dict(runtime.extra_env),
        additional_runtime_args=list(runtime.additional_runtime_args),
        network_disabled=runtime.network_disabled,
        read_only_root=runtime.read_only_root,
        pids_limit=runtime.pids_limit,
        ulimit_nofile=runtime.ulimit_nofile,
        no_new_privileges=runtime.no_new_privileges,
        drop_all_caps=runtime.drop_all_caps,
        seccomp_profile=runtime.seccomp_profile,
    )


def _runner_run_submission(
    *,
    submission: Any,
    cfg: Dict[str, Any],
    train_batches: List[Any],
    samples: List[Any],
    eval_tasks: List[Any],
    seed: int,
    preferred_device: str,
):
    return run_training(
        submission,
        cfg,
        train_loader_factory=_make_loader_factory(train_batches),
        val_loader_factory=_make_loader_factory(train_batches),
        evaluate_fn=_make_evaluate_fn(samples, eval_tasks, seed),
        preferred_device=preferred_device,
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
    "SandboxError",
    "SandboxExecutionError",
    "SandboxInvalidOutputError",
    "SandboxMissingOutputError",
    "SandboxResult",
    "SandboxRuntimeConfig",
    "SandboxRuntimeNotFound",
    "SandboxTimeoutError",
    "run_submission_in_sandbox",
]
