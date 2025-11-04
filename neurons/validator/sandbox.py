"""Adapter that routes validator sandbox requests through the container runner."""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

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
) -> SandboxResult:
    """Execute a miner submission using the hardened sandbox runner."""

    output_root = Path(tempfile.mkdtemp(prefix="epochor-validator-sandbox-"))
    summary_path = output_root / "summary.json"
    artifacts_dir = output_root / "artifacts"

    evaluation_config: dict[str, Any] = {
        "training_cfg": {
            "competition_id": int(competition_id),
            "seed": int(seed),
            "max_steps": len(train_batches),
            "max_epochs": 1,
        },
        "preferred_device": preferred_device,
        "max_memory_bytes": runtime.max_memory_bytes,
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

    return result


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
