"""Utilities for executing miner submissions inside a sandboxed container."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence

import torch

_LOGGER = logging.getLogger(__name__)


class SandboxError(RuntimeError):
    """Base exception for sandbox execution failures."""


class SandboxRuntimeNotFound(SandboxError):
    """Raised when the configured container runtime cannot be located."""


class SandboxTimeoutError(SandboxError):
    """Raised when the sandboxed execution exceeds the allotted time."""


class SandboxExecutionError(SandboxError):
    """Raised when the container process exits with a non-zero status."""

    def __init__(
        self,
        message: str,
        *,
        returncode: int,
        stdout: Sequence[str],
        stderr: Sequence[str],
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stdout = list(stdout)
        self.stderr = list(stderr)


class SandboxMissingOutputError(SandboxError):
    """Raised when the sandbox execution completes without producing output."""


class SandboxInvalidOutputError(SandboxError):
    """Raised when the sandbox output cannot be parsed safely."""


@dataclass
class SandboxResult:
    """Outcome of a sandbox run."""

    summary: Mapping[str, Any]
    stdout: Sequence[str]
    stderr: Sequence[str]
    artifacts_dir: Path


_DEFAULT_RUNTIME = "docker"
_DEFAULT_IMAGE = "epochor-sandbox:latest"
_CONTAINER_STAGING = Path("/sandbox")
_CONTAINER_VALIDATOR = Path("/validator")
_CONTAINER_SUBMISSION = Path("/submission")
_CONTAINER_OUTPUT = Path("/sandbox_out")
_STAGING_CONFIG = "cfg.json"
_STAGING_TRAIN = "train_batches.pt"
_STAGING_VAL = "val_batches.pt"
_STAGING_EVAL_SAMPLES = "eval_samples.pt"
_STAGING_EVAL_TASKS = "eval_tasks.pt"


def run_submission_in_sandbox(
    submission_dir: os.PathLike[str] | str,
    train_batches: Sequence[Mapping[str, torch.Tensor]],
    val_batches: Sequence[Mapping[str, torch.Tensor]],
    evaluation_config: Mapping[str, Any],
    *,
    output_path: os.PathLike[str] | str,
    artifacts_dir: os.PathLike[str] | str,
    evaluation_samples: Sequence[Sequence[Mapping[str, torch.Tensor]]],
    evaluation_tasks: Sequence[Any],
    evaluation_seed: Optional[int] = None,
    runtime: str = _DEFAULT_RUNTIME,
    image: str = _DEFAULT_IMAGE,
    timeout: Optional[float] = None,
    gpus: Optional[str] = None,
    cpus: Optional[str] = None,
    memory: Optional[str] = None,
    extra_env: Optional[Mapping[str, str]] = None,
    additional_runtime_args: Optional[Sequence[str]] = None,
    network_disabled: bool = True,
    read_only_root: bool = True,
    pids_limit: Optional[int] = 256,
    ulimit_nofile: Optional[int] = 1024,
    no_new_privileges: bool = True,
    drop_all_caps: bool = True,
    seccomp_profile: Optional[str] = None,
) -> SandboxResult:
    """Execute a miner submission within a sandboxed container.

    Parameters
    ----------
    submission_dir:
        Directory or file path containing the miner submission entry point.
    train_batches / val_batches:
        Materialized batches that the validator will provide to the submission.
        All tensors are moved to CPU memory before serialization.
    evaluation_config:
        JSON-serializable configuration consumed by the sandbox entry script.
        Must include a ``"training_cfg"`` mapping describing the validator run.
    output_path:
        Host path where the sandbox is expected to materialize the summary.
    artifacts_dir:
        Directory on the host where the sandbox will emit checkpoints and metadata.
    evaluation_samples / evaluation_tasks:
        Materialized evaluation payloads passed through to the sandbox for scoring.
    evaluation_seed:
        Seed forwarded to the sandbox to keep deterministic evaluation behaviour.
    runtime:
        Container runtime executable (defaults to ``docker``).
    image:
        Container image reference that contains the miner runtime environment.
    timeout:
        Optional timeout (in seconds) for the container execution.
    gpus / cpus / memory:
        Optional resource limits forwarded to the runtime (when supported).
    extra_env:
        Additional environment variables exposed to the sandbox.
    additional_runtime_args:
        Extra CLI flags appended to the runtime invocation.
    network_disabled / read_only_root / pids_limit / ulimit_nofile / no_new_privileges /
    drop_all_caps / seccomp_profile:
        Hardening options applied to the container runtime invocation.
    """

    submission_path = Path(submission_dir).resolve()
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission path does not exist: {submission_path}")

    validator_root = Path(__file__).resolve().parents[1]
    output_file = Path(output_path).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    artifacts_path = Path(artifacts_dir).resolve()
    artifacts_path.mkdir(parents=True, exist_ok=True)

    _validate_evaluation_config(evaluation_config)

    with tempfile.TemporaryDirectory(prefix="epochor-sandbox-") as tmpdir:
        staging_dir = Path(tmpdir)
        _LOGGER.debug("Staging sandbox inputs in %s", staging_dir)
        cfg_path = staging_dir / _STAGING_CONFIG
        train_path = staging_dir / _STAGING_TRAIN
        val_path = staging_dir / _STAGING_VAL
        samples_path = staging_dir / _STAGING_EVAL_SAMPLES
        tasks_path = staging_dir / _STAGING_EVAL_TASKS

        config_payload = dict(evaluation_config)
        if evaluation_seed is not None:
            config_payload = dict(config_payload)
            config_payload["evaluation_seed"] = evaluation_seed
        _write_json(cfg_path, config_payload)
        _serialize_batches(train_batches, train_path)
        _serialize_batches(val_batches, val_path)
        torch.save(evaluation_samples, samples_path)
        torch.save(list(evaluation_tasks), tasks_path)

        runtime_cmd = _build_runtime_command(
            runtime=runtime,
            image=image,
            staging_dir=staging_dir,
            validator_root=validator_root,
            submission_path=submission_path,
            output_path=output_file,
            artifacts_path=artifacts_path,
            evaluation_samples_name=_STAGING_EVAL_SAMPLES,
            evaluation_tasks_name=_STAGING_EVAL_TASKS,
            gpus=gpus,
            cpus=cpus,
            memory=memory,
            additional_runtime_args=additional_runtime_args,
            extra_env=extra_env,
            network_disabled=network_disabled,
            read_only_root=read_only_root,
            pids_limit=pids_limit,
            ulimit_nofile=ulimit_nofile,
            no_new_privileges=no_new_privileges,
            drop_all_caps=drop_all_caps,
            seccomp_profile=seccomp_profile,
        )

        _LOGGER.info("Executing sandbox command: %s", " ".join(runtime_cmd))
        try:
            completed = subprocess.run(
                runtime_cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError as exc:  # pragma: no cover - depends on host env
            raise SandboxRuntimeNotFound(f"Unable to locate container runtime '{runtime}'") from exc
        except subprocess.TimeoutExpired as exc:
            raise SandboxTimeoutError(
                f"Sandbox execution exceeded timeout of {timeout} seconds"
            ) from exc

    stdout_lines = _normalize_output(completed.stdout)
    stderr_lines = _normalize_output(completed.stderr)

    if completed.returncode != 0:
        raise SandboxExecutionError(
            f"Sandbox process exited with status {completed.returncode}",
            returncode=completed.returncode,
            stdout=stdout_lines,
            stderr=stderr_lines,
        )

    if not output_file.exists():
        raise SandboxMissingOutputError(
            f"Sandbox completed successfully but no output was produced at {output_file}"
        )

    summary = _load_summary(output_file)

    if not artifacts_path.exists():
        raise SandboxMissingOutputError(
            f"Sandbox completed successfully but no artefacts were produced at {artifacts_path}"
        )

    return SandboxResult(
        summary=summary,
        stdout=stdout_lines,
        stderr=stderr_lines,
        artifacts_dir=artifacts_path,
    )


def _validate_evaluation_config(config: Mapping[str, Any]) -> None:
    if "training_cfg" not in config:
        raise ValueError("evaluation_config must contain a 'training_cfg' mapping")
    if not isinstance(config["training_cfg"], MutableMapping):
        raise TypeError("'training_cfg' must be a mapping")


def _serialize_batches(
    batches: Sequence[Mapping[str, torch.Tensor]],
    destination: Path,
) -> None:
    prepared: list[dict[str, torch.Tensor]] = []
    for batch_idx, batch in enumerate(batches):
        if not isinstance(batch, Mapping):
            raise TypeError(f"Batch #{batch_idx} must be a mapping, received {type(batch)!r}")
        prepared_batch: dict[str, torch.Tensor] = {}
        for key, tensor in batch.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"Batch #{batch_idx} entry '{key}' is not a tensor (received {type(tensor)!r})"
                )
            prepared_batch[str(key)] = tensor.detach().cpu()
        prepared.append(prepared_batch)
    torch.save(prepared, destination)


def _write_json(destination: Path, payload: Mapping[str, Any]) -> None:
    try:
        with destination.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except TypeError as exc:
        raise TypeError("evaluation_config must be JSON serializable") from exc


def _build_runtime_command(
    *,
    runtime: str,
    image: str,
    staging_dir: Path,
    validator_root: Path,
    submission_path: Path,
    output_path: Path,
    artifacts_path: Path,
    evaluation_samples_name: str,
    evaluation_tasks_name: str,
    gpus: Optional[str],
    cpus: Optional[str],
    memory: Optional[str],
    additional_runtime_args: Optional[Sequence[str]],
    extra_env: Optional[Mapping[str, str]],
    network_disabled: bool,
    read_only_root: bool,
    pids_limit: Optional[int],
    ulimit_nofile: Optional[int],
    no_new_privileges: bool,
    drop_all_caps: bool,
    seccomp_profile: Optional[str],
) -> list[str]:
    cmd: list[str] = [runtime, "run", "--rm"]

    if artifacts_path.parent != output_path.parent:
        raise ValueError("artifacts_dir must share the same parent directory as output_path")

    if network_disabled:
        cmd.extend(["--network", "none"])
    if read_only_root:
        cmd.append("--read-only")
    if pids_limit is not None:
        cmd.extend(["--pids-limit", str(pids_limit)])
    if ulimit_nofile is not None:
        cmd.extend(["--ulimit", f"nofile={ulimit_nofile}:{ulimit_nofile}"])
    if no_new_privileges:
        cmd.extend(["--security-opt", "no-new-privileges"])
    if drop_all_caps:
        cmd.extend(["--cap-drop", "ALL"])
    if seccomp_profile:
        cmd.extend(["--security-opt", f"seccomp={seccomp_profile}"])

    if gpus:
        cmd.extend(["--gpus", gpus])
    if cpus:
        cmd.extend(["--cpus", cpus])
    if memory:
        cmd.extend(["--memory", memory])

    if additional_runtime_args:
        cmd.extend(list(additional_runtime_args))

    cmd.extend(
        [
            "-v",
            f"{staging_dir}:{_CONTAINER_STAGING}:rw",
            "-v",
            f"{validator_root}:{_CONTAINER_VALIDATOR}:ro",
            "-v",
            f"{submission_path}:{_CONTAINER_SUBMISSION}:ro",
            "-v",
            f"{output_path.parent}:{_CONTAINER_OUTPUT}:rw",
            "-w",
            str(_CONTAINER_VALIDATOR),
        ]
    )

    env_vars = dict(extra_env or {})

    env_vars.setdefault("PYTHONPATH", str(_CONTAINER_VALIDATOR))

    for key, value in env_vars.items():
        cmd.extend(["-e", f"{key}={value}"])

    output_name = output_path.name

    cmd.append(image)
    cmd.extend(
        [
            "python",
            "-m",
            "epochor.training.sandbox_entry",
            "--staging",
            str(_CONTAINER_STAGING),
            "--submission",
            str(_CONTAINER_SUBMISSION),
            "--output",
            str(_CONTAINER_OUTPUT / output_name),
            "--artifacts",
            str(_CONTAINER_OUTPUT / artifacts_path.name),
            "--samples-name",
            evaluation_samples_name,
            "--tasks-name",
            evaluation_tasks_name,
        ]
    )

    return cmd


def _normalize_output(stream: Optional[str]) -> list[str]:
    if not stream:
        return []
    return [line.rstrip() for line in stream.splitlines()]


def _load_summary(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            summary = json.load(fh)
    except json.JSONDecodeError as exc:
        raise SandboxInvalidOutputError(
            f"Sandbox output at {path} is not valid JSON"
        ) from exc

    if not isinstance(summary, Mapping):
        raise SandboxInvalidOutputError(
            "Sandbox output must be a JSON object mapping metric names to values"
        )

    return summary


__all__ = [
    "SandboxError",
    "SandboxExecutionError",
    "SandboxMissingOutputError",
    "SandboxInvalidOutputError",
    "SandboxResult",
    "SandboxRuntimeNotFound",
    "SandboxTimeoutError",
    "run_submission_in_sandbox",
]
