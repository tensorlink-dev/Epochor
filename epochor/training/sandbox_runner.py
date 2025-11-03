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


_DEFAULT_RUNTIME = "docker"
_DEFAULT_IMAGE = "epochor-sandbox:latest"
_CONTAINER_STAGING = Path("/sandbox")
_CONTAINER_VALIDATOR = Path("/validator")
_CONTAINER_SUBMISSION = Path("/submission")
_CONTAINER_OUTPUT = Path("/sandbox_out")
_STAGING_CONFIG = "cfg.json"
_STAGING_TRAIN = "train_batches.pt"
_STAGING_VAL = "val_batches.pt"
_HF_ENV_VARS = (
    "HF_TOKEN",
    "HF_ACCESS_TOKEN",
    "HF_API_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)


def run_submission_in_sandbox(
    submission_dir: os.PathLike[str] | str,
    train_batches: Sequence[Mapping[str, torch.Tensor]],
    val_batches: Sequence[Mapping[str, torch.Tensor]],
    evaluation_config: Mapping[str, Any],
    output_path: os.PathLike[str] | str,
    *,
    runtime: str = _DEFAULT_RUNTIME,
    image: str = _DEFAULT_IMAGE,
    timeout: Optional[float] = None,
    gpus: Optional[str] = None,
    cpus: Optional[str] = None,
    memory: Optional[str] = None,
    extra_env: Optional[Mapping[str, str]] = None,
    additional_runtime_args: Optional[Sequence[str]] = None,
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
    """

    submission_path = Path(submission_dir).resolve()
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission path does not exist: {submission_path}")

    validator_root = Path(__file__).resolve().parents[1]
    output_file = Path(output_path).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    _validate_evaluation_config(evaluation_config)

    with tempfile.TemporaryDirectory(prefix="epochor-sandbox-") as tmpdir:
        staging_dir = Path(tmpdir)
        _LOGGER.debug("Staging sandbox inputs in %s", staging_dir)
        cfg_path = staging_dir / _STAGING_CONFIG
        train_path = staging_dir / _STAGING_TRAIN
        val_path = staging_dir / _STAGING_VAL

        _write_json(cfg_path, evaluation_config)
        _serialize_batches(train_batches, train_path)
        _serialize_batches(val_batches, val_path)

        runtime_cmd = _build_runtime_command(
            runtime=runtime,
            image=image,
            staging_dir=staging_dir,
            validator_root=validator_root,
            submission_path=submission_path,
            output_path=output_file,
            gpus=gpus,
            cpus=cpus,
            memory=memory,
            additional_runtime_args=additional_runtime_args,
            extra_env=extra_env,
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
    return SandboxResult(summary=summary, stdout=stdout_lines, stderr=stderr_lines)


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
    gpus: Optional[str],
    cpus: Optional[str],
    memory: Optional[str],
    additional_runtime_args: Optional[Sequence[str]],
    extra_env: Optional[Mapping[str, str]],
) -> list[str]:
    cmd: list[str] = [runtime, "run", "--rm"]

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
    for key in _HF_ENV_VARS:
        if key in os.environ and key not in env_vars:
            env_vars[key] = os.environ[key]

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
