"""Validator-owned training entrypoint executed within a sandbox."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import math
import os
import random
import shutil
import socket
import time
from typing import Any, Dict, Iterator, Tuple

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from torch import nn

from epochor.utils.hf_io import HF_WRITE_TOKEN_ENV, push_artifacts_to_hf, write_meta

from .miner_protocol import MinerSubmissionProtocol


def _set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class _NetworkBlocker:
    """Context manager that disables outgoing network connections."""

    def __init__(self) -> None:
        self._orig_create_connection = socket.create_connection

    def __enter__(self) -> None:  # pragma: no cover - trivial
        def _blocked(*_args: Any, **_kwargs: Any) -> socket.socket:
            raise RuntimeError("Network disabled")

        socket.create_connection = _blocked  # type: ignore[assignment]

    def __exit__(self, _exc_type, _exc, _tb) -> None:  # pragma: no cover - trivial
        socket.create_connection = self._orig_create_connection


def _load_submission(sub_dir: str, module: str = "miner") -> MinerSubmissionProtocol:
    path = os.path.join(sub_dir, f"{module}.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing submission module: {path}")

    spec = importlib.util.spec_from_file_location(module, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module} from {path}")

    module_obj = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module_obj)  # type: ignore[arg-type]

    if hasattr(module_obj, "Submission"):
        submission = module_obj.Submission()  # type: ignore[operator]
    elif hasattr(module_obj, "get_submission"):
        submission = module_obj.get_submission()  # type: ignore[operator]
    else:
        raise ValueError("Submission must define Submission or get_submission()")

    if not isinstance(submission, MinerSubmissionProtocol):
        raise TypeError("Submission does not implement MinerSubmissionProtocol")

    return submission


class _ToyDataset(torch.utils.data.Dataset):
    def __init__(self, n: int = 4096, d_in: int = 16, noise: float = 0.1, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.X = rng.normal(size=(n, d_in)).astype("float32")
        w = rng.normal(size=(d_in, 1)).astype("float32")
        y = self.X @ w + 0.3
        self.y = (y + noise * rng.normal(size=y.shape)).astype("float32")

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.X[index]),
            "y": torch.from_numpy(self.y[index]),
        }


class _ArrayDataset(torch.utils.data.Dataset):
    """Dataset backed by in-memory arrays loaded from disk."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.ndim != 2:
            raise ValueError("Expected x to have shape (n_samples, n_features)")
        if y.ndim == 1:
            y = y[:, None]
        if y.ndim != 2:
            raise ValueError("Expected y to have shape (n_samples, 1)")
        if x.shape[0] != y.shape[0]:
            raise ValueError("Mismatched number of samples between x and y")
        self.X = np.asarray(x, dtype="float32")
        self.y = np.asarray(y, dtype="float32")

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.X[index]),
            "y": torch.from_numpy(self.y[index]),
        }


def _load_dataset_from_file(path: str) -> torch.utils.data.Dataset:
    if not os.path.exists(path):
        raise FileNotFoundError(f"dataset file not found: {path}")

    if path.endswith((".npz", ".npy")):
        data = np.load(path)
        if isinstance(data, np.ndarray):
            raise ValueError("NumPy file must contain named arrays 'x' and 'y'")
        x = data["x"]
        y = data["y"]
    elif path.endswith((".pt", ".pth")):
        loaded = torch.load(path, map_location="cpu")
        if isinstance(loaded, dict):
            x = loaded.get("x")
            y = loaded.get("y")
        elif isinstance(loaded, (list, tuple)) and len(loaded) >= 2:
            x, y = loaded[0], loaded[1]
        else:
            raise ValueError("Torch file must contain tensors 'x' and 'y'")
        x = x.cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        y = y.cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
    else:
        raise ValueError("Unsupported dataset extension. Use .npz or .pt")

    return _ArrayDataset(np.asarray(x), np.asarray(y))


def _build_dataset(cfg: Dict[str, Any], submission_dir: str) -> torch.utils.data.Dataset:
    dataset_path = cfg.get("dataset_path")
    if isinstance(dataset_path, str) and dataset_path:
        resolved = dataset_path if os.path.isabs(dataset_path) else os.path.join(submission_dir, dataset_path)
        return _load_dataset_from_file(resolved)

    batch_seed = int(cfg.get("seed", 0))
    input_dim = int(cfg.get("input_dim", 16))
    dataset_size = int(cfg.get("dataset_size", 4096))
    return _ToyDataset(n=dataset_size, d_in=input_dim, seed=batch_seed)


def _get_dataloader(cfg: Dict[str, Any], dataset: torch.utils.data.Dataset) -> Iterator[Dict[str, torch.Tensor]]:
    batch_size = int(cfg.get("train_batch_size", 64))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    def _iterator() -> Iterator[Dict[str, torch.Tensor]]:
        while True:
            for batch in loader:
                yield batch

    return _iterator()


def _resume_weights_if_any(model: nn.Module, cfg: Dict[str, Any]) -> None:
    checkpoint = cfg.get("resume_ckpt_path")
    if not isinstance(checkpoint, str) or not checkpoint:
        return
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"resume checkpoint not found: {checkpoint}")
    state = load_safetensors(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise ValueError(f"Missing keys on resume: {missing}")
    if unexpected:  # pragma: no cover - defensive; unexpected keys acceptable for forward-compat
        for key in unexpected:
            if key not in state:
                raise ValueError(f"Unexpected key on resume: {key}")


def _save_and_hash(model: nn.Module, out_dir: str, tag: str) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = os.path.join(out_dir, f"tmp_{tag}.safetensors")
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    save_safetensors(state, tmp_path, metadata={"tag": tag})

    hasher = hashlib.sha256()
    with open(tmp_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    artifact_id = f"sha256:{hasher.hexdigest()}"

    final_path = os.path.join(out_dir, f"model_{tag}.safetensors")
    os.replace(tmp_path, final_path)
    return final_path, artifact_id


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.numel() == 1:
            return tensor.item()
        return tensor.tolist()
    return str(value)


def _sanitize_repo_segment(segment: str) -> str:
    allowed = [ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in segment.strip()]
    sanitized = "".join(allowed).strip("-_.")
    return sanitized or "validator"


def _resolve_repo_id(cfg: Dict[str, Any], lease: Dict[str, Any]) -> str:
    explicit = cfg.get("hf_repo_id") or lease.get("hf_repo_id")
    if isinstance(explicit, str) and explicit:
        return explicit

    miner_hotkey = lease.get("miner_hotkey") or cfg.get("miner_hotkey")
    if not isinstance(miner_hotkey, str) or not miner_hotkey:
        raise ValueError("Missing miner_hotkey for Hugging Face upload")

    namespace = cfg.get("hf_repo_namespace") or lease.get("hf_repo_namespace") or miner_hotkey
    submission_id = lease.get("submission_id") or cfg.get("submission_id")
    model_id = lease.get("model_id") or cfg.get("model_id")
    base_name = cfg.get("hf_repo_name") or lease.get("hf_repo_name") or "validator-training"
    if submission_id:
        base_name = f"{base_name}-{submission_id}"
    elif model_id:
        base_name = f"{base_name}-{model_id}"

    namespace_seg = _sanitize_repo_segment(str(namespace))
    name_seg = _sanitize_repo_segment(str(base_name))
    return f"{namespace_seg}/{name_seg}"


def _build_metadata(
    cfg: Dict[str, Any],
    lease: Dict[str, Any],
    artifact_id: str,
    checkpoint_path: str,
    steps: int,
    best_loss: float,
    last_metrics: Dict[str, Any],
    elapsed_seconds: float,
) -> Dict[str, Any]:
    metadata = {
        "artifact_id": artifact_id,
        "checkpoint_path": checkpoint_path,
        "elapsed_seconds": elapsed_seconds,
        "steps": steps,
        "best_loss": best_loss,
        "lease": {key: _json_safe(value) for key, value in lease.items()},
        "cfg": {key: _json_safe(value) for key, value in cfg.items()},
        "last_metrics": _json_safe(last_metrics),
    }
    return metadata


def _prepare_upload_payload(
    artifacts_dir: str,
    checkpoint_path: str,
    metadata_path: str,
    artifact_id: str,
) -> str:
    slug = artifact_id.split(":", 1)[-1]
    upload_dir = os.path.join(artifacts_dir, f"upload_{slug}")
    os.makedirs(upload_dir, exist_ok=True)

    ckpt_dest = os.path.join(upload_dir, os.path.basename(checkpoint_path))
    meta_dest = os.path.join(upload_dir, os.path.basename(metadata_path))
    if checkpoint_path != ckpt_dest:
        shutil.copy2(checkpoint_path, ckpt_dest)
    if metadata_path != meta_dest:
        shutil.copy2(metadata_path, meta_dest)
    return upload_dir


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return default


def _elapsed_seconds(start: float) -> float:
    return max(0.0, time.time() - start)


def _ensure_metrics_dict(metrics: Any) -> Dict[str, Any]:
    if not isinstance(metrics, dict):
        raise TypeError("train_step must return a dict of metrics")
    return metrics


def _validate_loss(metrics: Dict[str, Any]) -> float:
    loss_val = float(metrics.get("loss", float("inf")))
    if not math.isfinite(loss_val):
        raise ValueError("train_step must return a finite 'loss'")
    return loss_val


def _prepare_device(cfg: Dict[str, Any]) -> torch.device:
    dev = cfg.get("device", "cuda")
    if dev == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _loop_training(
    submission: MinerSubmissionProtocol,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    device: torch.device,
    start_time: float,
    dataloader: Iterator[Dict[str, torch.Tensor]],
) -> Tuple[int, float, Dict[str, Any]]:
    max_steps = int(cfg.get("max_steps", 999_999))
    max_seconds = float(cfg.get("max_seconds", 3600.0))
    best_loss = float("inf")
    last_metrics: Dict[str, Any] = {}
    step = 0

    while step < max_steps and _elapsed_seconds(start_time) < max_seconds:
        batch = next(dataloader)

        batch = {key: tensor.to(device) for key, tensor in batch.items()}
        metrics = submission.train_step(model, batch, optimizer, step, cfg)
        metrics = _ensure_metrics_dict(metrics)
        loss_val = _validate_loss(metrics)
        if loss_val < best_loss:
            best_loss = loss_val
        last_metrics = metrics
        step += 1
        if _elapsed_seconds(start_time) >= max_seconds:
            break

    return step, best_loss, last_metrics


def _load_submission_and_prepare(cfg: Dict[str, Any], sub_dir: str, device: torch.device) -> Tuple[MinerSubmissionProtocol, nn.Module, torch.optim.Optimizer]:
    submission = _load_submission(sub_dir)
    model = submission.build_model(cfg).to(device)
    _resume_weights_if_any(model, cfg)
    optimizer = submission.build_optimizer(model, cfg)
    return submission, model, optimizer


def _resolve_tag(cfg: Dict[str, Any], lease: Dict[str, Any]) -> str:
    tag = cfg.get("output_tag")
    if isinstance(tag, str) and tag:
        return tag
    lease_round = lease.get("round")
    if lease_round is not None:
        return f"round{lease_round}"
    return "checkpoint"


async def run(inputs: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(inputs.get("cfg") or {})
    lease = dict(inputs.get("lease") or {})

    seed = int(cfg.get("seed", 0))
    _set_deterministic(seed)

    disable_network = _coerce_bool(cfg.get("disable_network", True), True)
    submission_dir = os.environ.get("SUBMISSION_DIR", "/submission")
    artifacts_dir = os.environ.get("ARTIFACTS_DIR", "/artifacts")

    device = _prepare_device(cfg)
    start_time = time.time()

    context_manager = _NetworkBlocker() if disable_network else contextlib.nullcontext()
    with context_manager:
        submission, model, optimizer = _load_submission_and_prepare(cfg, submission_dir, device)
        dataset = _build_dataset(cfg, submission_dir)
        dataloader = _get_dataloader(cfg, dataset)
        steps, best_loss, last_metrics = _loop_training(submission, model, optimizer, cfg, device, start_time, dataloader)

    elapsed = _elapsed_seconds(start_time)
    tag = _resolve_tag(cfg, lease)
    checkpoint_path, artifact_id = _save_and_hash(model, artifacts_dir, tag)

    metadata = _build_metadata(cfg, lease, artifact_id, checkpoint_path, steps, best_loss, last_metrics, elapsed)
    metadata_filename = f"{artifact_id.replace(':', '_')}_metadata.json"
    metadata_path = write_meta(metadata, path=os.path.join(artifacts_dir, metadata_filename))

    try:
        repo_id = _resolve_repo_id(cfg, lease)
        upload_dir = _prepare_upload_payload(artifacts_dir, checkpoint_path, str(metadata_path), artifact_id)
        commit_hash = push_artifacts_to_hf(
            repo_id,
            local_dir=upload_dir,
            token_env=HF_WRITE_TOKEN_ENV,
            commit_message=f"validator upload {artifact_id}",
        )
    except Exception as exc:  # pragma: no cover - error path exercised in integration
        return {
            "ok": False,
            "error": f"huggingface upload failed: {exc}",
            "steps": steps,
            "elapsed_seconds": elapsed,
            "best_loss": best_loss,
            "last_metrics": last_metrics,
            "artifact_id": artifact_id,
            "checkpoint_path": checkpoint_path,
            "metadata_path": str(metadata_path),
        }

    return {
        "ok": True,
        "steps": steps,
        "elapsed_seconds": elapsed,
        "best_loss": best_loss,
        "last_metrics": last_metrics,
        "artifact_id": artifact_id,
        "checkpoint_path": checkpoint_path,
        "metadata_path": str(metadata_path),
        "hf_repo_id": repo_id,
        "hf_commit_hash": commit_hash,
    }


__all__ = ["run"]
