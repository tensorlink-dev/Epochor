"""Validator-owned training entrypoint executed within a sandbox."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import socket
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Iterable, List, Optional, Tuple

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


def _ensure_float32_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype="float32")
    if array.size == 0:
        raise ValueError(f"Value for '{name}' produced an empty array")
    return array


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


def _finite_float_sequence(values: Iterable[float]) -> np.ndarray:
    sequence = np.asarray(list(values), dtype="float32").reshape(-1)
    if not np.isfinite(sequence).all():
        sequence = sequence[np.isfinite(sequence)]
    return sequence


def _sliding_windows_1d(
    series: np.ndarray, context: int, horizon: int, stride: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    series = np.asarray(series, dtype="float32").reshape(-1)
    required = context + horizon
    if series.size < required:
        return None, None
    window_count = 1 + (series.size - required) // max(stride, 1)
    step = series.strides[0]
    view = np.lib.stride_tricks.as_strided(
        series,
        shape=(window_count, required),
        strides=(stride * step, step),
    )
    return view[:, :context], view[:, context:]


def _stable_row_identifier(row: Dict[str, Any], *, target_key: str) -> str:
    candidate_keys = ("id", "uid", "series_id", "start", "timestamp")
    for key in candidate_keys:
        value = row.get(key)
        if value is not None:
            return f"{key}:{value}"
    target = row.get(target_key)
    if target is not None:
        array = np.asarray(target, dtype="float32").reshape(-1)
        head = np.nan_to_num(array[:64], nan=0.0, posinf=1e38, neginf=-1e38)
        digest = hashlib.blake2b(head.tobytes(), digest_size=12).hexdigest()
        return f"len={array.size}|head64={digest}"
    return json.dumps(row, sort_keys=True, default=str, separators=(",", ":"))


def _unit_interval_hash(*parts: str) -> float:
    digest = hashlib.blake2b("|".join(parts).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def _deterministic_row_inclusion(
    row_key: str, probability: float, seed: int, epoch: int
) -> bool:
    if probability >= 1.0:
        return True
    if probability <= 0.0:
        return False
    return _unit_interval_hash("row", row_key, str(seed), str(epoch)) < probability


def _deterministic_window_indices(
    total: int,
    sample_size: int,
    seed: int,
    epoch: int,
    row_key: str,
    *,
    with_replacement: bool,
) -> List[int]:
    if sample_size >= total and not with_replacement:
        return list(range(total))
    rng_seed = int(_unit_interval_hash("window", row_key, str(seed), str(epoch)) * (2**31 - 1))
    rng = random.Random(rng_seed)
    if with_replacement:
        return [rng.randrange(total) for _ in range(sample_size)]
    return rng.sample(range(total), sample_size)


def _select_active_shards(total: int, active: int, seed: int, epoch: int) -> List[int]:
    count = max(1, min(active, total))
    rng_seed = int(_unit_interval_hash("shards", str(seed), str(epoch)) * (2**31 - 1))
    rng = random.Random(rng_seed)
    indices = list(range(total))
    rng.shuffle(indices)
    return sorted(indices[:count])


@dataclass
class _HFWindowStreamConfig:
    dataset_repo: str
    split: str
    dataset_config: Optional[str]
    dataset_kwargs: Dict[str, Any]
    streaming: bool
    max_batches: int
    budget_batch_size: int
    total_shards: int
    active_shards: int
    streams_per_shard: int
    seed: int
    sample_fraction: float
    windows_per_series: Optional[int]
    windows_with_replacement: bool
    context_length: int
    forecast_horizon: int
    stride: int
    min_series_length: Optional[int]
    enable_worker_sharding: bool
    include_metadata: bool
    row_identity_keys: Optional[List[str]]
    target_key: str


class _HFWindowedTimeseriesStream(torch.utils.data.IterableDataset):
    """Stream Hugging Face timeseries windows under a deterministic budget."""

    def __init__(self, settings: _HFWindowStreamConfig) -> None:
        super().__init__()
        self._settings = settings
        self._epoch = 0
        self._base_iterable: Optional[Any] = None
        self._refresh_base(seed_offset=0)

    def _refresh_base(self, seed_offset: int) -> None:
        try:
            from datasets import interleave_datasets, load_dataset  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "datasets package is required for Hugging Face streaming"
            ) from exc

        cfg = self._settings
        streams: List[Any] = []
        active = _select_active_shards(
            cfg.total_shards, cfg.active_shards, cfg.seed, seed_offset
        )
        for shard_index in active:
            for shard_replica in range(cfg.streams_per_shard):
                dataset = load_dataset(
                    cfg.dataset_repo,
                    name=cfg.dataset_config,
                    split=cfg.split,
                    streaming=cfg.streaming,
                    **cfg.dataset_kwargs,
                )
                dataset = dataset.shard(
                    num_shards=cfg.total_shards,
                    index=shard_index,
                    contiguous=True,
                )
                streams.append(dataset)
        if not streams:
            raise RuntimeError(
                "No Hugging Face shards resolved for streaming dataset"
            )
        self._base_iterable = interleave_datasets(streams, seed=cfg.seed + seed_offset)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        self._refresh_base(seed_offset=self._epoch)

    @property
    def windows_budget(self) -> int:
        cfg = self._settings
        return int(cfg.max_batches * cfg.budget_batch_size)

    def _maybe_shard_for_worker(self, dataset: Any) -> Any:
        if not self._settings.enable_worker_sharding:
            return dataset
        info = torch.utils.data.get_worker_info()
        if info and info.num_workers > 1:
            dataset = dataset.shard(
                num_shards=info.num_workers,
                index=info.id,
                contiguous=False,
            )
        return dataset

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if self._base_iterable is None:
            raise RuntimeError("Hugging Face stream was not initialised")

        cfg = self._settings
        epoch = self._epoch
        dataset = self._maybe_shard_for_worker(self._base_iterable)

        emitted = 0
        budget = self.windows_budget
        target_key = cfg.target_key

        for row in dataset:
            if emitted >= budget:
                break

            target = row.get(target_key)
            if target is None:
                continue

            identifier_keys = cfg.row_identity_keys or []
            row_key = _stable_row_identifier(row, target_key=target_key)
            if identifier_keys:
                for key in identifier_keys:
                    value = row.get(key)
                    if value is not None:
                        row_key = f"{key}:{value}"
                        break

            if not _deterministic_row_inclusion(
                row_key, cfg.sample_fraction, cfg.seed, epoch
            ):
                continue

            series = _finite_float_sequence(target)
            if cfg.min_series_length is not None and series.size < cfg.min_series_length:
                continue

            contexts, horizons = _sliding_windows_1d(
                series, cfg.context_length, cfg.forecast_horizon, cfg.stride
            )
            if contexts is None or horizons is None:
                continue

            window_count = contexts.shape[0]
            if window_count <= 0:
                continue

            if cfg.windows_per_series and cfg.windows_per_series > 0:
                sample_count = min(cfg.windows_per_series, window_count)
                indices = _deterministic_window_indices(
                    window_count,
                    sample_count,
                    cfg.seed,
                    epoch,
                    row_key,
                    with_replacement=cfg.windows_with_replacement,
                )
            else:
                indices = range(window_count)

            for idx in indices:
                if emitted >= budget:
                    break

                features = np.ascontiguousarray(contexts[idx])
                targets = np.ascontiguousarray(horizons[idx])
                item: Dict[str, Any] = {
                    "x": torch.from_numpy(features),
                    "y": torch.from_numpy(targets),
                }
                if cfg.include_metadata:
                    metadata = {key: value for key, value in row.items() if key != target_key}
                    item["meta"] = metadata
                yield item
                emitted += 1

class _HFStreamingIterableDataset(torch.utils.data.IterableDataset):
    """Iterable dataset that streams samples from the Hugging Face Hub."""

    def __init__(
        self,
        loader: Callable[[], Iterable[Dict[str, Any]]],
        *,
        input_key: str,
        target_key: str,
        streaming: bool,
    ) -> None:
        super().__init__()
        self._loader = loader
        self._input_key = input_key
        self._target_key = target_key
        self._streaming = streaming

    def _convert(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if self._input_key not in sample:
            raise KeyError(f"Missing '{self._input_key}' key in Hugging Face sample")
        if self._target_key not in sample:
            raise KeyError(f"Missing '{self._target_key}' key in Hugging Face sample")

        x = _ensure_float32_array(sample[self._input_key], name=self._input_key)
        y = _ensure_float32_array(sample[self._target_key], name=self._target_key)
        x = x.reshape(-1).astype("float32")
        y = y.reshape(-1).astype("float32")

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
        }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        def _iterate_once() -> Iterator[Dict[str, torch.Tensor]]:
            dataset_iterable = self._loader()
            for sample in dataset_iterable:
                yield self._convert(sample)

        if not self._streaming:
            yield from _iterate_once()
            return

        while True:
            yield from _iterate_once()


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


def _maybe_window_stream_config(
    cfg: Dict[str, Any],
    *,
    repo_name: str,
    split: str,
    streaming: bool,
    dataset_kwargs: Dict[str, Any],
) -> Optional[_HFWindowStreamConfig]:
    window_section = cfg.get("hf_window_stream")
    has_prefixed_keys = any(key.startswith("hf_window_") for key in cfg)
    if not isinstance(window_section, dict) and not has_prefixed_keys:
        return None

    settings: Dict[str, Any] = {}
    if isinstance(window_section, dict):
        settings.update(window_section)

    prefix = "hf_window_"
    for key, value in cfg.items():
        if key.startswith(prefix):
            settings[key[len(prefix) :]] = value

    dataset_config = settings.get("dataset_config") or cfg.get("hf_dataset_config")
    total_shards = int(settings.get("total_shards", settings.get("total_slices", 64)))
    active_shards = int(settings.get("active_shards", settings.get("active_slices", 2)))
    streams_per_shard = int(
        settings.get("streams_per_shard", settings.get("streams_per_slice", 2))
    )
    budget_batch_size = int(
        settings.get("budget_batch_size", settings.get("batch_size_for_budget", 32))
    )
    max_batches = int(settings.get("max_batches", 1000))
    sample_fraction = float(settings.get("sample_fraction", 1.0))
    windows_per_series_val = settings.get("windows_per_series")
    if windows_per_series_val is None:
        windows_per_series = None
    else:
        windows_per_series = int(windows_per_series_val)
    context_length = int(settings.get("context_length", settings.get("ctx", 256)))
    forecast_horizon = int(
        settings.get("forecast_horizon", settings.get("hor", 64))
    )
    stride = int(settings.get("stride", 1))
    min_series_length_val = settings.get("min_series_length", settings.get("min_series_len"))
    min_series_length = int(min_series_length_val) if min_series_length_val is not None else None
    seed = int(settings.get("seed", cfg.get("seed", 0)))
    enable_worker_sharding = _coerce_bool(
        settings.get("enable_worker_sharding", True), True
    )
    include_metadata = _coerce_bool(
        settings.get("include_metadata", settings.get("return_meta", False)), False
    )
    windows_with_replacement = _coerce_bool(
        settings.get("window_with_replacement", False), False
    )

    row_identity_keys_val = settings.get("row_identity_keys")
    if isinstance(row_identity_keys_val, (list, tuple)):
        row_identity_keys = [str(key) for key in row_identity_keys_val]
    elif isinstance(row_identity_keys_val, str):
        row_identity_keys = [row_identity_keys_val]
    else:
        row_identity_keys = None

    total_shards = max(1, total_shards)
    active_shards = max(1, min(active_shards, total_shards))
    streams_per_shard = max(1, streams_per_shard)
    budget_batch_size = max(1, budget_batch_size)
    max_batches = max(1, max_batches)
    sample_fraction = max(0.0, min(1.0, sample_fraction))
    if windows_per_series is not None:
        windows_per_series = max(1, windows_per_series)
    context_length = max(1, context_length)
    forecast_horizon = max(1, forecast_horizon)
    stride = max(1, stride)
    if min_series_length is not None:
        min_series_length = max(1, min_series_length)

    target_key = str(settings.get("target_key") or cfg.get("hf_target_key") or "target")

    return _HFWindowStreamConfig(
        dataset_repo=str(settings.get("dataset_repo") or repo_name),
        split=str(settings.get("split") or split),
        dataset_config=(
            str(dataset_config)
            if isinstance(dataset_config, str) and dataset_config
            else None
        ),
        dataset_kwargs=dict(dataset_kwargs),
        streaming=streaming,
        max_batches=max_batches,
        budget_batch_size=budget_batch_size,
        total_shards=total_shards,
        active_shards=active_shards,
        streams_per_shard=streams_per_shard,
        seed=seed,
        sample_fraction=sample_fraction,
        windows_per_series=windows_per_series,
        windows_with_replacement=windows_with_replacement,
        context_length=context_length,
        forecast_horizon=forecast_horizon,
        stride=stride,
        min_series_length=min_series_length,
        enable_worker_sharding=enable_worker_sharding,
        include_metadata=include_metadata,
        row_identity_keys=row_identity_keys,
        target_key=target_key,
    )


def _build_hf_streaming_dataset(cfg: Dict[str, Any]) -> torch.utils.data.IterableDataset:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "datasets package is required to stream Hugging Face datasets"
        ) from exc

    repo_name = cfg.get("hf_dataset_repo") or cfg.get("hf_dataset_name")
    if not isinstance(repo_name, str) or not repo_name:
        nested = cfg.get("hf_window_stream")
        if isinstance(nested, dict):
            candidate = nested.get("dataset_repo") or nested.get("dataset")
            if isinstance(candidate, str) and candidate:
                repo_name = candidate
    if not isinstance(repo_name, str) or not repo_name:
        raise ValueError("cfg must define 'hf_dataset_repo' or 'hf_dataset_name'")

    split = cfg.get("hf_dataset_split", "train")
    if not isinstance(split, str) or not split:
        nested = cfg.get("hf_window_stream")
        if isinstance(nested, dict):
            candidate_split = nested.get("split")
            if isinstance(candidate_split, str) and candidate_split:
                split = candidate_split
    if not isinstance(split, str) or not split:
        raise ValueError("cfg['hf_dataset_split'] must be a non-empty string")

    streaming = _coerce_bool(cfg.get("hf_dataset_streaming", True), True)

    dataset_kwargs: Dict[str, Any] = dict(cfg.get("hf_dataset_kwargs") or {})
    config_name = cfg.get("hf_dataset_config")
    if isinstance(config_name, str) and config_name:
        dataset_kwargs["name"] = config_name

    data_files = cfg.get("hf_dataset_data_files")
    if data_files:
        dataset_kwargs["data_files"] = data_files

    revision = cfg.get("hf_dataset_revision")
    if isinstance(revision, str) and revision:
        dataset_kwargs["revision"] = revision

    token_env = cfg.get("hf_dataset_token_env")
    token_value = None
    if isinstance(token_env, str) and token_env:
        token_value = os.getenv(token_env)
        if token_value is None:
            raise RuntimeError(
                f"Environment variable '{token_env}' required for Hugging Face dataset access"
            )

    explicit_token = cfg.get("hf_dataset_token")
    if isinstance(explicit_token, str) and explicit_token:
        token_value = explicit_token

    if token_value:
        dataset_kwargs["token"] = token_value

    window_settings = _maybe_window_stream_config(
        cfg,
        repo_name=repo_name,
        split=split,
        streaming=streaming,
        dataset_kwargs=dataset_kwargs,
    )
    if window_settings is not None:
        return _HFWindowedTimeseriesStream(window_settings)

    def _loader() -> Iterable[Dict[str, Any]]:
        return load_dataset(repo_name, split=split, streaming=streaming, **dataset_kwargs)

    input_key = cfg.get("hf_input_key", "x")
    target_key = cfg.get("hf_target_key", "y")
    if not isinstance(input_key, str) or not input_key:
        raise ValueError("cfg['hf_input_key'] must be a non-empty string")
    if not isinstance(target_key, str) or not target_key:
        raise ValueError("cfg['hf_target_key'] must be a non-empty string")

    return _HFStreamingIterableDataset(
        _loader,
        input_key=input_key,
        target_key=target_key,
        streaming=streaming,
    )


def _build_dataset(cfg: Dict[str, Any], submission_dir: str) -> torch.utils.data.Dataset:
    hf_repo = cfg.get("hf_dataset_repo") or cfg.get("hf_dataset_name")
    if isinstance(hf_repo, str) and hf_repo:
        return _build_hf_streaming_dataset(cfg)

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
    if isinstance(dataset, torch.utils.data.IterableDataset):

        def _iterator() -> Iterator[Dict[str, torch.Tensor]]:
            iterator = iter(dataset)
            attempts = 0
            epoch = 0
            if hasattr(dataset, "set_epoch"):
                try:
                    dataset.set_epoch(epoch)  # type: ignore[attr-defined]
                except Exception:
                    pass
            while True:
                batch_x = []
                batch_y = []
                batch_meta: List[Any] = []
                while len(batch_x) < batch_size:
                    try:
                        sample = next(iterator)
                    except StopIteration:
                        attempts += 1
                        if attempts > 2:
                            raise RuntimeError("Iterable dataset did not yield enough samples to form a batch")
                        epoch += 1
                        if hasattr(dataset, "set_epoch"):
                            try:
                                dataset.set_epoch(epoch)  # type: ignore[attr-defined]
                            except Exception:
                                pass
                        iterator = iter(dataset)
                        continue
                    attempts = 0
                    x_tensor = torch.as_tensor(sample["x"], dtype=torch.float32)
                    y_tensor = torch.as_tensor(sample["y"], dtype=torch.float32)
                    batch_x.append(x_tensor)
                    batch_y.append(y_tensor)
                    if "meta" in sample:
                        batch_meta.append(sample["meta"])
                batch = {
                    "x": torch.stack(batch_x),
                    "y": torch.stack(batch_y),
                }
                if batch_meta:
                    batch["meta"] = batch_meta
                yield batch

        return _iterator()

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    def _iterator_map() -> Iterator[Dict[str, torch.Tensor]]:
        while True:
            for batch in loader:
                yield batch

    return _iterator_map()


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
