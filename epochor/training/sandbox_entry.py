"""Entry point executed inside the sandbox container."""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import torch

from .validator_runner import TrainingSummary, load_miner_module, run_training
from epochor.utils.hf_io import save_as_safetensors
from epochor.validation.validation import score_time_series_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validator training inside sandbox")
    parser.add_argument("--staging", required=True, help="Path to staged input directory")
    parser.add_argument("--submission", required=True, help="Path to the submission directory")
    parser.add_argument("--output", required=True, help="Destination for the training summary")
    parser.add_argument("--artifacts", required=True, help="Directory for emitted artefacts")
    parser.add_argument("--config-name", default="cfg.json", help="Name of the staged config file")
    parser.add_argument(
        "--train-name", default="train_batches.pt", help="Name of the staged training batches file"
    )
    parser.add_argument(
        "--val-name", default="val_batches.pt", help="Name of the staged validation batches file"
    )
    parser.add_argument(
        "--samples-name", default="eval_samples.pt", help="Name of the staged evaluation samples file"
    )
    parser.add_argument(
        "--tasks-name", default="eval_tasks.pt", help="Name of the staged evaluation tasks file"
    )
    args = parser.parse_args()

    staging_dir = Path(args.staging)
    cfg_path = staging_dir / args.config_name
    train_path = staging_dir / args.train_name
    val_path = staging_dir / args.val_name
    eval_samples_path = staging_dir / args.samples_name
    eval_tasks_path = staging_dir / args.tasks_name

    config = _load_json(cfg_path)
    train_batches = torch.load(train_path)
    val_batches = torch.load(val_path)
    eval_samples = torch.load(eval_samples_path)
    eval_tasks = torch.load(eval_tasks_path)

    submission_file = _resolve_submission(Path(args.submission))
    submission = load_miner_module(str(submission_file))

    evaluation_seed = config.get("evaluation_seed") or config.get("training_cfg", {}).get("seed")

    evaluate_fn = _make_evaluate_fn(eval_samples, eval_tasks, int(evaluation_seed or 0))
    train_factory = _make_loader_factory(train_batches)
    val_factory = _make_loader_factory(val_batches)

    summary = run_training(
        submission=submission,
        cfg=dict(config.get("training_cfg", {})),
        train_loader_factory=train_factory,
        val_loader_factory=val_factory,
        evaluate_fn=evaluate_fn,
        max_train_steps=config.get("max_train_steps"),
        preferred_device=config.get("preferred_device"),
        grad_clip_norm=config.get("grad_clip_norm"),
        max_memory_bytes=config.get("max_memory_bytes"),
    )

    output_path = Path(args.output)
    artifacts_path = Path(args.artifacts)

    summary_payload = _write_summary(output_path, summary)
    _materialize_artifacts(artifacts_path, summary, summary_payload, config)


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_submission(path: Path) -> Path:
    if path.is_file():
        return path
    candidate = path / "submission.py"
    if candidate.exists():
        return candidate
    raise FileNotFoundError("Unable to locate submission entry point within sandbox")


def _make_loader_factory(
    batches: Sequence[Mapping[str, torch.Tensor]]
) -> Callable[[Dict[str, Any]], Iterable[Mapping[str, torch.Tensor]]]:
    materialized = [
        {key: tensor.detach().clone() for key, tensor in batch.items()}
        for batch in _ensure_batches(batches)
    ]

    def _factory(_: Dict[str, Any]) -> Iterable[Mapping[str, torch.Tensor]]:
        for batch in materialized:
            yield {key: tensor.clone() for key, tensor in batch.items()}

    return _factory


def _ensure_batches(
    batches: Sequence[Mapping[str, torch.Tensor]]
) -> Sequence[Mapping[str, torch.Tensor]]:
    for idx, batch in enumerate(batches):
        if not isinstance(batch, Mapping):
            raise TypeError(f"Batch #{idx} must be a mapping")
        for key, value in batch.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"Batch #{idx} entry '{key}' must be a torch.Tensor (received {type(value)!r})"
                )
    return batches


def _make_evaluate_fn(
    samples: Sequence[Sequence[Mapping[str, torch.Tensor]]],
    eval_tasks: Sequence[Any],
    seed: int,
) -> Callable[[Any, Any, Iterable[Mapping[str, torch.Tensor]], torch.device, Dict[str, Any]], Dict[str, Any]]:
    materialized_samples: list[list[Mapping[str, torch.Tensor]]] = []
    for task_batches in samples:
        copied_batches: list[Mapping[str, torch.Tensor]] = []
        for batch in task_batches:
            copied_batches.append({key: tensor.clone() for key, tensor in batch.items()})
        materialized_samples.append(copied_batches)

    def _evaluate(
        submission: Any,
        model: Any,
        val_loader: Iterable[Mapping[str, torch.Tensor]],
        device: torch.device,
        cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        for _ in val_loader:
            pass
        score, score_details = score_time_series_model(
            model,
            materialized_samples,
            list(eval_tasks),
            str(device),
            seed,
        )
        return {"val_loss": score, "score_details": score_details}

    return _evaluate


def _write_summary(path: Path, summary: TrainingSummary) -> Dict[str, Any]:
    payload = {
        "train_metrics": _normalize_values(summary.train_metrics),
        "val_metrics": _normalize_values(summary.val_metrics),
        "num_steps": summary.num_steps,
        "device": summary.device,
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return payload


def _materialize_artifacts(
    root: Path,
    summary: TrainingSummary,
    summary_payload: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    model = summary.model
    model.eval()
    model.to("cpu")

    weights_path = root / "model.safetensors"
    save_as_safetensors(model, weights_path)

    config_path: Optional[Path] = None
    model_config = _extract_model_config(getattr(model, "config", None))
    if model_config is not None:
        config_path = root / "config.json"
        with config_path.open("w", encoding="utf-8") as fh:
            json.dump(model_config, fh, indent=2)

    metrics_path = root / "metrics.json"
    metrics_payload = {
        "train_metrics": summary_payload.get("train_metrics", {}),
        "val_metrics": summary_payload.get("val_metrics", {}),
        "num_steps": summary_payload.get("num_steps", 0),
        "device": summary_payload.get("device"),
    }
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, indent=2)

    manifest = {
        "weights": "model.safetensors",
        "config": "config.json" if config_path is not None else None,
        "metrics": "metrics.json",
        "training_config": config,
    }
    with (root / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def _normalize_values(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    if dataclasses.is_dataclass(obj):
        return {key: _normalize_values(value) for key, value in dataclasses.asdict(obj).items()}
    if isinstance(obj, MutableMapping):
        return {key: _normalize_values(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize_values(value) for value in obj]
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return obj.item()
        except Exception:  # pragma: no cover - guard against unexpected item() behaviour
            return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _extract_model_config(config: Any) -> Optional[Mapping[str, Any]]:
    if config is None:
        return None
    if hasattr(config, "to_dict") and callable(config.to_dict):
        return config.to_dict()
    if dataclasses.is_dataclass(config):
        return dataclasses.asdict(config)
    if hasattr(config, "__dict__"):
        return {
            key: value
            for key, value in config.__dict__.items()
            if not key.startswith("_") and not callable(value)
        }
    return None


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
