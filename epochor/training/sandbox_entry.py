"""Entry point executed inside the sandbox container."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Sequence

import torch

from .validator_runner import TrainingSummary, load_miner_module, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validator training inside sandbox")
    parser.add_argument("--staging", required=True, help="Path to staged input directory")
    parser.add_argument("--submission", required=True, help="Path to the submission directory")
    parser.add_argument("--output", required=True, help="Destination for the training summary")
    parser.add_argument("--config-name", default="cfg.json", help="Name of the staged config file")
    parser.add_argument(
        "--train-name", default="train_batches.pt", help="Name of the staged training batches file"
    )
    parser.add_argument(
        "--val-name", default="val_batches.pt", help="Name of the staged validation batches file"
    )
    args = parser.parse_args()

    staging_dir = Path(args.staging)
    cfg_path = staging_dir / args.config_name
    train_path = staging_dir / args.train_name
    val_path = staging_dir / args.val_name

    config = _load_json(cfg_path)
    train_batches = torch.load(train_path)
    val_batches = torch.load(val_path)

    submission_file = _resolve_submission(Path(args.submission))
    submission = load_miner_module(str(submission_file))

    evaluate_fn = _resolve_evaluate_fn(config.get("evaluate_fn"))
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

    _write_summary(Path(args.output), summary)


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


def _resolve_evaluate_fn(spec: Any) -> Callable[[Any, Iterable[Mapping[str, torch.Tensor]], torch.device, Dict[str, Any]], Dict[str, Any]]:
    if spec is None:
        return _default_evaluate_fn
    if not isinstance(spec, Mapping):
        raise TypeError("evaluate_fn specification must be a mapping")
    module_path = spec.get("module")
    attribute = spec.get("attribute") or spec.get("name")
    if not module_path or not attribute:
        raise ValueError("evaluate_fn specification requires 'module' and 'attribute'")
    module = importlib.import_module(str(module_path))
    fn = module
    for part in str(attribute).split("."):
        fn = getattr(fn, part)
    if not callable(fn):
        raise TypeError("Resolved evaluate_fn is not callable")
    return fn  # type: ignore[return-value]


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


def _default_evaluate_fn(
    model: Any,
    val_loader: Iterable[Mapping[str, torch.Tensor]],
    device: torch.device,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    del model, device, cfg
    count = 0
    for _ in val_loader:
        count += 1
    return {"num_validation_batches": count}


def _write_summary(path: Path, summary: TrainingSummary) -> None:
    payload = {
        "train_metrics": _normalize_values(summary.train_metrics),
        "val_metrics": _normalize_values(summary.val_metrics),
        "num_steps": summary.num_steps,
        "device": summary.device,
    }
    if path.suffix == ".pt":
        torch.save(payload, path)
    else:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)


def _normalize_values(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
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


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
