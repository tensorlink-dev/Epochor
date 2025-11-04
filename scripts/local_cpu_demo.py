"""Generate toy submissions and execute them in the validator sandbox on CPU."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from typing import Dict, List, Sequence

import torch

from epochor.generators.synthetic_v1 import SyntheticBenchmarkerV1
from epochor.model.model_constraints import DatasetId, EvalMethodId, EvalTask


_SUBMISSION_TEMPLATES: Dict[str, str] = {
    "moving_average": dedent(
        """
        from __future__ import annotations

        from typing import Any, Dict, Sequence

        import torch
        from torch import nn
        import torch.nn.functional as F

        from epochor.model.base import BaseTemporalModel, TemporalModelOutput
        from epochor.training.validator_contract import MinerSubmissionProtocol


        class MovingAverageForecaster(BaseTemporalModel):
            '''Tiny forecaster that learns an affine rescaling of a rolling mean.'''

            def __init__(self, window: int = 4) -> None:
                super().__init__()
                self.window = int(window)
                self.scale = nn.Parameter(torch.tensor(1.0))
                self.bias = nn.Parameter(torch.tensor(0.0))

            def forward(self, context: torch.Tensor, **_: Any) -> TemporalModelOutput:
                if context.dim() == 2:
                    context = context.unsqueeze(-1)
                preds = self.scale * context + self.bias
                return self._to_output({"predictions": preds})

            def forecast(
                self,
                inputs: torch.Tensor,
                *,
                prediction_length: int,
                quantiles: Sequence[float],
                **_: Any,
            ) -> torch.Tensor:
                context = inputs[..., 0]
                window = min(self.window, context.shape[1])
                mean = context[:, -window:].mean(dim=1, keepdim=True)
                mean = self.scale * mean + self.bias
                preds = mean.unsqueeze(1).unsqueeze(-1)
                return preds.repeat(1, prediction_length, 1, len(list(quantiles)))


        class Submission(MinerSubmissionProtocol):
            def build_model(self, cfg: Dict[str, Any]) -> nn.Module:
                window = int(cfg.get("moving_average_window", 4))
                return MovingAverageForecaster(window=window)

            def build_optimizer(self, model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
                return torch.optim.SGD(model.parameters(), lr=float(cfg.get("lr", 0.05)))

            def train_step(
                self,
                model: nn.Module,
                batch: Dict[str, torch.Tensor],
                optimizer: torch.optim.Optimizer,
                step_idx: int,
                cfg: Dict[str, Any],
            ) -> Dict[str, Any]:
                optimizer.zero_grad(set_to_none=True)
                target = batch["y"]
                forecast = model.forecast(
                    batch["x"],
                    prediction_length=target.shape[1],
                    quantiles=[0.5],
                )
                preds = forecast[..., 0]
                loss = F.mse_loss(preds, target)
                loss.backward()
                optimizer.step()
                return {"loss": float(loss.detach().cpu())}


        def get_submission() -> MinerSubmissionProtocol:
            return Submission()
        """
    ),
    "trend": dedent(
        """
        from __future__ import annotations

        from typing import Any, Dict, Sequence

        import torch
        from torch import nn
        import torch.nn.functional as F

        from epochor.model.base import BaseTemporalModel, TemporalModelOutput
        from epochor.training.validator_contract import MinerSubmissionProtocol


        class LinearTrendForecaster(BaseTemporalModel):
            '''Captures a per-series drift parameter learned during training.'''

            def __init__(self) -> None:
                super().__init__()
                self.drift = nn.Parameter(torch.tensor(0.0))

            def forward(self, context: torch.Tensor, **_: Any) -> TemporalModelOutput:
                if context.dim() == 2:
                    context = context.unsqueeze(-1)
                return self._to_output({"predictions": context})

            def forecast(
                self,
                inputs: torch.Tensor,
                *,
                prediction_length: int,
                quantiles: Sequence[float],
                **_: Any,
            ) -> torch.Tensor:
                context = inputs[..., 0]
                last = context[:, -1]
                if context.shape[1] > 1:
                    trend = context[:, -1] - context[:, -2]
                else:
                    trend = torch.zeros_like(last)

                preds = []
                current = last
                for _ in range(int(prediction_length)):
                    current = current + trend + self.drift
                    preds.append(current.unsqueeze(-1).unsqueeze(-1))
                stacked = torch.stack(preds, dim=1)
                return stacked.repeat(1, 1, 1, len(list(quantiles)))


        class Submission(MinerSubmissionProtocol):
            def build_model(self, cfg: Dict[str, Any]) -> nn.Module:
                return LinearTrendForecaster()

            def build_optimizer(self, model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
                return torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr", 0.01)))

            def train_step(
                self,
                model: nn.Module,
                batch: Dict[str, torch.Tensor],
                optimizer: torch.optim.Optimizer,
                step_idx: int,
                cfg: Dict[str, Any],
            ) -> Dict[str, Any]:
                optimizer.zero_grad(set_to_none=True)
                target = batch["y"]
                preds = model.forecast(
                    batch["x"],
                    prediction_length=target.shape[1],
                    quantiles=[0.5],
                )
                mean_preds = preds[..., 0]
                loss = F.mse_loss(mean_preds, target)
                loss.backward()
                optimizer.step()
                return {"loss": float(loss.detach().cpu())}


        def get_submission() -> MinerSubmissionProtocol:
            return Submission()
        """
    ),
}


def _write_submission_templates(destination: Path, *, force: bool) -> Dict[str, Path]:
    created: Dict[str, Path] = {}
    for name, template in _SUBMISSION_TEMPLATES.items():
        submission_dir = destination / name
        submission_dir.mkdir(parents=True, exist_ok=True)
        submission_file = submission_dir / "submission.py"
        if force or not submission_file.exists():
            submission_file.write_text(template.lstrip(), encoding="utf-8")
        created[name] = submission_file
    return created


def _build_training_batches(
    *,
    context_length: int,
    prediction_length: int,
    phases: Sequence[float],
    noise_std: float,
    seed: int,
) -> List[Dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    batches: List[Dict[str, torch.Tensor]] = []
    total_steps = context_length + prediction_length

    for phase in phases:
        steps = torch.linspace(0, 2 * math.pi, total_steps, dtype=torch.float32)
        base = torch.sin(steps + phase)
        noise = torch.randn(total_steps, generator=generator) * noise_std
        series = base + noise

        context = series[:context_length].clone()
        target = series[context_length:].clone()

        batches.append(
            {
                "x": context.unsqueeze(0).unsqueeze(-1),
                "y": target.unsqueeze(0).unsqueeze(-1),
            }
        )

    return batches


def _stage_validator_payload(
    staging_dir: Path,
    *,
    seed: int,
    context_length: int,
    prediction_length: int,
) -> None:
    staging_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "training_cfg": {
            "lr": 0.05,
            "max_steps": 3,
            "seed": seed,
            "moving_average_window": min(6, context_length),
        },
        "max_train_steps": 3,
        "preferred_device": "cpu",
        "grad_clip_norm": None,
        "max_memory_bytes": None,
        "evaluation_seed": seed,
    }

    train_batches = _build_training_batches(
        context_length=context_length,
        prediction_length=prediction_length,
        phases=[0.0, 0.7, 1.4],
        noise_std=0.02,
        seed=seed,
    )
    val_batches = _build_training_batches(
        context_length=context_length,
        prediction_length=prediction_length,
        phases=[0.35, 1.05],
        noise_std=0.02,
        seed=seed + 1,
    )

    benchmarker = SyntheticBenchmarkerV1(length=context_length + prediction_length, n_series=6)
    evaluation_payload = benchmarker.prepare_data(seed=seed)

    evaluation_samples = [
        [
            {
                "inputs_padded": evaluation_payload["inputs_padded"],
                "targets_padded": evaluation_payload["targets_padded"],
                "attention_mask": evaluation_payload["attention_mask"],
                "actual_target_lengths": evaluation_payload["actual_target_lengths"],
            }
        ]
    ]

    evaluation_tasks = [
        EvalTask(
            name="synthetic-demo",
            method_id=EvalMethodId.CRPS_LOSS,
            dataset_id=DatasetId.UNIVARIATE_SYNTHETIC,
            quantiles=[0.1, 0.5, 0.9],
            dataset_kwargs={
                "length": context_length + prediction_length,
                "n_series": 6,
            },
            weight=1.0,
        )
    ]

    (staging_dir / "cfg.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    torch.save(train_batches, staging_dir / "train_batches.pt")
    torch.save(val_batches, staging_dir / "val_batches.pt")
    torch.save(evaluation_samples, staging_dir / "eval_samples.pt")
    torch.save(evaluation_tasks, staging_dir / "eval_tasks.pt")


def _run_sandbox(
    *,
    staging_dir: Path,
    submission_file: Path,
    output_dir: Path,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    artifacts_dir = output_dir / "artifacts"

    cmd = [
        sys.executable,
        "-m",
        "epochor.training.sandbox_entry",
        "--staging",
        str(staging_dir),
        "--submission",
        str(submission_file),
        "--output",
        str(summary_path),
        "--artifacts",
        str(artifacts_dir),
    ]

    subprocess.run(cmd, check=True)
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate toy submissions and run them through the sandbox on CPU.",
    )
    parser.add_argument(
        "--output-dir",
        default="local_demo",
        help="Directory where staging files, runs, and submissions will be written.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=24,
        help="Length of the conditioning window for generated batches.",
    )
    parser.add_argument(
        "--prediction-length",
        type=int,
        default=12,
        help="Number of future steps each model should forecast during training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed used for synthetic data generation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing submission templates if they already exist.",
    )

    args = parser.parse_args(argv)

    output_root = Path(args.output_dir).resolve()
    submissions_dir = output_root / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    submission_files = _write_submission_templates(submissions_dir, force=args.force)

    staging_dir = output_root / "staging"
    _stage_validator_payload(
        staging_dir,
        seed=args.seed,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
    )

    runs_dir = output_root / "runs"
    results: Dict[str, Dict[str, object]] = {}
    for name, submission_path in submission_files.items():
        run_dir = runs_dir / name
        summary = _run_sandbox(
            staging_dir=staging_dir,
            submission_file=submission_path,
            output_dir=run_dir,
        )
        results[name] = summary

    print("Sandbox run complete. Summaries:")
    for name in sorted(results):
        summary = results[name]
        train_loss = summary.get("train_metrics", {}).get("loss")
        val_loss = summary.get("val_metrics", {}).get("val_loss")
        print(f"  - {name}: train_loss={train_loss!r}, val_loss={val_loss!r}")


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
