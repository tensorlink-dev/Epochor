from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from epochor.training import (
    MAX_TRAIN_STEPS,
    MinerSubmissionProtocol,
    TrainingSummary,
    load_miner_module,
    run_training,
)


class _ToySubmission(MinerSubmissionProtocol):
    def __init__(self) -> None:
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.history: List[int] = []

    def build_model(self, cfg: Dict[str, Any]) -> nn.Module:
        self.model = nn.Linear(1, 1)
        for param in self.model.parameters():
            nn.init.constant_(param, 0.5)
        return self.model

    def build_optimizer(self, model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
        self.optimizer = torch.optim.SGD(model.parameters(), lr=float(cfg.get("lr", 0.1)))
        return self.optimizer

    def process_data(self, batch: Dict[str, torch.Tensor], cfg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        context_len = int(cfg["context_length"])
        pred_len = int(cfg["prediction_length"])
        sequence = batch["x"]
        context = sequence[:, :context_len]
        target = sequence[:, context_len : context_len + pred_len]
        return {"inputs": context, "targets": target}

    def forecast(self, model: nn.Module, inputs: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
        return model(inputs)

    def train_step(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        step_idx: int,
        cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        self.history.append(step_idx)
        optimizer.zero_grad(set_to_none=True)
        processed = self.process_data(batch, cfg)
        preds = model(processed["inputs"])
        loss = torch.nn.functional.mse_loss(preds, processed["targets"])
        loss.backward()
        optimizer.step()
        return {"loss": float(loss.detach())}


def _fixed_batches(cfg: Dict[str, Any]) -> Iterable[Dict[str, torch.Tensor]]:
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    return [
        {"x": x},
        {"x": x * 2},
        {"x": x * 3},
    ]


def _evaluate(
    submission: _ToySubmission,
    model: nn.Module,
    loader: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    losses: List[float] = []
    for batch in loader:
        batch_on_device = {k: v.to(device) for k, v in batch.items()}
        processed = submission.process_data(batch_on_device, cfg)
        preds = model(processed["inputs"])
        loss = torch.nn.functional.mse_loss(preds, processed["targets"])
        losses.append(float(loss.detach().cpu()))
    return {"val_loss": sum(losses) / len(losses)}


def test_run_training_respects_step_cap():
    submission = _ToySubmission()
    cfg = {"max_steps": 2, "seed": 42, "context_length": 1, "prediction_length": 1}
    summary = run_training(
        submission,
        cfg,
        train_loader_factory=_fixed_batches,
        val_loader_factory=_fixed_batches,
        evaluate_fn=_evaluate,
        max_train_steps=MAX_TRAIN_STEPS,
        submission_id="sub-1",
        run_id="run-1",
    )

    assert isinstance(summary, TrainingSummary)
    assert summary.num_steps == 2
    assert len(submission.history) == 2
    assert "loss" in summary.train_metrics
    assert "val_loss" in summary.val_metrics
    assert isinstance(summary.model, nn.Module)
    assert summary.submission_id == "sub-1"
    assert summary.run_id == "run-1"


def test_run_training_rejects_missing_loss():
    class BadSubmission(_ToySubmission):
        def train_step(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[override]
            return {}

    submission = BadSubmission()
    cfg: Dict[str, Any] = {"max_steps": 1, "context_length": 1, "prediction_length": 1}

    try:
        run_training(
            submission,
            cfg,
            train_loader_factory=_fixed_batches,
            val_loader_factory=_fixed_batches,
            evaluate_fn=_evaluate,
        )
    except ValueError as exc:
        assert "loss" in str(exc)
    else:  # pragma: no cover - defensive, ensure failure if no error
        raise AssertionError("Expected ValueError for missing loss")


def test_load_miner_module(tmp_path: Path):
    module_code = """
from typing import Any, Dict
import torch
import torch.nn as nn

from epochor.training.validator_contract import MinerSubmissionProtocol


class Demo(MinerSubmissionProtocol):
    def build_model(self, cfg: Dict[str, Any]) -> nn.Module:
        return nn.Linear(1, 1)

    def build_optimizer(self, model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
        return torch.optim.SGD(model.parameters(), lr=0.1)

    def process_data(self, batch, cfg):
        return {"inputs": batch["x"], "targets": batch["x"]}

    def forecast(self, model, inputs, cfg):
        return model(inputs)

    def train_step(self, model, batch, optimizer, step_idx, cfg):
        optimizer.zero_grad(set_to_none=True)
        processed = self.process_data(batch, cfg)
        preds = model(processed["inputs"])
        loss = torch.nn.functional.mse_loss(preds, processed["targets"])
        loss.backward()
        optimizer.step()
        return {"loss": float(loss.detach())}


def get_submission() -> MinerSubmissionProtocol:
    return Demo()
"""
    submission_file = tmp_path / "miner_submission.py"
    submission_file.write_text(module_code)

    submission = load_miner_module(str(submission_file))
    assert isinstance(submission, MinerSubmissionProtocol)
