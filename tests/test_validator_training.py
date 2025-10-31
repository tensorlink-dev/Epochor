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
        preds = model(batch["x"])
        loss = torch.nn.functional.mse_loss(preds, batch["y"])
        loss.backward()
        optimizer.step()
        return {"loss": float(loss.detach())}


def _fixed_batches(cfg: Dict[str, Any]) -> Iterable[Dict[str, torch.Tensor]]:
    x = torch.tensor([[1.0]], dtype=torch.float32)
    y = torch.tensor([[2.0]], dtype=torch.float32)
    return [
        {"x": x, "y": y},
        {"x": x * 2, "y": y * 2},
        {"x": x * 3, "y": y * 3},
    ]


def _evaluate(model: nn.Module, loader: Iterable[Dict[str, torch.Tensor]], device: torch.device, cfg: Dict[str, Any]) -> Dict[str, Any]:
    losses: List[float] = []
    for batch in loader:
        preds = model(batch["x"].to(device))
        loss = torch.nn.functional.mse_loss(preds, batch["y"].to(device))
        losses.append(float(loss.detach().cpu()))
    return {"val_loss": sum(losses) / len(losses)}


def test_run_training_respects_step_cap():
    submission = _ToySubmission()
    cfg = {"max_steps": 2, "seed": 42}
    summary = run_training(
        submission,
        cfg,
        train_loader_factory=_fixed_batches,
        val_loader_factory=_fixed_batches,
        evaluate_fn=_evaluate,
        max_train_steps=MAX_TRAIN_STEPS,
    )

    assert isinstance(summary, TrainingSummary)
    assert summary.num_steps == 2
    assert len(submission.history) == 2
    assert "loss" in summary.train_metrics
    assert "val_loss" in summary.val_metrics
    assert isinstance(summary.model, nn.Module)


def test_run_training_rejects_missing_loss():
    class BadSubmission(_ToySubmission):
        def train_step(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[override]
            return {}

    submission = BadSubmission()
    cfg: Dict[str, Any] = {"max_steps": 1}

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

    def train_step(self, model, batch, optimizer, step_idx, cfg):
        optimizer.zero_grad(set_to_none=True)
        preds = model(batch["x"])
        loss = torch.nn.functional.mse_loss(preds, batch["y"])
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
