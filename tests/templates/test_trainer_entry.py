from __future__ import annotations

import hashlib
import os
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
from safetensors.torch import save_file as save_safetensors

from templates.validator_training import trainer_entry
from templates.validator_training.miner_protocol import MinerSubmissionProtocol


SUBMISSION_CODE = """
from typing import Any, Dict

import torch
from torch import nn

from templates.validator_training.miner_protocol import MinerSubmissionProtocol


class Submission(MinerSubmissionProtocol):
    def __init__(self) -> None:
        self._model: nn.Module | None = None

    def build_model(self, cfg: Dict[str, Any]) -> nn.Module:
        model = nn.Linear(int(cfg.get("input_dim", 16)), 1)
        for param in model.parameters():
            nn.init.constant_(param, 0.1)
        self._model = model
        return model

    def build_optimizer(self, model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
        return torch.optim.SGD(model.parameters(), lr=float(cfg.get("lr", 0.01)))

    def train_step(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        step_idx: int,
        cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        optimizer.zero_grad(set_to_none=True)
        preds = model(batch["x"])
        loss = torch.nn.functional.mse_loss(preds, batch["y"])
        loss.backward()
        optimizer.step()
        return {"loss": float(loss.detach().cpu())}
"""


@pytest.fixture(name="submission_dir")
def _submission_dir(tmp_path: Path) -> Path:
    submission_dir = tmp_path / "submission"
    submission_dir.mkdir()
    (submission_dir / "miner.py").write_text(SUBMISSION_CODE)
    return submission_dir


@pytest.fixture(name="artifacts_dir")
def _artifacts_dir(tmp_path: Path) -> Path:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    return artifacts_dir


async def _run_training(
    cfg: Dict[str, Any],
    submission_dir: Path,
    artifacts_dir: Path,
    *,
    lease: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    env = os.environ.copy()
    env["SUBMISSION_DIR"] = str(submission_dir)
    env["ARTIFACTS_DIR"] = str(artifacts_dir)

    # run() reads environment variables directly; patch via os.environ
    old_submission = os.environ.get("SUBMISSION_DIR")
    old_artifacts = os.environ.get("ARTIFACTS_DIR")
    os.environ["SUBMISSION_DIR"] = env["SUBMISSION_DIR"]
    os.environ["ARTIFACTS_DIR"] = env["ARTIFACTS_DIR"]
    try:
        lease_payload = {"round": 1, "submission_id": "sub-001", "model_id": "model-001", "miner_hotkey": "miner-hotkey"}
        if lease:
            lease_payload.update(lease)
        result = await trainer_entry.run({"cfg": cfg, "lease": lease_payload})
    finally:
        if old_submission is None:
            os.environ.pop("SUBMISSION_DIR", None)
        else:
            os.environ["SUBMISSION_DIR"] = old_submission
        if old_artifacts is None:
            os.environ.pop("ARTIFACTS_DIR", None)
        else:
            os.environ["ARTIFACTS_DIR"] = old_artifacts
    return result


@pytest.fixture(name="hf_uploads")
def _hf_uploads(monkeypatch: pytest.MonkeyPatch) -> List[Dict[str, Any]]:
    monkeypatch.setenv("HF_TOKEN", "dummy-token")
    uploads: List[Dict[str, Any]] = []

    def _fake_push(
        repo_id: str,
        *,
        local_dir: os.PathLike[str] | str,
        token_env: str = "HF_TOKEN",
        private: bool = True,
        commit_message: str | None = None,
    ) -> str:
        uploads.append(
            {
                "repo_id": repo_id,
                "local_dir": Path(local_dir),
                "token_env": token_env,
                "private": private,
                "commit_message": commit_message,
            }
        )
        return "fake-commit"

    monkeypatch.setattr(trainer_entry, "push_artifacts_to_hf", _fake_push)
    return uploads


@pytest.mark.asyncio
async def test_deterministic_seed(submission_dir: Path, artifacts_dir: Path, hf_uploads: List[Dict[str, Any]]) -> None:
    cfg = {
        "seed": 123,
        "max_steps": 25,
        "max_seconds": 5,
        "output_tag": "det",
        "train_batch_size": 32,
        "input_dim": 8,
    }
    first = await _run_training(cfg, submission_dir, artifacts_dir)
    second = await _run_training(cfg, submission_dir, artifacts_dir)
    assert first["artifact_id"] == second["artifact_id"]
    assert pytest.approx(first["best_loss"], rel=1e-6) == second["best_loss"]


@pytest.mark.asyncio
async def test_time_cap_enforced(submission_dir: Path, artifacts_dir: Path, hf_uploads: List[Dict[str, Any]]) -> None:
    cfg = {"seed": 7, "max_steps": 10_000, "max_seconds": 0.2, "output_tag": "cap"}
    result = await _run_training(cfg, submission_dir, artifacts_dir)
    assert result["elapsed_seconds"] <= cfg["max_seconds"] + 0.5
    assert result["steps"] < cfg["max_steps"]


@pytest.mark.asyncio
async def test_protocol_enforced(tmp_path: Path, artifacts_dir: Path, hf_uploads: List[Dict[str, Any]]) -> None:
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "miner.py").write_text("class NotSubmission: pass\n")
    old_submission = os.environ.get("SUBMISSION_DIR")
    old_artifacts = os.environ.get("ARTIFACTS_DIR")
    os.environ["ARTIFACTS_DIR"] = str(artifacts_dir)
    os.environ["SUBMISSION_DIR"] = str(bad_dir)
    try:
        with pytest.raises((ValueError, TypeError)):
            await trainer_entry.run({"cfg": {}, "lease": {}})
    finally:
        if old_submission is None:
            os.environ.pop("SUBMISSION_DIR", None)
        else:
            os.environ["SUBMISSION_DIR"] = old_submission
        if old_artifacts is None:
            os.environ.pop("ARTIFACTS_DIR", None)
        else:
            os.environ["ARTIFACTS_DIR"] = old_artifacts


@pytest.mark.asyncio
async def test_resume_weights_optional(submission_dir: Path, artifacts_dir: Path, hf_uploads: List[Dict[str, Any]]) -> None:
    resume_dir = submission_dir / "weights"
    resume_dir.mkdir()
    resume_path = resume_dir / "resume.safetensors"

    # Build a model and save deterministic state
    input_dim = 16
    model = torch.nn.Linear(input_dim, 1)
    torch.nn.init.ones_(model.weight)
    torch.nn.init.zeros_(model.bias)
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    save_safetensors(state, str(resume_path))

    cfg = {"seed": 5, "max_steps": 5, "max_seconds": 5, "resume_ckpt_path": str(resume_path)}
    result = await _run_training(cfg, submission_dir, artifacts_dir)
    assert result["ok"] is True
    assert result["steps"] > 0


@pytest.mark.asyncio
async def test_checkpoint_and_hash(submission_dir: Path, artifacts_dir: Path, hf_uploads: List[Dict[str, Any]]) -> None:
    cfg = {"seed": 11, "max_steps": 3, "max_seconds": 5, "output_tag": "hash"}
    result = await _run_training(cfg, submission_dir, artifacts_dir)
    ckpt_path = Path(result["checkpoint_path"])
    assert ckpt_path.exists()
    with open(ckpt_path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    assert result["artifact_id"] == f"sha256:{digest}"
    assert result["metadata_path"].endswith("_metadata.json")
    assert result["hf_commit_hash"] == "fake-commit"
    assert hf_uploads, "expected Hugging Face upload"
    upload = hf_uploads[0]
    assert upload["repo_id"].startswith("miner-hotkey/")
    metadata = json.loads(Path(result["metadata_path"]).read_text())
    assert metadata["lease"]["miner_hotkey"] == "miner-hotkey"
