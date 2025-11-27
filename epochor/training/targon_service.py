"""Targon app exposing validator-owned training and evaluation."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
import uuid

import targon
from targon import Compute


image = (
    targon.Image.debian_slim("3.12")
    .pip_install("torch")
    .add_local_dir("./", "/app/validator")
    .workdir("/app")
)

app = targon.App(name="epo-validator-api", image=image, project_name="epo-project")


@app.function(resource=Compute.H200_SMALL, timeout=3600, max_replicas=16)
@targon.fastapi_endpoint(method="POST", docs=True, requires_auth=True)
def submit_and_train(payload: dict) -> dict:
    """FastAPI-style endpoint that trains and evaluates a miner submission."""

    if not isinstance(payload, dict):
        raise TypeError("Payload must be a JSON object")
    if "submission_code" not in payload or "cfg" not in payload:
        raise ValueError("Payload must contain 'submission_code' and 'cfg'")

    submission_code = payload["submission_code"]
    cfg = dict(payload["cfg"])
    submission_id = str(payload.get("submission_id") or cfg.get("submission_id") or uuid.uuid4())
    run_id = str(payload.get("run_id") or cfg.get("run_id") or uuid.uuid4())
    cfg.update({"submission_id": submission_id, "run_id": run_id})

    submissions_dir = Path("/app/submissions")
    submissions_dir.mkdir(parents=True, exist_ok=True)
    submission_path = submissions_dir / f"{submission_id}.py"
    submission_path.write_text(submission_code, encoding="utf-8")

    if "/app/validator" not in sys.path:
        sys.path.insert(0, "/app/validator")

    from epochor.training.data_and_eval import evaluate_fn, make_train_loader, make_val_loader
    from epochor.training.validator_runner import load_miner_module, run_training

    submission = load_miner_module(str(submission_path))
    preferred_device = cfg.get("preferred_device", "cuda")
    summary = run_training(
        submission=submission,
        cfg=cfg,
        train_loader_factory=make_train_loader,
        val_loader_factory=make_val_loader,
        evaluate_fn=evaluate_fn,
        max_train_steps=cfg.get("max_train_steps"),
        preferred_device=preferred_device,
        grad_clip_norm=cfg.get("grad_clip_norm"),
        max_memory_bytes=cfg.get("max_memory_bytes"),
        submission_id=submission_id,
        run_id=run_id,
    )

    runs_dir = Path("/app/run_records")
    runs_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "train_metrics": summary.train_metrics,
        "val_metrics": summary.val_metrics,
        "num_steps": summary.num_steps,
        "device": summary.device,
        "submission_id": summary.submission_id,
        "run_id": summary.run_id,
    }
    (runs_dir / f"{summary.run_id}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    return record


async def run_single_training(submission_code: str, cfg: dict) -> dict:
    """Run one isolated training job using a fresh ephemeral app session.

    Each ``async with app.run()`` block starts a clean execution environment;
    calling this helper multiple times guarantees no Python state is shared
    across runs.
    """

    async with app.run():
        return await submit_and_train.remote({"submission_code": submission_code, "cfg": cfg})


@app.local_entrypoint()
def main(submission_file: str, cfg_path: str = "config.json") -> dict:
    """Local helper to trigger remote training for manual testing."""

    submission_code = Path(submission_file).read_text(encoding="utf-8")
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    return asyncio.run(run_single_training(submission_code, cfg))

