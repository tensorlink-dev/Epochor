"""Core evaluation engine for scoring miner submissions under validator control."""

import dataclasses
import logging
import math
import os
import threading
import traceback
import typing
from collections import defaultdict

import bittensor as bt

from epochor.model.model_constraints import Competition
from epochor.model.storage.disk_model_store import DiskModelStore
from epochor.training import load_miner_module, run_training
from epochor.model.model_data import MinerSubmissionSnapshot, TrainingResultRecord
from epochor.validation.validation import ScoreDetails, score_time_series_model

from .state import ValidatorState


@dataclasses.dataclass
class PerUIDEvalState:
    """State tracked per UID during a single evaluation run."""
    block: int = math.inf
    hotkey: str = "Unknown"
    repo_name: str = "Unknown"
    score: float = math.inf
    score_details: typing.Dict[str, ScoreDetails] = dataclasses.field(default_factory=dict)
    train_metrics: typing.Dict[str, typing.Any] = dataclasses.field(default_factory=dict)
    val_metrics: typing.Dict[str, typing.Any] = dataclasses.field(default_factory=dict)


class EvaluationService:
    """
    Acts as the core evaluation engine. It takes a list of UIDs and prepared data,
    executes the validator-owned training loop for each miner submission, and
    produces scoring artefacts for downstream weighting.
    """
    def __init__(self, state: ValidatorState, metagraph: "bt.metagraph", local_store: DiskModelStore, device: str, metagraph_lock: threading.RLock):
        """Initializes the EvaluationService."""
        self.state = state
        self.metagraph = metagraph
        self.local_store = local_store
        self.device = device
        self.metagraph_lock = metagraph_lock

    def evaluate_uids(
        self,
        uids: list[int],
        competition: Competition,
        samples: list,
        eval_tasks: list,
        seed: int,
    ) -> dict[int, PerUIDEvalState]:
        """Train and evaluate each UID's submission, returning performance details."""

        uid_to_state: dict[int, PerUIDEvalState] = defaultdict(PerUIDEvalState)
        train_batches = self._flatten_samples(samples)
        if not train_batches:
            logging.warning("No batches available for training; skipping evaluation.")
            return uid_to_state

        train_factory = self._make_loader_factory(train_batches)
        val_factory = self._make_loader_factory(train_batches)
        evaluate_fn = self._make_evaluate_fn(samples, eval_tasks, seed)

        for uid in uids:
            with self.metagraph_lock:
                hotkey = self.metagraph.hotkeys[uid]
            uid_to_state[uid].hotkey = hotkey

            submission_snapshot = self.state.model_tracker.get_submission_for_miner_hotkey(hotkey)
            if not submission_snapshot or submission_snapshot.competition_id != competition.id:
                continue

            uid_to_state[uid].block = submission_snapshot.block
            uid_to_state[uid].repo_name = self._format_submission_name(submission_snapshot)

            if not submission_snapshot.snapshot_path:
                logging.error(f"No cached submission path for hotkey {hotkey}")
                continue

            submission_path = self._find_submission_file(submission_snapshot.snapshot_path)
            if submission_path is None:
                logging.error(f"miner_submission.py not found for hotkey {hotkey}")
                continue

            try:
                submission = load_miner_module(submission_path)
            except Exception:
                logging.error(f"Failed to load submission for {hotkey}: {traceback.format_exc()}")
                continue

            cfg = {
                "competition_id": int(competition.id),
                "seed": seed,
                "max_steps": len(train_batches),
                "max_epochs": 1,
            }

            try:
                summary = run_training(
                    submission,
                    cfg,
                    train_loader_factory=train_factory,
                    val_loader_factory=val_factory,
                    evaluate_fn=evaluate_fn,
                    preferred_device=self.device,
                )
            except Exception:
                logging.error(f"Training failed for UID {uid}: {traceback.format_exc()}")
                continue

            val_metrics = dict(summary.val_metrics)
            score = float(val_metrics.get("val_loss", math.inf))
            score_details = val_metrics.get("score_details", {})

            if summary.model is not None:
                summary.model = summary.model.to("cpu")

            uid_to_state[uid].score = score
            if isinstance(score_details, dict):
                uid_to_state[uid].score_details = score_details
            uid_to_state[uid].train_metrics = dict(summary.train_metrics)
            uid_to_state[uid].val_metrics = val_metrics

            training_record = TrainingResultRecord(
                competition_id=competition.id,
                block=submission_snapshot.block,
                train_metrics=dict(summary.train_metrics),
                val_metrics=val_metrics,
                num_steps=summary.num_steps,
                device=summary.device,
            )
            self.state.model_tracker.record_training_result(hotkey, training_record)

        return uid_to_state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _flatten_samples(self, samples: list) -> list:
        flat: list = []
        for batch_list in samples:
            flat.extend(batch_list)
        return flat

    def _make_loader_factory(self, batches: list):
        def factory(cfg: dict):
            for batch in batches:
                yield batch

        return factory

    def _make_evaluate_fn(self, samples: list, eval_tasks: list, seed: int):
        def evaluate(model, loader, device, cfg):
            for _ in loader:
                pass
            score, score_details = score_time_series_model(
                model,
                samples,
                eval_tasks,
                str(device),
                seed,
            )
            return {"val_loss": score, "score_details": score_details}

        return evaluate

    def _format_submission_name(self, snapshot: MinerSubmissionSnapshot) -> str:
        return f"{snapshot.model_id.namespace}/{snapshot.model_id.name}"

    def _find_submission_file(self, snapshot_dir: str) -> typing.Optional[str]:
        if not os.path.isdir(snapshot_dir):
            return None
        for root, _, files in os.walk(snapshot_dir):
            if "miner_submission.py" in files:
                return os.path.join(root, "miner_submission.py")
        return None
