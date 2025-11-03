"""Core evaluation engine for scoring miner submissions under validator control."""

import dataclasses
import json
import logging
import math
import threading
import traceback
import typing
from collections import defaultdict

import bittensor as bt

from epochor.model.model_constraints import Competition
from epochor.model.storage.disk_model_store import DiskModelStore
from epochor.model.model_data import MinerSubmissionSnapshot, TrainingResultRecord
from epochor.validation.validation import ScoreDetails

from .sandbox import SandboxRuntimeConfig, run_submission_in_sandbox
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
    def __init__(
        self,
        state: ValidatorState,
        metagraph: "bt.metagraph",
        local_store: DiskModelStore,
        device: str,
        metagraph_lock: threading.RLock,
        sandbox_runtime: SandboxRuntimeConfig,
    ):
        """Initializes the EvaluationService."""
        self.state = state
        self.metagraph = metagraph
        self.local_store = local_store
        self.device = device
        self.metagraph_lock = metagraph_lock
        self.sandbox_runtime = sandbox_runtime

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

            result = run_submission_in_sandbox(
                submission_snapshot.snapshot_path,
                competition_id=int(competition.id),
                seed=seed,
                train_batches=train_batches,
                samples=samples,
                eval_tasks=eval_tasks,
                preferred_device=self.device,
                runtime=self.sandbox_runtime,
            )

            if not result.ok:
                logging.error(
                    "Sandbox execution failed",
                    extra={
                        "hotkey": hotkey,
                        "sandbox_status": result.status,
                        "sandbox_error": result.error,
                        "sandbox_returncode": result.returncode,
                    },
                )
                continue

            try:
                summary_payload = json.loads(result.summary_json)
            except json.JSONDecodeError:
                logging.error(
                    "Sandbox summary parse failure",
                    extra={
                        "hotkey": hotkey,
                        "sandbox_status": "summary_parse_error",
                        "sandbox_error": traceback.format_exc(),
                    },
                )
                continue

            val_metrics = dict(summary_payload.get("val_metrics", {}))
            score = float(val_metrics.get("val_loss", math.inf))
            score_details = val_metrics.get("score_details", {})

            uid_to_state[uid].score = score
            if isinstance(score_details, dict):
                uid_to_state[uid].score_details = score_details
            uid_to_state[uid].train_metrics = dict(summary_payload.get("train_metrics", {}))
            uid_to_state[uid].val_metrics = val_metrics

            training_record = TrainingResultRecord(
                competition_id=competition.id,
                block=submission_snapshot.block,
                train_metrics=dict(summary_payload.get("train_metrics", {})),
                val_metrics=val_metrics,
                num_steps=int(summary_payload.get("num_steps", 0)),
                device=str(summary_payload.get("device", self.device)),
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

    def _format_submission_name(self, snapshot: MinerSubmissionSnapshot) -> str:
        return f"{snapshot.model_id.namespace}/{snapshot.model_id.name}"
