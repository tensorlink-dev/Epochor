"""Core evaluation engine for scoring miner submissions under validator control."""

import dataclasses
import logging
import math
import shutil
import threading
import typing
from collections import defaultdict
from pathlib import Path

import bittensor as bt

from epochor.model.model_constraints import Competition
from epochor.model.model_data import Model, ModelId, MinerSubmissionSnapshot, TrainingResultRecord
from epochor.model.storage.disk_model_store import DiskModelStore
from epochor.utils.hashing import hash_directory
from epochor.utils.hf_io import HF_WRITE_TOKEN_ENV, push_artifacts_to_hf, write_meta
from epochor.validation.validation import ScoreDetails

from .sandbox import SandboxError, SandboxRuntimeConfig, run_submission_in_sandbox
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

            try:
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
            except SandboxError as exc:
                logging.error(
                    "Sandbox execution failed",
                    extra={
                        "hotkey": hotkey,
                        "sandbox_error": str(exc),
                    },
                )
                continue

            summary_payload = dict(result.summary)

            if result.stdout:
                logging.debug(
                    "Sandbox stdout",
                    extra={"hotkey": hotkey, "stdout": result.stdout},
                )
            if result.stderr:
                logging.debug(
                    "Sandbox stderr",
                    extra={"hotkey": hotkey, "stderr": result.stderr},
                )

            val_metrics = dict(summary_payload.get("val_metrics", {}))
            score = float(val_metrics.get("val_loss", math.inf))
            score_details = val_metrics.get("score_details", {})

            uid_to_state[uid].score = score
            if isinstance(score_details, dict):
                uid_to_state[uid].score_details = score_details
            uid_to_state[uid].train_metrics = dict(summary_payload.get("train_metrics", {}))
            uid_to_state[uid].val_metrics = val_metrics

            artifacts_dir = Path(result.artifacts_dir)
            meta_payload = {
                "hotkey": hotkey,
                "competition_id": competition.id,
                "block": submission_snapshot.block,
                "summary": summary_payload,
            }
            write_meta(meta_payload, path=artifacts_dir / "model_meta.json")

            try:
                secure_hash = hash_directory(str(artifacts_dir))
            except Exception:
                logging.exception("Failed to hash sandbox artefacts", extra={"hotkey": hotkey})
                self._cleanup_artifacts(artifacts_dir)
                continue

            repo_id = f"{submission_snapshot.model_id.namespace}/{submission_snapshot.model_id.name}"
            try:
                commit_hash = push_artifacts_to_hf(
                    repo_id,
                    local_dir=artifacts_dir,
                    token_env=HF_WRITE_TOKEN_ENV,
                )
            except Exception:
                logging.exception("Failed to upload artefacts to Hugging Face", extra={"hotkey": hotkey})
                self._cleanup_artifacts(artifacts_dir)
                continue

            trained_model_id = ModelId(
                namespace=submission_snapshot.model_id.namespace,
                name=submission_snapshot.model_id.name,
                competition_id=submission_snapshot.model_id.competition_id,
                commit=commit_hash,
                secure_hash=secure_hash,
                hash=secure_hash,
            )

            model_record = Model(id=trained_model_id, model=None, source_path=str(artifacts_dir))
            try:
                self.local_store.store_model(hotkey, model_record)
            except Exception:
                logging.exception("Failed to persist sandbox artefacts locally", extra={"hotkey": hotkey})
                self._cleanup_artifacts(artifacts_dir)
                continue

            training_record = TrainingResultRecord(
                competition_id=competition.id,
                block=submission_snapshot.block,
                train_metrics=dict(summary_payload.get("train_metrics", {})),
                val_metrics=val_metrics,
                num_steps=int(summary_payload.get("num_steps", 0)),
                device=str(summary_payload.get("device", self.device)),
                model_id=trained_model_id,
            )
            self.state.model_tracker.record_training_result(hotkey, training_record)

            self._cleanup_artifacts(artifacts_dir)

        return uid_to_state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _flatten_samples(self, samples: list) -> list:
        flat: list = []
        for batch_list in samples:
            flat.extend(batch_list)
        return flat

    def _cleanup_artifacts(self, path: Path) -> None:
        try:
            shutil.rmtree(path.parent)
        except Exception:
            logging.debug("Failed to remove sandbox artefacts", exc_info=True)

    def _format_submission_name(self, snapshot: MinerSubmissionSnapshot) -> str:
        return f"{snapshot.model_id.namespace}/{snapshot.model_id.name}"
