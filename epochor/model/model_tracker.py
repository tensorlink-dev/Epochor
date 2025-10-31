from __future__ import annotations

import pickle
import threading
from typing import Dict, List, Optional

from epochor.model.model_data import (
    EvalResult,
    MinerSubmissionSnapshot,
    ModelMetadata,
    TrainingResultRecord,
)


class ModelTracker:
    """Tracks miner submissions, training outcomes, and evaluation history."""

    def __init__(self) -> None:
        self.miner_hotkey_to_submission: Dict[str, MinerSubmissionSnapshot] = {}
        self.miner_hotkey_to_training_result: Dict[str, TrainingResultRecord] = {}
        self.miner_hotkey_to_eval_results: Dict[str, Dict[int, List[EvalResult]]] = {}
        self.lock = threading.RLock()

    # ------------------------------------------------------------------
    # Hotkey lifecycle
    # ------------------------------------------------------------------
    def on_hotkeys_updated(self, hotkeys: set[str]) -> None:
        """Prune tracker entries for hotkeys that are no longer active."""

        with self.lock:
            for hotkey in list(self.miner_hotkey_to_submission.keys()):
                if hotkey not in hotkeys:
                    self.miner_hotkey_to_submission.pop(hotkey, None)
                    self.miner_hotkey_to_training_result.pop(hotkey, None)
                    self.miner_hotkey_to_eval_results.pop(hotkey, None)

    # ------------------------------------------------------------------
    # Submission + training updates
    # ------------------------------------------------------------------
    def on_submission_updated(
        self,
        hotkey: str,
        submission: MinerSubmissionSnapshot,
    ) -> None:
        """Record the latest submission snapshot for a miner.

        Any prior training/evaluation history is cleared if the submission changed.
        """

        with self.lock:
            previous = self.miner_hotkey_to_submission.get(hotkey)
            self.miner_hotkey_to_submission[hotkey] = submission
            if previous != submission:
                self.miner_hotkey_to_training_result.pop(hotkey, None)
                if hotkey in self.miner_hotkey_to_eval_results:
                    self.miner_hotkey_to_eval_results[hotkey].clear()

    # Backward compatibility with legacy callers
    def on_model_updated(self, hotkey: str, submission) -> None:  # pragma: no cover - compatibility shim
        if isinstance(submission, MinerSubmissionSnapshot):
            self.on_submission_updated(hotkey, submission)
            return
        if isinstance(submission, ModelMetadata):
            legacy_snapshot = MinerSubmissionSnapshot(
                model_id=submission.id,
                competition_id=submission.id.competition_id,
                block=submission.block,
                snapshot_path="",
            )
            self.on_submission_updated(hotkey, legacy_snapshot)
            return
        raise TypeError("on_model_updated expects a MinerSubmissionSnapshot or ModelMetadata")

    def record_training_result(self, hotkey: str, result: TrainingResultRecord) -> None:
        """Store the validator-recorded training summary for a miner."""

        with self.lock:
            self.miner_hotkey_to_training_result[hotkey] = result

    # ------------------------------------------------------------------
    # Evaluation history
    # ------------------------------------------------------------------
    def on_model_evaluated(self, hotkey: str, competition_id: int, eval_result: EvalResult) -> None:
        with self.lock:
            if hotkey not in self.miner_hotkey_to_eval_results:
                self.miner_hotkey_to_eval_results[hotkey] = {}
            if competition_id not in self.miner_hotkey_to_eval_results[hotkey]:
                self.miner_hotkey_to_eval_results[hotkey][competition_id] = []
            self.miner_hotkey_to_eval_results[hotkey][competition_id].append(eval_result)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_submission_for_miner_hotkey(self, hotkey: str) -> Optional[MinerSubmissionSnapshot]:
        with self.lock:
            return self.miner_hotkey_to_submission.get(hotkey)

    def get_training_result_for_miner_hotkey(self, hotkey: str) -> Optional[TrainingResultRecord]:
        with self.lock:
            return self.miner_hotkey_to_training_result.get(hotkey)

    def get_model_metadata_for_miner_hotkey(self, hotkey: str) -> Optional[ModelMetadata]:  # pragma: no cover - legacy
        submission = self.get_submission_for_miner_hotkey(hotkey)
        if submission is None:
            return None
        return ModelMetadata(id=submission.model_id, block=submission.block)

    def get_eval_results_for_miner_hotkey(self, hotkey: str, competition_id: int) -> List[EvalResult]:
        with self.lock:
            return self.miner_hotkey_to_eval_results.get(hotkey, {}).get(competition_id, [])

    def get_block_last_evaluated(self, hotkey: str) -> Optional[int]:
        with self.lock:
            competitions = self.miner_hotkey_to_eval_results.get(hotkey)
            if not competitions:
                return None
            last_block = 0
            for results in competitions.values():
                if results:
                    last_block = max(last_block, results[-1].block)
            return last_block or None

    def get_miner_hotkey_to_submission_dict(self) -> Dict[str, MinerSubmissionSnapshot]:
        with self.lock:
            return self.miner_hotkey_to_submission.copy()

    def get_miner_hotkey_to_model_metadata_dict(self) -> Dict[str, ModelMetadata]:  # pragma: no cover - legacy
        with self.lock:
            return {
                hotkey: ModelMetadata(id=sub.model_id, block=sub.block)
                for hotkey, sub in self.miner_hotkey_to_submission.items()
            }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_state(self, filepath: str) -> None:
        with self.lock:
            with open(filepath, "wb") as f:
                pickle.dump(self.miner_hotkey_to_submission, f)
                pickle.dump(self.miner_hotkey_to_training_result, f)
                pickle.dump(self.miner_hotkey_to_eval_results, f)

    def load_state(self, filepath: str) -> None:
        with open(filepath, "rb") as f:
            first = pickle.load(f)
            try:
                second = pickle.load(f)
                third = pickle.load(f)
            except EOFError:
                legacy_metadata = first
                legacy_eval_results = second if 'second' in locals() else {}
                self.miner_hotkey_to_submission = {
                    hotkey: MinerSubmissionSnapshot(
                        model_id=meta.id,
                        competition_id=meta.id.competition_id,
                        block=meta.block,
                        snapshot_path="",
                    )
                    for hotkey, meta in legacy_metadata.items()
                }
                self.miner_hotkey_to_training_result = {}
                self.miner_hotkey_to_eval_results = legacy_eval_results
            else:
                self.miner_hotkey_to_submission = first
                self.miner_hotkey_to_training_result = second
                self.miner_hotkey_to_eval_results = third
