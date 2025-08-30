"""Manages the persistent state of the validator, including model and competition trackers."""

import os
import pickle
import logging
import typing
import threading
import copy
from collections import defaultdict

import bittensor as bt
import constants
from epochor.utils import misc as utils
from epochor.validation.ema_tracker import CompetitionEMATracker
from epochor.model.model_tracker import ModelTracker


class ValidatorState:
    """
    Manages the state of the validator, including model metadata, competition scores,
    and UIDs pending evaluation. It handles loading state from disk and saving it back.
    """
    MODEL_TRACKER_FILENAME = "model_tracker.pickle"
    COMPETITION_TRACKER_FILENAME = "competition_tracker.json"
    UIDS_FILENAME = "uids.pickle"
    VERSION_FILENAME = "version.txt"

    def __init__(self, metagraph: "bt.metagraph", base_dir: str, metagraph_lock: threading.RLock):
        """Initializes the ValidatorState."""
        self.base_dir = base_dir
        self.uids_filepath = os.path.join(base_dir, self.UIDS_FILENAME)
        self.model_tracker_filepath = os.path.join(base_dir, self.MODEL_TRACKER_FILENAME)
        self.competition_tracker_filepath = os.path.join(base_dir, self.COMPETITION_TRACKER_FILENAME)
        self.version_filepath = os.path.join(base_dir, self.VERSION_FILENAME)

        self.model_tracker = ModelTracker()
        self.ema_tracker = CompetitionEMATracker(num_neurons=len(metagraph.uids))

        self.uids_to_eval: typing.Dict[int, typing.Set[int]] = defaultdict(set)
        self.pending_uids_to_eval: typing.Dict[int, typing.Set[int]] = defaultdict(set)
        self.pending_uids_to_eval_lock = threading.RLock()
        self.metagraph_lock = metagraph_lock

    # ---------------------------
    # Persistence
    # ---------------------------
    def load(self):
        """Loads the validator's state from disk, handling version upgrades."""
        previous_version = utils.get_version(self.version_filepath)
        utils.save_version(self.version_filepath, constants.__spec_version__)

        if previous_version != constants.__spec_version__:
            bt.logging.info(f"Validator updated. Wiping state for re-evaluation.")
            if os.path.exists(self.uids_filepath):
                os.remove(self.uids_filepath)
            if os.path.exists(self.model_tracker_filepath):
                os.remove(self.model_tracker_filepath)

        if os.path.exists(self.model_tracker_filepath):
            try:
                self.model_tracker.load_state(self.model_tracker_filepath)
            except Exception as e:
                bt.logging.warning(f"Failed to load model tracker: {e}. Starting fresh.")

        if os.path.exists(self.competition_tracker_filepath):
            try:
                self.ema_tracker.load(self.competition_tracker_filepath)
            except Exception as e:
                bt.logging.warning(f"Failed to load competition tracker: {e}. Starting fresh.")

        if os.path.exists(self.uids_filepath):
            try:
                with open(self.uids_filepath, "rb") as f:
                    self.uids_to_eval = pickle.load(f)
                    self.pending_uids_to_eval = pickle.load(f)
            except Exception as e:
                bt.logging.warning(f"Failed to load UIDs: {e}. Wiping state.")
                self.uids_to_eval, self.pending_uids_to_eval = defaultdict(set), defaultdict(set)
                self.model_tracker = ModelTracker()
                if os.path.exists(self.model_tracker_filepath):
                    os.remove(self.model_tracker_filepath)

    def save(self):
        """Saves the current validator state to disk."""
        bt.logging.trace("Saving validator state.")
        os.makedirs(self.base_dir, exist_ok=True)
        with self.pending_uids_to_eval_lock:
            with open(self.uids_filepath, "wb") as f:
                pickle.dump(self.uids_to_eval, f)
                pickle.dump(self.pending_uids_to_eval, f)
        self.model_tracker.save_state(self.model_tracker_filepath)
        self.ema_tracker.save(self.competition_tracker_filepath)

    # ---------------------------
    # Accessors / Mutators
    # ---------------------------
    def get_uids_to_eval(self, competition_id: int) -> typing.Set[int]:
        """Returns a copy of the UIDs currently being evaluated for a competition."""
        with self.pending_uids_to_eval_lock:
            return copy.deepcopy(self.uids_to_eval.get(competition_id, set()))

    def get_pending_uids_to_eval(self, competition_id: int) -> typing.Set[int]:
        """Returns a copy of the UIDs pending evaluation for a competition."""
        with self.pending_uids_to_eval_lock:
            return copy.deepcopy(self.pending_uids_to_eval.get(competition_id, set()))

    def add_pending_uid_to_eval(self, competition_id: int, uid: int):
        """Adds a UID to the pending evaluation queue for a competition."""
        with self.pending_uids_to_eval_lock:
            self.pending_uids_to_eval[competition_id].add(uid)

    def update_uids_to_eval(self, competition_id: int, uids_to_keep: typing.Set[int], active_competitions: typing.Set[int]):
        """Updates the set of UIDs to evaluate, cleaning up inactive competitions."""
        with self.pending_uids_to_eval_lock:
            self.uids_to_eval[competition_id] = uids_to_keep
            comps_to_delete = (set(self.uids_to_eval.keys()) | set(self.pending_uids_to_eval.keys())) - active_competitions
            for comp in comps_to_delete:
                bt.logging.debug(f"Cleaning up UIDs from sunset competition {comp}.")
                self.uids_to_eval.pop(comp, None)
                self.pending_uids_to_eval.pop(comp, None)

    def get_pending_and_current_uid_counts(self) -> typing.Tuple[int, int]:
        """Returns (#pending, #current) across all competitions."""
        with self.pending_uids_to_eval_lock:
            pending = sum(len(s) for s in self.pending_uids_to_eval.values())
            current = sum(len(s) for s in self.uids_to_eval.values())
        return pending, current

    # ---------------------------
    # EMA helpers (encapsulation)
    # ---------------------------
    def update_ema_scores(self, scores_for_ema: typing.Dict[int, float], competition_id: int, block: int, uid_to_hotkey: typing.Dict[int, str]):
        """Updates the EMA scores for a given competition."""
        for uid, score in scores_for_ema.items():
            self.ema_tracker.update(competition_id, uid, score, block, uid_to_hotkey[uid])

    def reset_ema_competitions(self, active_competition_ids: typing.Set[int]):
        """Resets EMA data for any competitions that are no longer active."""
        self.ema_tracker.reset_competitions(active_competition_ids)

    def reset_ema_uid(self, uid: int):
        """Reset EMA for a specific UID across competitions."""
        self.ema_tracker.reset_uid(uid=uid)

    def reset_ema_hotkey_score(self, hotkey: str):
        """Reset EMA score for a specific hotkey."""
        self.ema_tracker.reset_score_for_hotkey(hotkey=hotkey)
