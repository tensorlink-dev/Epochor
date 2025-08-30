"""Manages the lifecycle of miner models, including updating, cleaning, and queuing for evaluation."""

import os
import datetime as dt
import time
import threading
import asyncio
import traceback
import math
import typing
import copy

import bittensor as bt
import constants
from competitions import competitions
from epochor.utils import misc as utils, metagraph_utils
from epochor.model.storage.disk_model_store import DiskModelStore
from epochor.model.model_updater import ModelUpdater, MinerMisconfiguredError
from epochor.model.model_data import EvalResult
from competitions.epsilon import EpsilonFunc
from epochor.utils import competition_utils

from .state import ValidatorState


def should_retry_model(epsilon_func: EpsilonFunc, current_block: int, eval_history: typing.List[EvalResult]) -> bool:
    """Determines whether a model should be re-evaluated based on its past performance."""
    if not eval_history:
        return True

    last_eval_block = eval_history[-1].block
    decayed_score = epsilon_func.compute_epsilon(current_block, last_eval_block)
    # If the model has ever performed better than the current decayed cutoff, don't retry.
    return not any(eval_result.score <= decayed_score for eval_result in eval_history)


class ModelManager:
    """
    A class to manage miner models. It runs two background threads:
    1. update_models: Continuously iterates through miners, syncing their latest models.
    2. clean_models: Periodically removes stale models from local storage.
    """
    def __init__(
        self,
        model_updater: ModelUpdater,
        miner_iterator,
        metagraph: "bt.metagraph",
        state: ValidatorState,
        metagraph_lock: threading.RLock,
        local_store: DiskModelStore,
        get_current_block_fn: typing.Callable[[], int],
    ):
        """Initializes the ModelManager."""
        self.model_updater = model_updater
        self.model_tracker = state.model_tracker
        self.miner_iterator = miner_iterator
        self.metagraph = metagraph
        self.state = state
        self.stop_event = threading.Event()
        self.update_thread = threading.Thread(target=self._update_models_loop, daemon=True)
        self.clean_thread = threading.Thread(target=self._clean_models_loop, daemon=True)
        self.metagraph_lock = metagraph_lock
        self.local_store = local_store
        self.get_current_block = get_current_block_fn

        # internal cadence tracking
        self._last_checked_top_models_time: typing.Optional[dt.datetime] = None
        self._uid_last_checked: dict[int, dt.datetime] = {}

    # ---------------------------
    # Lifecycle
    # ---------------------------
    def start(self):
        """Starts the background threads for updating and cleaning models."""
        self.update_thread.start()
        self.clean_thread.start()

    def stop(self):
        """Stops the background threads."""
        self.stop_event.set()
        self.update_thread.join()
        self.clean_thread.join()

    # ---------------------------
    # Main loops
    # ---------------------------
    def _update_models_loop(self):
        """The main loop for the model update thread."""
        time.sleep(60)
        while not self.stop_event.is_set():
            try:
                # Occasionally ensure “top models” get queued
                if (
                    self._last_checked_top_models_time is None
                    or (dt.datetime.now() - self._last_checked_top_models_time) > constants.scan_top_model_cadence
                ):
                    self._last_checked_top_models_time = dt.datetime.now()
                    self._queue_top_models_for_eval()

                self._wait_for_open_eval_slot()
                self._update_next_miner_model()
            except Exception as e:
                bt.logging.error(f"Error in update loop: {e} \n{traceback.format_exc()}")
        bt.logging.info("Exiting update models loop.")

    def _update_next_miner_model(self):
        """Checks and potentially updates the model for the next miner in the sequence."""
        next_uid = next(self.miner_iterator)

        # Validate UID with current metagraph size
        with self.metagraph_lock:
            n = len(self.metagraph.hotkeys)
        if not isinstance(next_uid, int) or next_uid < 0 or next_uid >= n:
            bt.logging.debug(f"UID {next_uid} is no longer valid, skipping.")
            return

        # Rate-limit sequential checking by chain cadence
        now = dt.datetime.now()
        last = self._uid_last_checked.get(next_uid, dt.datetime.min)
        if now - last < constants.chain_update_cadence:
            sleep_s = (constants.chain_update_cadence - (now - last)).total_seconds()
            bt.logging.trace(f"Checked UID {next_uid} too recently; sleeping {sleep_s:.1f}s")
            time.sleep(max(0.0, sleep_s))
        self._uid_last_checked[next_uid] = dt.datetime.now()

        with self.metagraph_lock:
            hotkey = self.metagraph.hotkeys[next_uid]

        curr_block = self.get_current_block()
        model_metadata = self.model_tracker.get_model_metadata_for_miner_hotkey(hotkey)

        # Decide if we should force re-sync (retry policy)
        force_sync = False
        if model_metadata:
            is_queued = (
                next_uid in self.state.get_uids_to_eval(model_metadata.id.competition_id)
                or next_uid in self.state.get_pending_uids_to_eval(model_metadata.id.competition_id)
            )
            if not is_queued:
                competition = competition_utils.get_competition_for_block(
                    model_metadata.id.competition_id,
                    curr_block,
                    competitions.COMPETITION_SCHEDULE_BY_BLOCK,
                )
                if competition is not None:
                    eval_history = self.model_tracker.get_eval_results_for_miner_hotkey(hotkey, competition.id)
                    if should_retry_model(competition.constraints.epsilon_func, curr_block, eval_history):
                        bt.logging.debug(f"Forcing sync for UID {next_uid} as it's eligible for retry.")
                        force_sync = True

        try:
            updated = asyncio.run(self.model_updater.sync_model(
                uid=next_uid, hotkey=hotkey, curr_block=curr_block,
                schedule=competitions.COMPETITION_SCHEDULE_BY_BLOCK, force=force_sync
            ))
            if updated:
                metadata = self.model_tracker.get_model_metadata_for_miner_hotkey(hotkey)
                if metadata:
                    self.state.add_pending_uid_to_eval(metadata.id.competition_id, next_uid)
                    bt.logging.debug(
                        f"UID {next_uid} has a new model for competition {metadata.id.competition_id}, queued for eval."
                    )
                else:
                    bt.logging.warning(f"Synced model for UID {next_uid}, but no metadata found.")
                # Reset EMA for this UID (encapsulated helper)
                self.state.reset_ema_uid(next_uid)

        except MinerMisconfiguredError as e:
            bt.logging.warning(f"Failed to sync UID {next_uid}: {e}")
            self.model_tracker.on_model_evaluated(hotkey, 0, EvalResult(block=curr_block, score=math.inf))

    def _wait_for_open_eval_slot(self):
        """Pauses the update thread if the evaluation queue is full."""
        pending, current = self.state.get_pending_and_current_uid_counts()
        while pending + current >= constants.updated_models_limit:
            bt.logging.info("Evaluation queue is full. Pausing model updates for 5 minutes.")
            time.sleep(300)
            pending, current = self.state.get_pending_and_current_uid_counts()

    def _clean_models_loop(self):
        """The main loop for the model cleaning thread."""
        time.sleep(dt.timedelta(hours=1).total_seconds())
        while not self.stop_event.is_set():
            try:
                bt.logging.trace("Starting cleanup of stale models.")
                hotkey_to_model_id = {
                    hotkey: metadata.id
                    for hotkey, metadata in self.model_tracker.get_miner_hotkey_to_model_metadata_dict().items()
                }

                # Keep models for UIDs that are pending or in eval
                uids_to_keep = set()
                with self.state.pending_uids_to_eval_lock:
                    for s in self.state.pending_uids_to_eval.values():
                        uids_to_keep.update(s)
                    for s in self.state.uids_to_eval.values():
                        uids_to_keep.update(s)

                hotkeys_to_keep = set()
                with self.metagraph_lock:
                    for uid in uids_to_keep:
                        if isinstance(uid, int) and 0 <= uid < len(self.metagraph.hotkeys):
                            hotkeys_to_keep.add(self.metagraph.hotkeys[uid])

                evaluated_hotkeys_to_model_id = {
                    hk: mid for hk, mid in hotkey_to_model_id.items() if hk in hotkeys_to_keep
                }
                self.local_store.delete_unreferenced_models(
                    valid_models_by_hotkey=evaluated_hotkeys_to_model_id,
                    grace_period_seconds=600,
                )
            except Exception as e:
                bt.logging.error(f"Error in clean loop: {e}")
            time.sleep(dt.timedelta(minutes=5).total_seconds())
        bt.logging.info("Exiting clean models loop.")

    # ---------------------------
    # Top-model shortcut
    # ---------------------------
    def _queue_top_models_for_eval(self) -> None:
        """Ensure highly-weighted miner models across the network get evaluated here, too."""
        with self.metagraph_lock:
            mg_copy = copy.deepcopy(self.metagraph)

        top_miner_uids = metagraph_utils.get_top_miners(
            mg_copy,
            min_validator_stake=constants.WEIGHT_SYNC_VALI_MIN_STAKE,
            min_percent=constants.WEIGHT_SYNC_MINER_MIN_PERCENT,
        )

        all_uids_to_eval = set()
        all_pending_uids_to_eval = set()
        with self.state.pending_uids_to_eval_lock:
            for uids in self.state.uids_to_eval.values():
                all_uids_to_eval.update(uids)
            for uids in self.state.pending_uids_to_eval.values():
                all_pending_uids_to_eval.update(uids)

        uids_to_add = top_miner_uids - all_uids_to_eval - all_pending_uids_to_eval

        curr_block = self.get_current_block()
        for uid in uids_to_add:
            if not isinstance(uid, int) or uid < 0 or uid >= len(mg_copy.hotkeys):
                bt.logging.debug(f"[queue_top_models] Skipping invalid UID {uid}; size={len(mg_copy.hotkeys)}.")
                continue
            hotkey = mg_copy.hotkeys[uid]
            last_eval_block = self.model_tracker.get_block_last_evaluated(hotkey) or 0

            if curr_block - last_eval_block >= constants.model_retry_cadence:
                try:
                    updated = asyncio.run(self.model_updater.sync_model(
                        uid=uid, hotkey=hotkey, curr_block=curr_block,
                        schedule=competitions.COMPETITION_SCHEDULE_BY_BLOCK, force=True
                    ))
                except MinerMisconfiguredError as e:
                    bt.logging.warning(f"Failed to sync top-model UID {uid}: {e}")
                    self.model_tracker.on_model_evaluated(
                        hotkey, 0, EvalResult(block=curr_block, score=math.inf,
                                              winning_model_block=0, winning_model_score=0)
                    )
                    updated = False

                if not updated:
                    continue

                metadata = self.model_tracker.get_model_metadata_for_miner_hotkey(hotkey)
                if metadata is not None:
                    self.state.add_pending_uid_to_eval(metadata.id.competition_id, uid)
                    bt.logging.trace(f"Queued top model UID={uid} for competition {metadata.id.competition_id}.")
                else:
                    bt.logging.warning(f"Synced top-model for UID {uid}, but no metadata found.")
