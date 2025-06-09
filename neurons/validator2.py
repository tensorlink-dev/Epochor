# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const

import os
import pickle
import logging
import typing

import bittensor as bt

from epochor import constants
from epochor import utils
from epochor.ema_tracker import EMATracker
from epochor.model.model_tracker import ModelTracker


class ValidatorState:
    MODEL_TRACKER_FILENAME = "model_tracker.pickle"
    COMPETITION_TRACKER_FILENAME = "competition_tracker.pickle"
    UIDS_FILENAME = "uids.pickle"
    VERSION_FILENAME = "version.txt"

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.uids_filepath = os.path.join(base_dir, ValidatorState.UIDS_FILENAME)
        self.model_tracker_filepath = os.path.join(
            base_dir, ValidatorState.MODEL_TRACKER_FILENAME
        )
        self.competition_tracker_filepath = os.path.join(
            base_dir, ValidatorState.COMPETITION_TRACKER_FILENAME
        )
        self.version_filepath = os.path.join(base_dir, ValidatorState.VERSION_FILENAME)

        self.model_tracker = ModelTracker()
        self.ema_tracker = EMATracker(alpha=constants.alpha)

    def load(self):
        # Check if the version has changed since we last restarted.
        previous_version = utils.get_version(self.version_filepath)
        utils.save_version(self.version_filepath, constants.__spec_version__)

        # If this is an upgrade, blow away state so that everything is re-evaluated.
        if previous_version != constants.__spec_version__:
            logging.info(
                f"Validator updated. Previous version={previous_version}. Current version={constants.__spec_version__}"
            )
            if os.path.exists(self.uids_filepath):
                logging.info(
                    f"Because the validator updated, deleting {self.uids_filepath} so everything is re-evaluated."
                )
                os.remove(self.uids_filepath)
            if os.path.exists(self.model_tracker_filepath):
                logging.info(
                    f"Because the validator updated, deleting {self.model_tracker_filepath} so everything is re-evaluated."
                )
                os.remove(self.model_tracker_filepath)

        # Initialize the model tracker.
        if not os.path.exists(self.model_tracker_filepath):
            logging.warning("No model tracker state file found. Starting from scratch.")
        else:
            try:
                self.model_tracker.load_state(self.model_tracker_filepath)
            except Exception as e:
                logging.warning(
                    f"Failed to load model tracker state. Reason: {e}. Starting from scratch."
                )

        # Initialize the competition tracker.
        if not os.path.exists(self.competition_tracker_filepath):
            logging.warning(
                "No competition tracker state file found. Starting from scratch."
            )
        else:
            try:
                self.ema_tracker.load_state(self.competition_tracker_filepath)
            except Exception as e:
                logging.warning(
                    f"Failed to load competition tracker state. Reason: {e}. Starting from scratch."
                )

        # Initialize the UIDs to eval.
        if not os.path.exists(self.uids_filepath):
            logging.warning("No uids state file found. Starting from scratch.")
            self.uids_to_eval: typing.Dict[int, typing.Set] = {}
            self.pending_uids_to_eval: typing.Dict[int, typing.Set] = {}
        else:
            try:
                with open(self.uids_filepath, "rb") as f:
                    self.uids_to_eval = pickle.load(f)
                    self.pending_uids_to_eval = pickle.load(f)
            except Exception as e:
                logging.warning(
                    f"Failed to load uids to eval state. Reason: {e}. Starting from scratch."
                )
                self.uids_to_eval: typing.Dict[int, typing.Set] = {}
                self.pending_uids_to_eval: typing.Dict[int, typing.Set] = {}
                # We also need to wipe the model tracker state in this case to ensure we re-evaluate all the models.
                self.model_tracker = ModelTracker()
                if os.path.exists(self.model_tracker_filepath):
                    logging.warning(
                        f"Because the uids to eval state failed to load, deleting model tracker state at {self.model_tracker_filepath} so everything is re-evaluated."
                    )
                    os.remove(self.model_tracker_filepath)

    def save(self, uids_to_eval, pending_uids_to_eval):
        logging.trace("Saving validator state.")
        os.makedirs(self.base_dir, exist_ok=True)

        # Save the state of the validator uids to file.
        with open(self.uids_filepath, "wb") as f:
            pickle.dump(uids_to_eval, f)
            pickle.dump(pending_uids_to_eval, f)

        # Save the state of the trackers to file.
        self.model_tracker.save_state(self.model_tracker_filepath)
        self.ema_tracker.save_state(self.competition_tracker_filepath)

import asyncio
import copy
import dataclasses
import datetime as dt
import functools
import json
import logging
import math
import os
import pickle
import random
import threading
import time
import traceback
import typing
from collections import defaultdict
from itertools import cycle
from typing import Any

import bittensor as bt
import torch
import wandb
from retry import retry
from rich.console import Console
from rich.table import Table

# Epochor Imports
from epochor import constants
from epochor import utils
from neurons import config
from epochor.competition import utils as competition_utils 
from epochor.competition.data import Competition, CompetitionId, EpsilonFunc
from epochor.dataloaders import DatasetLoaderFactory
from epochor.ema_tracker import EMATracker
from epochor.evaluation import Evaluator
from epochor.model.data import EvalResult, ScoreDetails
from epochor.rewards import compute_scores
from epochor.model.storage.disk_model_store import DiskModelStore
from epochor.model.storage.hf_model_store import HuggingFaceModelStore
from epochor.model.storage.metadata_model_store import ChainModelMetadataStore
from epochor.model.model_tracker import ModelTracker
from epochor.model.model_updater import ModelUpdater, MinerMisconfiguredError
from epochor.utils import metagraph_utils
from epochor.utils.logging import configure_logging, reinitialize_logging
from taoverse.utilities.perf_monitor import PerfMonitor
from taoverse.metagraph.metagraph_syncer import MetagraphSyncer
from taoverse.metagraph.miner_iterator import MinerIterator
from competitions.data import CompetitionId
import constants

class ModelManager:
    def __init__(self, model_updater, model_tracker, miner_iterator, metagraph):
        self.model_updater = model_updater
        self.model_tracker = model_tracker
        self.miner_iterator = miner_iterator
        self.metagraph = metagraph
        self.stop_event = threading.Event()
        self.update_thread = threading.Thread(target=self.update_models, daemon=True)
        self.clean_thread = threading.Thread(target=self.clean_models, daemon=True)
        self.metagraph_lock = threading.RLock()

    def start(self):
        self.update_thread.start()
        self.clean_thread.start()

    def stop(self):
        self.stop_event.set()
        self.update_thread.join()
        self.clean_thread.join()

    def update_models(self):
        """Updates the models in the local store based on the latest metadata from the chain."""

        # Track how recently we updated each uid from sequential iteration.
        uid_last_checked_sequential = dict()
        # Track how recently we checked the list of top models.
        last_checked_top_models_time = None

        # Delay the first update loop until the metagraph has been synced.
        time.sleep(60)

        # The below loop iterates across all miner uids and checks to see
        # if they should be updated.
        while not self.stop_event.is_set():
            try:
                # At most once per `scan_top_model_cadence`, check which models are being assigned weight by
                # the top validators and ensure they'll be evaluated soon.
                if (
                    not last_checked_top_models_time
                    or dt.datetime.now() - last_checked_top_models_time
                    > constants.scan_top_model_cadence
                ):
                    last_checked_top_models_time = dt.datetime.now()
                    self._queue_top_models_for_eval()

                # Top model check complete. Now continue with the sequential iterator to check for the next miner
                # to update.

                self._wait_for_open_eval_slot()

                # We have space to add more models for eval. Process the next UID.
                next_uid = next(self.miner_iterator)

                # Confirm that we haven't already checked it within the chain update cadence.
                time_diff = (
                    dt.datetime.now() - uid_last_checked_sequential[next_uid]
                    if next_uid in uid_last_checked_sequential
                    else None
                )
                if time_diff and time_diff < constants.chain_update_cadence:
                    # If we have seen it within chain update cadence then sleep until it has been at least that long.
                    time_to_sleep = (
                        constants.chain_update_cadence - time_diff
                    ).total_seconds()
                    logging.trace(
                        f"Update loop has already processed all UIDs in the last {constants.chain_update_cadence}. Sleeping {time_to_sleep} seconds."
                    )
                    time.sleep(time_to_sleep)

                uid_last_checked_sequential[next_uid] = dt.datetime.now()
                curr_block = self._get_current_block()

                # Get their hotkey from the metagraph.
                with self.metagraph_lock:
                    hotkey = self.metagraph.hotkeys[next_uid]

                # Check if we should retry this model and force a sync if necessary.
                force_sync = False
                model_metadata = self.model_tracker.get_model_metadata_for_miner_hotkey(
                    hotkey
                )

                if model_metadata:
                    # Check if the model is already queued for eval.
                    is_queued_for_eval = False
                    #with self.pending_uids_to_eval_lock:
                    #    is_queued_for_eval = (
                    #        next_uid
                    #        in self.pending_uids_to_eval[
                    #            model_metadata.id.competition_id
                    #        ]
                    #        or next_uid
                    #        in self.uids_to_eval[model_metadata.id.competition_id]
                    #    )

                    competition = competition_utils.get_competition_for_block(
                        model_metadata.id.competition_id,
                        curr_block,
                        constants.COMPETITION_SCHEDULE_BY_BLOCK,
                    )
                    if competition is not None and not is_queued_for_eval:
                        eval_history = (
                            self.model_tracker.get_eval_results_for_miner_hotkey(
                                hotkey, competition.id
                            )
                        )
                        force_sync = should_retry_model(
                            competition.constraints.epsilon_func,
                            curr_block,
                            eval_history,
                        )
                        if force_sync:
                            logging.debug(
                                f"Force downloading model for UID {next_uid} because it should be retried. Eval_history={eval_history}"
                            )

                # Compare metadata and tracker, syncing new model from remote store to local if necessary.
                try:
                    updated = asyncio.run(
                        self.model_updater.sync_model(
                            uid=next_uid,
                            hotkey=hotkey,
                            curr_block=curr_block,
                            schedule_by_block=constants.COMPETITION_SCHEDULE_BY_BLOCK,
                            force=force_sync,
                        )
                    )
                except MinerMisconfiguredError as e:
                    self.model_tracker.on_model_evaluated(
                        hotkey,
                        0,  # Technically this is B7 but that is unused.
                        EvalResult(
                            block=curr_block,
                            score=math.inf,
                            # We don't care about the winning model for this check since we just need to log the model eval failure.
                            winning_model_block=0,
                            winning_model_score=0,
                        ),
                    )
                    raise e

                if updated:
                    metadata = self.model_tracker.get_model_metadata_for_miner_hotkey(
                        hotkey
                    )
                    if metadata is not None:
                        #with self.pending_uids_to_eval_lock:
                        #    self.pending_uids_to_eval[metadata.id.competition_id].add(
                        #        next_uid
                        #    )
                        #    logging.debug(
                        #        f"Found a new model for UID={next_uid} for competition {metadata.id.competition_id}. It will be evaluated on the next loop."
                        #    )
                        pass
                    else:
                        logging.warning(
                            f"Failed to find metadata for uid {next_uid} with hotkey {hotkey}"
                        )
                    # inside the `if updated:` block, after you know this `uid` got a fresh model
                    #self.ema_tracker.reset_uid(
                    #    competition_id = metadata.id.competition_id,
                    #    uid = next_uid
                    #)
                    pass

            except InvalidStatus as e:
                logging.info(
                    f"Websocket exception in update loop: {e}. Waiting 3 minutes."
                )
                time.sleep(180)
            except (RepositoryNotFoundError, RevisionNotFoundError) as e:
                logging.trace(e)
            except MinerMisconfiguredError as e:
                logging.trace(e)
            except Exception as e:
                logging.error(f"Error in update loop: {e} 
 {traceback.format_exc()}")

        logging.info("Exiting update models loop.")

    def _wait_for_open_eval_slot(self) -> None:
        """Waits until there is at least one slot open to download and evaluate a model."""
        #pending_uid_count, current_uid_count = self.get_pending_and_current_uid_counts()
        pending_uid_count = 0
        current_uid_count = 0

        while pending_uid_count + current_uid_count >= 10:#self.config.updated_models_limit:
            # Wait 5 minutes for the eval loop to process them.
            logging.info(
                f"Update loop: There are already {pending_uid_count + current_uid_count} synced models pending eval. Checking again in 5 minutes."
            )
            time.sleep(300)
            # Check to see if the pending uids have been cleared yet.
            #pending_uid_count, current_uid_count = (
            #    self.get_pending_and_current_uid_counts()
            #)
            pass

    def _queue_top_models_for_eval(self) -> None:
        # Take a deep copy of the metagraph for use in the top uid retry check.
        # The regular loop below will use self.metagraph which may be updated as we go.
        with self.metagraph_lock:
            metagraph = copy.deepcopy(self.metagraph)

        # Find any miner UIDs which top valis are assigning weight and aren't currently scheduled for an eval.
        # This is competition agnostic, as anything with weight is 'winning' a competition for some vali.
        top_miner_uids = metagraph_utils.get_top_miners(
            metagraph,
            constants.WEIGHT_SYNC_VALI_MIN_STAKE,
            constants.WEIGHT_SYNC_MINER_MIN_PERCENT,
        )
        
        #with self.pending_uids_to_eval_lock:
        #    all_uids_to_eval = set()
        #    all_pending_uids_to_eval = set()
        #    # Loop through the uids across all competitions.
        #    for uids in self.uids_to_eval.values():
        #        all_uids_to_eval.update(uids)
        #    for uids in self.pending_uids_to_eval.values():
        #        all_pending_uids_to_eval.update(uids)

        #    # Reduce down to top models that are not in any competition yet.
        #    uids_to_add = top_miner_uids - all_uids_to_eval - all_pending_uids_to_eval
        uids_to_add = []

        for uid in uids_to_add:
            # Check when we last evaluated this model.
            hotkey = metagraph.hotkeys[uid]
            last_eval_block = self.model_tracker.get_block_last_evaluated(hotkey) or 0
            curr_block = self._get_current_block()
            if curr_block - last_eval_block >= constants.model_retry_cadence:
                try:
                    # It's been long enough - redownload this model and schedule it for eval.
                    # This still respects the eval block delay so that previously top uids can't bypass it.
                    try:
                        should_retry = asyncio.run(
                            self.model_updater.sync_model(
                                uid=uid,
                                hotkey=hotkey,
                                curr_block=curr_block,
                                schedule_by_block=constants.COMPETITION_SCHEDULE_BY_BLOCK,
                                force=True,
                            )
                        )
                    except MinerMisconfiguredError as e:
                        self.model_tracker.on_model_evaluated(
                            hotkey,
                            0,  # Technically this is B7 but that is unused.
                            EvalResult(
                                block=curr_block,
                                score=math.inf,
                                # We don't care about the winning model for this check since we just need to log the model eval failure.
                                winning_model_block=0,
                                winning_model_score=0,
                            ),
                        )
                        raise e

                    if not should_retry:
                        continue

                    # Since this is a top model (as determined by other valis),
                    # we don't worry if self.pending_uids is already "full". At most
                    # there can be 10 * comps top models that we'd add here and that would be
                    # a wildy exceptional case. It would require every vali to have a
                    # different top model.
                    # Validators should only have ~1 winner per competition and we only check bigger valis
                    # so there should not be many simultaneous top models not already being evaluated.
                    top_model_metadata = (
                        self.model_tracker.get_model_metadata_for_miner_hotkey(hotkey)
                    )
                    if top_model_metadata is not None:
                        logging.trace(
                            f"Shortcutting to top model or retrying evaluation for previously discarded top model with incentive for UID={uid}"
                        )
                        #with self.pending_uids_to_eval_lock:
                        #    self.pending_uids_to_eval[
                        #        top_model_metadata.id.competition_id
                        #    ].add(uid)
                        pass
                    else:
                        logging.warning(
                            f"Failed to find metadata for uid {uid} with hotkey {hotkey}"
                        )

                except Exception:
                    logging.debug(
                        f"Failure in update loop for UID={uid} during top model check. {traceback.format_exc()}"
                    )

    def clean_models(self):
        """Cleans up models that are no longer referenced."""

        # Delay the clean-up thread until the update loop has had time to run one full pass after an upgrade.
        # This helps prevent unnecessarily deleting a model which is on disk, but hasn't yet been re-added to the
        # model tracker by the update loop.
        time.sleep(dt.timedelta(hours=1).total_seconds())

        # The below loop checks to clear out all models in local storage that are no longer referenced.
        while not self.stop_event.is_set():
            try:
                logging.trace("Starting cleanup of stale models.")

                # Get a mapping of all hotkeys to model ids.
                hotkey_to_model_metadata = (
                    self.model_tracker.get_miner_hotkey_to_model_metadata_dict()
                )
                hotkey_to_model_id = {
                    hotkey: metadata.id
                    for hotkey, metadata in hotkey_to_model_metadata.items()
                }

                # Find all hotkeys that are currently being evaluated or pending eval.
                #uids_to_keep = set()
                #with self.pending_uids_to_eval_lock:
                #    for pending_uids in self.pending_uids_to_eval.values():
                #        uids_to_keep.update(pending_uids)
                #    for eval_uids in self.uids_to_eval.values():
                #        uids_to_keep.update(eval_uids)
                uids_to_keep = []

                hotkeys_to_keep = set()
                with self.metagraph_lock:
                    for uid in uids_to_keep:
                        hotkeys_to_keep.add(self.metagraph.hotkeys[uid])

                # Only keep those hotkeys.
                evaluated_hotkeys_to_model_id = {
                    hotkey: model_id
                    for hotkey, model_id in hotkey_to_model_id.items()
                    if hotkey in hotkeys_to_keep
                }

                #self.local_store.delete_unreferenced_models(
                #    valid_models_by_hotkey=evaluated_hotkeys_to_model_id,
                #    grace_period_seconds=600,
                #)
                pass
            except Exception as e:
                logging.error(f"Error in clean loop: {e}")

            # Only check every 5 minutes.
            time.sleep(dt.timedelta(minutes=5).total_seconds())

        logging.info("Exiting clean models loop.")

    def _get_current_block(self) -> int:
        """Returns the current block."""

        @retry(tries=5, delay=1, backoff=2)
        def _get_block_with_retry():
            #return self.subtensor.block
            return 1

        try:
            return _get_block_with_retry()
        except:
            logging.debug(
                "Failed to get the latest block from the chain. Using the block from the cached metagraph."
            )
            # Network call failed. Fallback to using the block from the metagraph,
            # even though it'll be a little stale.
            with self.metagraph_lock:
                return self.metagraph.block.item()


class WeightSetter:
    def __init__(self, subtensor, wallet, netuid, weights):
        self.subtensor = subtensor
        self.wallet = wallet
        self.netuid = netuid
        self.weights = weights
        self.stop_event = threading.Event()
        self.weight_thread = threading.Thread(target=self.set_weights, daemon=True)
        self.metagraph_lock = threading.RLock()

    def start(self):
        self.weight_thread.start()

    def stop(self):
        self.stop_event.set()
        self.weight_thread.join()

    def set_weights(self):
        """Set weights on the chain regularly."""

        # Check that we have some weights internally for startup situations.
        all_zero_weights = True
        while all_zero_weights is True:
            # Technically returns a tensor but it evaluates to true.
            with self.metagraph_lock:
                all_zero_weights = torch.all(self.weights == 0)
            logging.trace(
                "Waiting 60 seconds for internal weights before continuing to try set weights."
            )
            time.sleep(60)

        while not self.stop_event.is_set():
            try:
                set_weights_success = False
                while not set_weights_success:
                    set_weights_success, _ = asyncio.run(self.try_set_weights(ttl=60))
                    # Wait for 120 seconds before we try to set weights again.
                    if set_weights_success:
                        logging.info("Successfully set weights.")
                    else:
                        time.sleep(120)
            except Exception as e:
                logging.error(f"Error in set weights: {e}")

            # Only set weights once every hour
            time.sleep(60 * 60)

        logging.info("Exiting set weights loop.")

    async def try_set_weights(self, ttl: int) -> typing.Tuple[bool, str]:
        """Sets the weights on the chain with ttl, without raising exceptions if it times out."""

        async def _try_set_weights() -> typing.Tuple[bool, str]:
            with self.metagraph_lock:
                uids = [1] #self.metagraph.uids
            try:
                #with self.weight_lock:
                self.weights.nan_to_num(0.0)
                weights_to_set = self.weights

                return self.subtensor.set_weights(
                    netuid=self.netuid,
                    wallet=self.wallet,
                    uids=uids,
                    weights=weights_to_set.numpy(),
                    wait_for_inclusion=True,
                    version_key=constants.weights_version_key,
                    max_retries=1,
                )
            except Exception as e:
                logging.warning(
                    f"Failed to set weights due to {e}. Trying again later."
                )
                return (False, str(e))

        try:
            logging.debug(f"Setting weights.")
            status = await asyncio.wait_for(_try_set_weights(), ttl)
            logging.debug(f"Finished setting weights with status: {status}.")
            return status
        except asyncio.TimeoutError:
            logging.error(f"Failed to set weights after {ttl} seconds")
            return (False, f"Timeout after {ttl} seconds")


def should_retry_model(
    epsilon_func: EpsilonFunc,
    current_block: int,
    eval_history: typing.List[EvalResult],
) -> bool:
    """Determines whether the model should be retried based on its eval history.

    Args:
        epsilon_func (EpsilonFunc):
            The epsilon function used to determine the score threshold.
        current_block (int):
            The current block.
        eval_history (typing.List[EvalResult]):
            The model's evaluation history.

    Returns:
        bool: True if the model should be retried, False otherwise.
    """
    if not eval_history:
        return True

    # Should retry the model if it was last evaluated more than X blocks ago
    # and if it has a reasonable chance of still being competitive.
    last_eval_block = eval_history[-1].block
    decayed_score = epsilon_func.compute_epsilon(current_block, last_eval_block)
    score_cutoff = decayed_score

    # Check to see if the model ever beat the score cutoff.
    for eval_result in eval_history:
        if eval_result.score <= score_cutoff:
            # It was already competitive, don't retry it.
            return False

    # It has not been competitive, retry it.
    return True