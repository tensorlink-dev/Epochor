# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const

import os
import time
import asyncio
import threading
import traceback
import typing
import copy
import dataclasses
from collections import defaultdict
import datetime as dt
import logging
import random
import functools
import math
from rich.console import Console
from rich.table import Table
import json

import bittensor as bt
import torch
import wandb
from retry import retry

from template.base.validator import BaseValidatorNeuron
from epochor.model.model_constraints import Competition
from competitions import CompetitionId
from epochor.utils.competition_utils import get_competition_schedule_for_block
from neurons import config
from neurons.validator_components import ValidatorState, ModelManager, WeightSetter, should_retry_model
from epochor.model.model_updater import ModelUpdater, MinerMisconfiguredError
from epochor.model.storage.disk_model_store import DiskModelStore
from epochor.model.storage.hf_model_store import HuggingFaceModelStore
from epochor.model.storage.metadata_model_store import ChainModelMetadataStore
from epochor.utils import metagraph_utils
from epochor.utils.miner_iterator import MinerIterator
from epochor.model.model_constraints import MODEL_CONSTRAINTS_BY_COMPETITION_ID
from taoverse.utilities.perf_monitor import PerfMonitor
from taoverse.metagraph.metagraph_syncer import MetagraphSyncer

import constants
from competitions import competitions 

# Epochor Imports
from epochor.datasets.dataloaders import DatasetLoaderFactory
from epochor.model.model_data import EvalResult
from epochor import utils
from epochor.model import model_utils
from epochor.validation.validation import  score_time_series_model, ScoreDetails, compute_scores
from torch.utils.data import Dataset, DataLoader

from epochor.utils.logging import configure_logging,reinitialize_logging
from epochor.validation.ema_tracker import CompetitionEMATracker

from bittensor.utils.btlogging.defines import BITTENSOR_LOGGER_NAME
from bittensor.utils.btlogging.helpers import all_loggers

import numpy as np # Import numpy


@dataclasses.dataclass
class PerUIDEvalState:
    """State tracked per UID in the eval loop"""
    block: int = math.inf
    hotkey: str = "Unknown"
    repo_name: str = "Unknown"
    score: float = math.inf
    score_details: typing.Dict[str, ScoreDetails] = dataclasses.field(default_factory=dict)

class Validator:
    def _configure_logging(self, config: bt.config) -> None:
        #BT logging is noisy, so set it to only log errors.
        bt.logging.set_warning()

        #Setting logging level on bittensor messes with all loggers, which we don't want, so set explicitly to warning here.
        for logger in all_loggers():
            if not logger.name.startswith(BITTENSOR_LOGGER_NAME):
                logger.setLevel(logging.WARNING)

    # Configure the Taoverse logger, which is our primary logger.
        reinitialize_logging()
        configure_logging(config)
        
    def __init__(self):
        self.config = config.validator_config() 
        self._configure_logging(self.config)

        # === Bittensor objects ====
        try:
            self.wallet = bt.wallet(config=self.config) if self.config.wallet.name and self.config.wallet.hotkey else None
        except Exception:
            self.wallet = None
        self.subtensor = bt.subtensor(config=self.config)
        self.weights_subtensor = bt.subtensor(config=self.config)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph_lock = threading.RLock()

        # Setup metagraph syncer
        syncer_subtensor = bt.subtensor(config=self.config)
        self.subnet_metagraph_syncer = MetagraphSyncer(
            syncer_subtensor,
            config={self.config.netuid: dt.timedelta(minutes=20).total_seconds()},
            lite=False,
        )
        self.subnet_metagraph_syncer.do_initial_sync()
        self.metagraph: bt.metagraph = self.subnet_metagraph_syncer.get_metagraph(self.config.netuid)
        self.subnet_metagraph_syncer.register_listener(
            self._on_subnet_metagraph_updated, netuids=[self.config.netuid]
        )
        self.subnet_metagraph_syncer.start()
        self.known_hotkeys = set(self.metagraph.hotkeys)


        torch.backends.cudnn.benchmark = True

        if not self.config.offline:
            self.uid = metagraph_utils.assert_registered(self.wallet, self.metagraph)

        self.run_step_count = 0
        if not self.config.offline and self.config.wandb.on:
            self._new_wandb_run()

        # === Running args ===
        # inside __init__
        S_tensor = torch.from_numpy(self.metagraph.S)                    # now a Tensor
        self.weights = torch.zeros_like(S_tensor, dtype=torch.float32)   # zeros of same shape
        self.global_step = 0
        self.miner_iterator = MinerIterator(self.metagraph.uids.tolist())

        # Setup Stores
        self.metadata_store = ChainModelMetadataStore(subtensor=self.subtensor, subnet_uid=self.config.netuid, wallet=self.wallet)
        self.remote_store = HuggingFaceModelStore()
        self.local_store = DiskModelStore(base_dir=self.config.model_dir)

        # Setup State
        state_dir = os.path.join(self.config.model_dir, "vali-state")
        self.state = ValidatorState(metagraph = self.metagraph, base_dir=state_dir, metagraph_lock=self.metagraph_lock)
        self.state.load()

        # Setup Model Updater & Manager
        self.model_updater = ModelUpdater(
            metadata_store=self.metadata_store,
            remote_store=self.remote_store,
            local_store=self.local_store,
            model_tracker=self.state.model_tracker,
        )
        self.model_manager = ModelManager(
            model_updater=self.model_updater,
            model_tracker=self.state.model_tracker,
            miner_iterator=self.miner_iterator,
            metagraph=self.metagraph,
            state=self.state,
            metagraph_lock=self.metagraph_lock,
            local_store=self.local_store,
            get_current_block_fn=self._get_current_block,
        )

        # Initial weights from loaded state

        competition_schedule = get_competition_schedule_for_block(
            block=self._get_current_block(),
            schedule_by_block=competitions.COMPETITION_SCHEDULE_BY_BLOCK,
        )
        self.weights = self.state.ema_tracker.subnet_weights(competition_schedule)

        # Setup WeightSetter
        self.weight_setter = WeightSetter(
            subtensor=self.weights_subtensor,
            wallet=self.wallet,
            netuid=self.config.netuid,
            metagraph=self.metagraph,
            weights=self.weights,
            metagraph_lock=self.metagraph_lock,
        )

    def run_in_background_thread(self):
        self.model_manager.start()
        if not self.config.offline and not self.config.dont_set_weights:
            self.weight_setter.start()

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.model_manager.stop()
        if not self.config.offline and not self.config.dont_set_weights:
            self.weight_setter.stop()

    async def run_step(self):
        cur_block = self._get_current_block()
        logging.info(f"Current block: {cur_block}")
        
        competition_schedule = get_competition_schedule_for_block(
            block=cur_block,
            schedule_by_block=competitions.COMPETITION_SCHEDULE_BY_BLOCK,
        )
        competition = competition_schedule[self.global_step % len(competition_schedule)]
        logging.info(f"Starting evaluation for competition: {competition.id}")

        with self.state.pending_uids_to_eval_lock:
            self.state.uids_to_eval[competition.id].update(self.state.pending_uids_to_eval[competition.id])
            self.state.pending_uids_to_eval[competition.id].clear()
        
        uids = list(self.state.uids_to_eval.get(competition.id, set()))

        if not uids:
            logging.debug(f"No uids to eval for competition {competition.id}. Checking again in 5 minutes.")
            time.sleep(300)
            return

        uid_to_state = defaultdict(PerUIDEvalState)
        seed = self._get_seed()
        eval_tasks, data_loaders, samples = [], [], []

        load_data_perf = PerfMonitor("Eval: Load data")
        with load_data_perf.sample():
            for eval_task in competition.eval_tasks:
                try:
                    data_loader = DatasetLoaderFactory.get_loader(
                        dataset_id=eval_task.dataset_id,
                        dataset_kwargs=eval_task.dataset_kwargs,
                        seed=seed,
                       # sequence_length=competition.constraints.sequence_length,
                    )
                    batches = list(data_loader)
                    if batches:
                        random.Random(seed).shuffle(batches)
                        eval_tasks.append(eval_task)
                        data_loaders.append(data_loader)
                        samples.append(batches)
                except Exception as e:
                    logging.error(f"Error loading data for task {eval_task.name}: {e}")
        
        if not samples:
            logging.warning(f"No evaluation data loaded for competition {competition.id}. Skipping step.")
            return

        #flat_samples = [batch for task_batches in samples for batch in task_batches]

        logging.debug(f"Competition {competition.id} | Computing scores on {uids}")
        load_model_perf = PerfMonitor("Eval: Load model")
        compute_loss_perf = PerfMonitor("Eval: Compute loss")

        for uid_i in uids:
            score, score_details = math.inf, {task.name: ScoreDetails() for task in eval_tasks}
            with self.metagraph_lock:
                hotkey = self.metagraph.hotkeys[uid_i]
            uid_to_state[uid_i].hotkey = hotkey
            model_i_metadata = self.state.model_tracker.get_model_metadata_for_miner_hotkey(hotkey)
           # model_i_fingerprint = self.state.model_tracker.get_model_fingerprint_for_miner_hotkey(hotkey)

            if model_i_metadata and model_i_metadata.id.competition_id == competition.id:
                try:
                    uid_to_state[uid_i].block = model_i_metadata.block
                    uid_to_state[uid_i].repo_name = model_utils.get_hf_repo_name(model_i_metadata)
                    
                    with load_model_perf.sample():
                        model_i = self.local_store.retrieve_model(hotkey, 
                            model_i_metadata.id,
                            model_constraints=MODEL_CONSTRAINTS_BY_COMPETITION_ID[competition.id])

                    with compute_loss_perf.sample():
                        score, score_details = utils.misc.run_in_subprocess(
                            functools.partial(
                                score_time_series_model, 
                                model_i.model,  
                                samples, 
                                eval_tasks, 
                                self.config.device, 
                                seed
                            ),
                            ttl=550, mode="spawn"
                        )
                    del model_i
                except Exception:
                    logging.error(f"Error in eval loop for UID {uid_i}: {traceback.format_exc()}")
            
            uid_to_state[uid_i].score = score
            uid_to_state[uid_i].score_details = score_details 
           # fingerprints[uid] = model_i_fingerprint

        # Initialize scoring_metrics to an empty dictionary
        scorings_metrics = {}
        try:
            # Pass raw_losses for each UID to compute_scores
            uid_to_raw_losses = {uid: state.score_details.get("flat_evaluation", ScoreDetails()).raw_score for uid, state in uid_to_state.items()}
            # Filter out None values or empty arrays if any
            uid_to_raw_losses = {uid: losses for uid, losses in uid_to_raw_losses.items() if losses is not None and len(losses) > 0}

            scorings_metrics = self._compute_and_set_competition_weights(cur_block, uids, uid_to_state, competition, uid_to_raw_losses) #fingerprints
        except Exception as e:
            logging.error(f"Error computing and setting competition weights: {e}{traceback.format_exc()}")
            # If an error occurs, scorings_metrics remains an empty dictionary, 
            # preventing NameError in log_step.

        win_rate = scorings_metrics.get('win_rate_dict', {})
        active_competition_ids = {comp.id for comp in competition_schedule}
        self.state.ema_tracker.reset_competitions(active_competition_ids)
        self.weights = self.state.ema_tracker.subnet_weights(competition_schedule)

        tracker_competition_weights = self.state.ema_tracker.get_competition_weights(competition.id)
        model_prioritization = {
            uid: (1 + tracker_competition_weights[uid].item()) if uid < len(tracker_competition_weights) and tracker_competition_weights[uid].item() >= 0.001 else win_rate.get(uid, 0.0)
            for uid in uids
        }
        models_to_keep = set(sorted(model_prioritization, key=model_prioritization.get, reverse=True)[: self.config.sample_min])
        
        if len(models_to_keep) < self.config.sample_min:
            uid_to_average_score = {uid: state.score for uid, state in uid_to_state.items()}
            for uid in sorted(uid_to_average_score, key=uid_to_average_score.get):
                if len(models_to_keep) >= self.config.sample_min:
                    break
                models_to_keep.add(uid)
        
        self._update_uids_to_eval(competition.id, models_to_keep, active_competition_ids)
        self.state.save()

        self.log_step(competition=competition, uids=uids, uid_to_state=uid_to_state, scoring_metrics=scorings_metrics)
        self.global_step += 1

    def _get_current_block(self) -> int:
        @retry(tries=5, delay=1, backoff=2)
        def _get_block_with_retry():
            return self.subtensor.block
        try:
            return _get_block_with_retry()
        except Exception:
            with self.metagraph_lock:
                return self.metagraph.block.item()

    def _on_subnet_metagraph_updated(self, metagraph: bt.metagraph, netuid: int):
        if netuid != self.config.netuid: return
        with self.metagraph_lock:
            # Check if the number of neurons has changed.
            if len(self.metagraph.uids) != len(metagraph.uids):
                # Re-initialize the ema_tracker with the new number of neurons.
                self.state.ema_tracker = CompetitionEMATracker(num_neurons=len(metagraph.uids))
                logging.info("Re-initialized CompetitionEMATracker due to a change in the number of neurons.")

            new_hotkeys = set(metagraph.hotkeys)
            added = new_hotkeys - self.known_hotkeys
            if added:
                logging.info(f"New hotkeys registered: {added}")
                for hk in added:
                    self.state.reset_ema_hotkey_score(hotkey=hk)
            self.known_hotkeys = new_hotkeys
            self.metagraph = copy.deepcopy(metagraph)
            self.miner_iterator.set_miner_uids(metagraph.uids.tolist())
            self.state.model_tracker.on_hotkeys_updated(new_hotkeys)

    def _new_wandb_run(self):
        run_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = f"validator-{self.uid}-{run_id}"
        self.wandb_run = wandb.init(
            name=name, project=self.config.wandb_project, entity="macrocosmos",
            config={"uid": self.uid, "hotkey": self.wallet.hotkey.ss58_address, "run_name": run_id, "version": constants.__version__, "type": "validator"},
            allow_val_change=True
        )

    def _get_seed(self):
        try:
            @retry(tries=3, delay=1, backoff=2)
            def _get_seed_with_retry():
                return metagraph_utils.get_hash_of_sync_block(self.subtensor, constants.SYNC_BLOCK_CADENCE)
            return _get_seed_with_retry()
        except Exception as e:
            logging.info(f"Failed to get hash of sync block, using fallback seed: {e}")
            return random.randint(0, 2**32 - 1)

    def _compute_and_set_competition_weights(self, cur_block, uids, uid_to_state, competition, uid_to_raw_losses):
        # uid_to_score = {uid: state.score for uid, state in uid_to_state.items()}
        uid_to_block = {uid: state.block for uid, state in uid_to_state.items()}
       # uid_to_time = {uid: cur_block - state.block for uid, state in uid_to_state.items()}
       # zip(uid_to_time,fingerprints) 
       # time_lower_tri = [uid_to_time .. ]
        # arch_lower_tri = [compare_arch(i,j)]

        scorings_metrics = compute_scores(uids, uid_to_raw_losses) #, competition.constraints.epsilon_func, cur_block)
        # final_score = 'apply_copy_penalty'(logging_metrics[uid]["final_scores_dict"], time_lower_tri, arch_lower_tri, P=0.1)

        scores_for_ema = {uid: scorings_metrics["final_scores_dict"][uid] for uid in uids}

        # Get uid_to_hotkey mapping
        uid_to_hotkey = {uid: self.metagraph.hotkeys[uid] for uid in uids}

        self.state.update_ema_scores(scores_for_ema, competition.id, cur_block, uid_to_hotkey)
        scores = self.state.get_ema_scores(competition.id)
        
        top_uid = max(scores, key=scores.get, default=None)
        if top_uid is not None:
            self._record_eval_results(top_uid, cur_block, uid_to_state, competition.id)

        model_weights = torch.tensor([scores.get(uid, 0) for uid in range(self.metagraph.n)], dtype=torch.float32)
        step_weights = torch.softmax(model_weights / constants.temperature, dim=0)
        
        self.state.ema_tracker.record_competition_weights(competition.id, step_weights)
        
        return scorings_metrics

    def _record_eval_results(self, top_uid, curr_block, uid_to_state, competition_id):
        top_model_score = uid_to_state[top_uid].score
        for _, state in uid_to_state.items():
            eval_result = EvalResult(
                block=curr_block, score=state.score, 
                winning_model_block=uid_to_state[top_uid].block, winning_model_score=top_model_score
            )
            self.state.model_tracker.on_model_evaluated(state.hotkey, competition_id, eval_result)

    def _update_uids_to_eval(self, competition_id, uids, active_competitions):
        self.state.update_uids_to_eval(competition_id, uids, active_competitions)

    def log_step(self, competition: Competition, uids: list, uid_to_state: dict, scoring_metrics: dict):
        step_log = {"timestamp": time.time(), "competition_id": competition.id, "uids": uids, "uid_data": {}}
        
        final_scores_dict = scoring_metrics.get("final_scores_dict", {})
        win_rate_dict = scoring_metrics.get("win_rate_dict", {})
        gap_score_dict = scoring_metrics.get("gap_score_dict", {})
        raw_loss_dict = scoring_metrics.get("raw_loss_dict", {})

        sub_comp_weights = self.state.ema_tracker.get_competition_weights(competition.id)
        
        for uid in uids:
            uid_data = {
                "uid": uid,
                "block": uid_to_state[uid].block,
                "hf": uid_to_state[uid].repo_name,
                "raw_score": uid_to_state[uid].score,
                "ema_score": final_scores_dict.get(uid, math.inf),
                "win_rate": win_rate_dict.get(uid, 0.0),
                "gap_score": gap_score_dict.get(uid, math.inf),
                "raw_loss": raw_loss_dict.get(uid, math.inf),
                "weight": self.weights[uid].item(),
                "norm_weight": sub_comp_weights[uid].item()
            }
            step_log["uid_data"][str(uid)] = uid_data

        console = Console()
        table = Table(title="Step", expand=True)
        
        columns = ["uid", "hf", "raw_score", "ema_score", "win_rate", "gap_score", "raw_loss", "weight", "norm_weight", "block"]
        for col in columns:
            table.add_column(col, justify="center")

        for uid in uids:
            uid_data = step_log["uid_data"][str(uid)]
            row_values = []
            for col in columns:
                value = uid_data.get(col, 0)
                if isinstance(value, float):
                    row_values.append(str(round(value, 4)))
                else:
                    row_values.append(str(value))
            table.add_row(*row_values)
        console.print(table)
        
        if self.config.wandb.on and not self.config.offline:
            self.wandb_run.log(step_log)
            self.run_step_count += 1

    def _get_uids_to_competition_ids(self):
        hotkey_to_metadata = self.state.model_tracker.get_miner_hotkey_to_model_metadata_dict()
        with self.metagraph_lock:
            uids_to_competition_ids = {
                uid: (hotkey_to_metadata.get(self.metagraph.hotkeys[uid]).id.competition_id if self.metagraph.hotkeys[uid] in hotkey_to_metadata else None)
                for uid in range(len(self.metagraph.uids))
            }
        return uids_to_competition_ids

    async def try_run_step(self, ttl: int):
        async def _try_run_step():
            await self.run_step()
        try:
            await asyncio.wait_for(_try_run_step(), ttl)
        except asyncio.TimeoutError:
            logging.error(f"Failed to run step after {ttl} seconds")

    async def run(self):
        while True:
            try:
                await self.try_run_step(ttl=120 * 60)
            except KeyboardInterrupt:
                logging.info("Gracefully shutting down...")
                if self.wandb_run:
                    self.wandb_run.finish()
                exit()
            except Exception as e:
                logging.error(f"Error in validator loop: {e}{traceback.format_exc()}")


if __name__ == "__main__":
    with Validator() as validator:
        asyncio.run(validator.run())