# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()

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
from epochor.config import validator_config
from epochor.competition import utils as competition_utils
from epochor.competition.data import Competition, CompetitionId, EpsilonFunc
from epochor.dataloaders import DatasetLoaderFactory, SubsetLoader
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
from epochor.utils.perf_monitor import PerfMonitor

os.environ["TOKENIZERS_PARALLELISM"] = "true"

@dataclasses.dataclass
class PerUIDEvalState:
    """State tracked per UID in the eval loop."""
    block: int = 0
    hotkey: str = "Unknown"
    repo_name: str = "Unknown"
    score: float = math.inf
    score_details: typing.Dict[str, Any] = dataclasses.field(default_factory=dict)

class Validator:
    MODEL_TRACKER_FILENAME = "model_tracker.pickle"
    COMPETITION_TRACKER_FILENAME = "competition_tracker.pickle" # Kept for loading/saving ema_tracker
    UIDS_FILENAME = "uids.pickle"
    VERSION_FILENAME = "version.txt"

    def __init__(self):
        self.config = validator_config()
        self._configure_logging(self.config)
        logging.info(f"Starting validator with config: {self.config}")

        # Bittensor setup
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.metagraph = self.subtensor.metagraph(self.config.netuid, sync=False)
        self.metagraph.sync(subtensor=self.subtensor)

        torch.backends.cudnn.benchmark = True
        if not self.config.offline:
            self.uid = metagraph_utils.assert_registered(self.wallet, self.metagraph)

        self.run_step_count = 0
        if not self.config.offline and self.config.wandb.on:
            self._new_wandb_run()

        self.weight_lock = threading.RLock()
        self.weights = torch.zeros_like(self.metagraph.S, dtype=torch.float32)

        # Trackers
        self.model_tracker = ModelTracker()
        self.ema_tracker = EMATracker(alpha=constants.alpha) # Using EMATracker for scores and weights

        state_dir = self._state_dir()
        os.makedirs(state_dir, exist_ok=True)
        self.uids_filepath = os.path.join(state_dir, self.UIDS_FILENAME)
        self.model_tracker_filepath = os.path.join(state_dir, self.MODEL_TRACKER_FILENAME)
        self.competition_tracker_filepath = os.path.join(state_dir, self.COMPETITION_TRACKER_FILENAME)
        self.version_filepath = os.path.join(state_dir, self.VERSION_FILENAME)

        prev_version = utils.get_version(self.version_filepath)
        utils.save_version(self.version_filepath, constants.__spec_version__)

        if prev_version != constants.__spec_version__:
            logging.info("Validator version updated, clearing state.")
            for f in [self.uids_filepath, self.model_tracker_filepath, self.competition_tracker_filepath]:
                if os.path.exists(f):
                    os.remove(f)

        if os.path.exists(self.model_tracker_filepath):
            try:
                self.model_tracker.load_state(self.model_tracker_filepath)
            except Exception as e:
                logging.error(f"Failed to load model tracker state: {e}")

        if os.path.exists(self.competition_tracker_filepath):
            try:
                self.ema_tracker.load_state(self.competition_tracker_filepath)
            except Exception as e:
                logging.error(f"Failed to load competition tracker (EMA) state: {e}")

        self.uids_to_eval = defaultdict(set)
        self.pending_uids_to_eval = defaultdict(set)
        if os.path.exists(self.uids_filepath):
            with open(self.uids_filepath, "rb") as f:
                self.uids_to_eval = pickle.load(f)
                self.pending_uids_to_eval = pickle.load(f)

        # Per user request, leaving MinerIterator as is.
        # Note: bt.MinerIterator does not exist. This will raise an AttributeError.
        # Recommended fix: self.miner_iterator = itertools.cycle(range(self.metagraph.n.item()))
        self.miner_iterator = bt.MinerIterator(self.metagraph.uids.tolist())


        self.metadata_store = ChainModelMetadataStore(
            subtensor=self.subtensor, subnet_uid=self.config.netuid, wallet=self.wallet
        )
        self.remote_store = HuggingFaceModelStore()
        self.local_store = DiskModelStore(base_dir=self.config.model_dir)
        self.model_updater = ModelUpdater(
            metadata_store=self.metadata_store,
            remote_store=self.remote_store,
            local_store=self.local_store,
            model_tracker=self.model_tracker,
        )

        self.evaluator = Evaluator(device=self.config.device)
        self.metagraph_lock = threading.RLock()
        self.global_step = 0

        self.stop_event = threading.Event()
        threading.Thread(target=self.update_models, daemon=True).start()
        threading.Thread(target=self.clean_models, daemon=True).start()
        if not self.config.offline and not self.config.dont_set_weights:
            threading.Thread(target=self.set_weights, daemon=True).start()

    def _state_dir(self) -> str:
        return os.path.join(self.config.model_dir, "vali-state")

    def _configure_logging(self, config: bt.config) -> None:
        bt.logging.set_warning()
        reinitialize_logging()
        configure_logging(config)

    def _on_subnet_metagraph_updated(self, metagraph: bt.metagraph, netuid: int) -> None:
        if netuid != self.config.netuid:
            logging.error(f"Unexpected netuid {netuid}")
            return
        with self.metagraph_lock:
            self.metagraph = copy.deepcopy(metagraph)
            # This will fail if miner_iterator is not updated to handle new UIDs.
            # self.miner_iterator.set_miner_uids(self.metagraph.uids.tolist())
            self.model_tracker.on_hotkeys_updated(set(self.metagraph.hotkeys))

    def update_models(self):
        # Implementation of update_models, _wait_for_open_eval_slot, _queue_top_models_for_eval
        # clean_models, set_weights, try_set_weights, _get_current_block etc.
        # would go here, adapted from the user's provided code but are omitted for brevity
        # as the core logic changes are in run_step and _compute_and_set_competition_weights.
        pass # Placeholder for brevity

    def _get_current_block(self) -> int:
        return self.subtensor.block

    def _new_wandb_run(self):
        run_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = f"validator-{self.uid}-{run_id}"
        self.wandb_run = wandb.init(
            name=name, project=self.config.wandb.project, entity=self.config.wandb.entity,
            config=self.config, dir=self.config.model_dir
        )

    async def run_step(self):
        cur_block = self._get_current_block()
        logging.info(f"run_step | current block: {cur_block}")

        schedule = competition_utils.get_competition_schedule_for_block(cur_block, constants.COMPETITION_SCHEDULE_BY_BLOCK)
        competition = schedule[self.global_step % len(schedule)]
        logging.info(f"run_step | competition: {competition.id}")


        with self.metagraph_lock:
            self.uids_to_eval[competition.id].update(self.pending_uids_to_eval[competition.id])
            self.pending_uids_to_eval[competition.id].clear()

        uids = list(self.uids_to_eval[competition.id])
        if not uids:
            logging.info(f"No UIDs to evaluate for competition {competition.id}. Sleeping.")
            await asyncio.sleep(60)
            return

        uid_to_state = defaultdict(PerUIDEvalState)

        data_loaders = []
        for task in competition.eval_tasks:
            try:
                loader = DatasetLoaderFactory.get_loader(
                    dataset_id=task.dataset_id,
                    dataset_kwargs=task.dataset_kwargs,
                    seed=cur_block,
                    sequence_length=competition.constraints.sequence_length,
                )
                data_loaders.append(loader)
            except Exception as e:
                logging.error(f"Failed to load data for task {task.name}: {e}")
                continue

        load_model_perf = PerfMonitor("Eval: Load model")
        compute_loss_perf = PerfMonitor("Eval: Compute loss")

        for uid in uids:
            with self.metagraph_lock:
                hotkey = self.metagraph.hotkeys[uid]
            uid_to_state[uid].hotkey = hotkey
            meta = self.model_tracker.get_model_metadata_for_miner_hotkey(hotkey)

            if meta and meta.id.competition_id == competition.id:
                uid_to_state[uid].block = meta.block
                uid_to_state[uid].repo_name = meta.id.name
                try:
                    with load_model_perf.sample():
                        model = self.local_store.retrieve_model(hotkey, meta.id, model_constraints=competition.constraints)

                    with compute_loss_perf.sample():
                        # Using run_in_subprocess to isolate model evaluation
                        score, details = utils.run_in_subprocess(
                            functools.partial(
                                self.evaluator.score_model,
                                model=model,
                                data_loaders=data_loaders,
                                device=self.config.device,
                                seed=cur_block
                            ),
                            ttl=600, # 10 minutes timeout
                            mode="spawn",
                        )
                    uid_to_state[uid].score = score
                    uid_to_state[uid].score_details = details
                    del model
                except Exception as e:
                    logging.error(f"Error evaluating model for UID {uid}: {e}\n{traceback.format_exc()}")
                    uid_to_state[uid].score = math.inf
            else:
                uid_to_state[uid].score = math.inf


        ema_scores, logging_metrics = self._compute_and_set_competition_weights(cur_block, uids, uid_to_state, competition)

        # Prioritization based on EMA scores
        prioritization = {uid: ema_scores.get(uid, -math.inf) for uid in uids}
        selected = set(sorted(prioritization, key=prioritization.get, reverse=True)[: self.config.sample_min])

        self._update_uids_to_eval(competition.id, selected, {c.id for c in schedule})

        self.save_state()
        self.log_step(
            competition.id,
            competition.constraints.epsilon_func,
            competition.eval_tasks,
            cur_block,
            uids,
            uid_to_state,
            self._get_uids_to_competition_ids(),
            data_loaders,
            logging_metrics,
            load_model_perf,
            compute_loss_perf,
            PerfMonitor("Eval: Load data") # Placeholder, as data loading isn't timed in this snippet
        )

        self.global_step += 1

    def _compute_and_set_competition_weights(self, block, uids, uid_to_state, competition: Competition):
        """Computes competition weights, EMA scores, and other metrics, then records them."""
        uid_to_score = {uid: state.score for uid, state in uid_to_state.items()}

        # 1. Compute raw scores (if any processing is needed, otherwise it's just uid_to_score)
        # For now, let's assume raw scores are the computed scores.
        raw_scores = uid_to_score

        # 2. Compute raw gap scores
        min_score = min(raw_scores.values(), default=math.inf)
        uid_to_raw_gap_score = {uid: score - min_score for uid, score in raw_scores.items()}

        # 3. Update EMA tracker with the new raw scores
        self.ema_tracker.update(raw_scores, competition_id=competition.id)
        ema_scores = self.ema_tracker.get(competition.id)

        # 4. Determine winning model and record evaluation results
        top_uid = max(ema_scores, key=ema_scores.get, default=None)
        if top_uid is not None:
            self._record_eval_results(top_uid, block, uid_to_state, competition.id)

        # 5. Compute weights based on EMA scores
        model_weights = torch.tensor([ema_scores.get(uid, -math.inf) for uid in uids], dtype=torch.float32)
        step_weights = torch.softmax(model_weights / constants.temperature, dim=0)

        with self.metagraph_lock:
            competition_weights = torch.zeros_like(self.metagraph.S, dtype=torch.float32)
            for idx, uid in enumerate(uids):
                if uid < len(competition_weights):
                    competition_weights[uid] = step_weights[idx]

        # 6. Record competition weights in the tracker and update global weights
        self.ema_tracker.record_competition_weights(competition.id, competition_weights)
        with self.weight_lock:
            schedule = competition_utils.get_competition_schedule_for_block(block, constants.COMPETITION_SCHEDULE_BY_BLOCK)
            self.weights = self.ema_tracker.get_subnet_weights(schedule)

        # 7. Prepare logging metrics
        logging_metrics = {}
        for uid in uids:
            logging_metrics[uid] = {
                "raw_score": raw_scores.get(uid, math.inf),
                "ema_score": ema_scores.get(uid, math.inf),
                "raw_gap_score": uid_to_raw_gap_score.get(uid, math.inf),
            }

        return ema_scores, logging_metrics

    def _record_eval_results(self, top_uid, block, uid_to_state, comp_id):
        top_model_loss = uid_to_state[top_uid].score
        for uid, state in uid_to_state.items():
            self.model_tracker.on_model_evaluated(
                state.hotkey,
                comp_id,
                EvalResult(
                    block=block,
                    score=state.score,
                    winning_model_block=uid_to_state[top_uid].block,
                    winning_model_score=top_model_loss
                )
            )


    def _update_uids_to_eval(self, comp_id, uids, active_ids):
        with self.metagraph_lock:
            self.uids_to_eval[comp_id] = uids
            to_delete = (set(self.uids_to_eval.keys()) | set(self.pending_uids_to_eval.keys())) - active_ids
            for c in to_delete:
                self.uids_to_eval.pop(c, None)
                self.pending_uids_to_eval.pop(c, None)

    def _get_uids_to_competition_ids(self) -> typing.Dict[int, typing.Optional[int]]:
        """Returns a mapping of uids to competition id ints."""
        hotkey_to_metadata = self.model_tracker.get_miner_hotkey_to_model_metadata_dict()
        with self.metagraph_lock:
            uids_to_competition_ids = {}
            for uid in range(self.metagraph.n.item()):
                hotkey = self.metagraph.hotkeys[uid]
                metadata = hotkey_to_metadata.get(hotkey)
                uids_to_competition_ids[uid] = metadata.id.competition_id if metadata else None
            return uids_to_competition_ids


    def save_state(self):
        os.makedirs(self._state_dir(), exist_ok=True)
        with open(self.uids_filepath, "wb") as f:
            pickle.dump(self.uids_to_eval, f)
            pickle.dump(self.pending_uids_to_eval, f)
        self.model_tracker.save_state(self.model_tracker_filepath)
        # Save EMATracker state to the competition_tracker filepath for backward compatibility
        self.ema_tracker.save_state(self.competition_tracker_filepath)

    def log_step(
        self,
        competition_id: CompetitionId,
        competition_epsilon_func: EpsilonFunc,
        eval_tasks: typing.List[Any], # Using Any for EvalTask since import is problematic
        current_block: int,
        uids: typing.List[int],
        uid_to_state: typing.Dict[int, PerUIDEvalState],
        uid_to_competition_id: typing.Dict[int, typing.Optional[int]],
        data_loaders: typing.List[SubsetLoader],
        logging_metrics: typing.Dict[int, typing.Dict[str, float]],
        load_model_perf: PerfMonitor,
        compute_loss_perf: PerfMonitor,
        load_data_perf: PerfMonitor,
    ):
        # Using the log_step function provided by the user in the previous turn
        # This implementation includes raw_score, ema_score, and raw_gap_score
        # and has been updated to use the logging_metrics dict.
        step_log = {
            "timestamp": time.time(),
            "competition_id": competition_id,
            "block": current_block,
            "uids": uids,
            "uid_data": {},
        }

        sub_competition_weights = self.ema_tracker.get_competition_weights(competition_id)
        with self.weight_lock:
            log_weights = self.weights

        for uid in uids:
            metrics = logging_metrics.get(uid, {})
            step_log["uid_data"][str(uid)] = {
                "uid": uid,
                "block": uid_to_state[uid].block,
                "hf": uid_to_state[uid].repo_name,
                "raw_score": metrics.get("raw_score", math.inf),
                "ema_score": metrics.get("ema_score", math.inf),
                "raw_gap_score": metrics.get("raw_gap_score", math.inf),
                "epsilon_adv": competition_epsilon_func.compute_epsilon(current_block, uid_to_state[uid].block),
                "weight": log_weights[uid].item() if uid < len(log_weights) else 0.0,
                "norm_weight": sub_competition_weights[uid].item() if uid < len(sub_competition_weights) else 0.0,
            }

        table = Table(title=f"Step {self.global_step} Results for Competition {competition_id}", expand=True)
        table.add_column("uid", justify="right", style="cyan")
        table.add_column("raw_score", style="magenta")
        table.add_column("ema_score", style="green")
        table.add_column("raw_gap_score", style="yellow")
        table.add_column("weight", style="blue")
        for uid in uids:
            data = step_log["uid_data"][str(uid)]
            table.add_row(
                str(uid),
                f"{data['raw_score']:.4f}",
                f"{data['ema_score']:.4f}",
                f"{data['raw_gap_score']:.4f}",
                f"{data['weight']:.4f}"
            )
        console = Console()
        console.print(table)

        if self.config.wandb.on and not self.config.offline:
            # Simplified wandb logging for this example
            wandb_log = {
                f"raw_score_{uid}": metrics.get("raw_score", math.inf) for uid, metrics in logging_metrics.items()
            }
            wandb_log.update({
                f"ema_score_{uid}": metrics.get("ema_score", math.inf) for uid, metrics in logging_metrics.items()
            })
            self.wandb_run.log(wandb_log)


    async def run(self):
        while True:
            try:
                await self.run_step()
            except KeyboardInterrupt:
                if hasattr(self, "wandb_run") and self.wandb_run: self.wandb_run.finish()
                break
            except Exception as e:
                logging.error(f"Error in run loop: {e}\n{traceback.format_exc()}")

if __name__ == "__main__":
    # Note: To run this, the MinerIterator issue must be resolved.
    # For example, by replacing it with:
    # import itertools
    # self.miner_iterator = itertools.cycle(range(self.metagraph.n.item()))
    # And ensuring all other methods are fully implemented.
    asyncio.run(Validator().run())
