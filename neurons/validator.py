# epochor/validator.py

import os
import time
import math
import json
import pickle
import asyncio
import dataclasses
import traceback
import torch
import numpy as np
from typing import Dict, Set

import bittensor as bt
from huggingface_hub.utils import disable_progress_bars
from retry import retry

from epochor.config import EPOCHOR_CONFIG
from epochor.generators import CombinedGenerator
from epochor.evaluation import CRPSEvaluator
from epochor.validation import validate
from epochor.logging import reinitialize
from epochor.ema_tracker import EMATracker
from epochor.metrics_logger import MetricsLogger

from taoverse.model.model_tracker import ModelTracker
from taoverse.model.model_updater import ModelUpdater
from taoverse.model.storage.disk.disk_model_store import DiskModelStore
from taoverse.model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
from taoverse.model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore
from taoverse.metagraph.metagraph_syncer import MetagraphSyncer
from taoverse.metagraph.miner_iterator import MinerIterator
from taoverse.model.competition.competition_tracker import CompetitionTracker
from taoverse.metagraph import utils as metagraph_utils
from taoverse.model.competition import utils as competition_utils
from taoverse.model.competition.data import Competition
from taoverse.model.competition.epsilon import EpsilonFunc

import wandb
import constants

disable_progress_bars()
reinitialize()

STATE_DIR = os.path.join(EPOCHOR_CONFIG.model_dir, "validator_state")
MODEL_TRACKER_FP = os.path.join(STATE_DIR, "model_tracker.pkl")
UIDS_FP = os.path.join(STATE_DIR, "uids.pkl")
EMA_FP = os.path.join(STATE_DIR, "ema_scores.pkl")


@dataclasses.dataclass
class EvalState:
    block: int = -1
    score: float = float("inf")
    validated: float = float("inf")
    ema: float = float("inf")
    weight: float = 0.0


class EpochorValidator(bt.Validator):
    def __init__(self, config: bt.Config):
        super().__init__(config=config)
        self.config = config
        os.makedirs(STATE_DIR, exist_ok=True)

        self.wallet = bt.wallet(config=config)
        self.subtensor = bt.subtensor(config=config)
        self.metagraph_syncer = MetagraphSyncer(self.subtensor, {config.netuid: 300}, lite=False)
        self.metagraph_syncer.do_initial_sync()
        self.metagraph_syncer.start()
        self.metagraph = self.metagraph_syncer.get_metagraph(config.netuid)
        self.miner_iterator = MinerIterator(self.metagraph.uids.tolist())

        self.local_store = DiskModelStore(base_dir=config.model_dir)
        self.remote_store = HuggingFaceModelStore()
        self.metadata_store = ChainModelMetadataStore(self.subtensor, config.netuid, self.wallet)
        self.model_tracker = self._load_or_initialize(MODEL_TRACKER_FP, ModelTracker)
        self.model_updater = ModelUpdater(self.metadata_store, self.remote_store, self.local_store, self.model_tracker)
        self.competition_tracker = CompetitionTracker(num_neurons=len(self.metagraph.uids), alpha=constants.alpha)

        self.generator = CombinedGenerator(
            samplers=EPOCHOR_CONFIG.samplers,
            registries=EPOCHOR_CONFIG.registries,
            length=EPOCHOR_CONFIG.series_length,
            weights=EPOCHOR_CONFIG.category_weights
        )
        self.evaluator = CRPSEvaluator()
        alpha = 2 / (EPOCHOR_CONFIG.ema_span + 1) if EPOCHOR_CONFIG.ema_span > 0 else 1.0
        self.ema_tracker = self._load_or_initialize(EMA_FP, lambda: EMATracker(alpha=alpha))
        self.metrics_logger = MetricsLogger(
            project_name=EPOCHOR_CONFIG.wandb_project,
            entity=EPOCHOR_CONFIG.wandb_entity,
            disabled=not EPOCHOR_CONFIG.use_wandb
        )
        self.uid_eval_set: Set[int] = self._load_or_initialize(UIDS_FP, lambda: set())

        self.previous_block = -1
        self.global_step = 0

    def _load_or_initialize(self, filepath, constructor):
        if os.path.exists(filepath):
            try:
                with open(filepath, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                bt.logging.warning(f"Failed to load {filepath}, reinitializing. Error: {e}")
        return constructor()

    def _save_state(self):
        with open(MODEL_TRACKER_FP, "wb") as f:
            pickle.dump(self.model_tracker, f)
        with open(EMA_FP, "wb") as f:
            pickle.dump(self.ema_tracker, f)
        with open(UIDS_FP, "wb") as f:
            pickle.dump(self.uid_eval_set, f)

    def _get_seed(self):
        @retry(tries=3, delay=1, backoff=2)
        def _get_seed_with_retry():
            return metagraph_utils.get_hash_of_sync_block(self.subtensor, constants.SYNC_BLOCK_CADENCE)
        try:
            return _get_seed_with_retry()
        except:
            bt.logging.warning("Failed to get hash of sync block. Using fallback seed.")
            return np.random.randint(0, 2**32 - 1)

    def _get_burn_uid(self):
        try:
            hotkey = self.subtensor.query_subtensor("SubnetOwnerHotkey", [self.config.netuid])
            return self.subtensor.get_uid_for_hotkey_on_subnet(hotkey_ss58=hotkey, netuid=self.config.netuid)
        except:
            return 0

    def forward(self):
        block = bt.utils.get_current_block()
        if block <= self.previous_block:
            return

        series, _ = self.generator.generate(seed=block)
        metagraph = self.metagraph_syncer.get_metagraph(self.config.netuid)
        active_uids = [uid.item() for uid in metagraph.uids if metagraph.axons[uid.item()].is_serving]

        uid_states: Dict[int, EvalState] = {}
        for uid in active_uids:
            try:
                hotkey = metagraph.hotkeys[uid]
                asyncio.run(self.model_updater.sync_model(uid, hotkey, block, constants.COMPETITION_SCHEDULE_BY_BLOCK))
                metadata = self.model_tracker.get_model_metadata_for_miner_hotkey(hotkey)
                if metadata is None:
                    continue
                model = self.local_store.retrieve_model(hotkey, metadata.id, {})
                model.eval()
                preds = model(series)
                score = self.evaluator.evaluate(series, preds)
                uid_states[uid] = EvalState(block=metadata.block, score=score)
            except Exception as e:
                bt.logging.warning(f"Failed to score UID {uid}: {e}")

        raw_scores = {uid: s.score for uid, s in uid_states.items()}
        try:
            validated = validate(raw_scores)
            for uid, val in validated.items():
                if np.isfinite(val):
                    self.ema_tracker.update(uid, val)
                    uid_states[uid].validated = val
                    uid_states[uid].ema = self.ema_tracker.get(uid)
        except Exception as e:
            bt.logging.error(f"Validation step failed: {e}")
            for uid, state in uid_states.items():
                if np.isfinite(state.score):
                    self.ema_tracker.update(uid, state.score)
                    state.validated = state.score
                    state.ema = self.ema_tracker.get(uid)

        ema_scores = self.ema_tracker.get_all_scores()
        scores = torch.tensor([
            1. / (1. + ema_scores[uid]) if uid in ema_scores and np.isfinite(ema_scores[uid]) else 0.0
            for uid in active_uids
        ])
        softmax_weights = torch.softmax(scores, dim=0)
        weights_dict = {uid: w.item() for uid, w in zip(active_uids, softmax_weights)}

        try:
            final_weights = {uid: weights_dict.get(uid, 0.0) for uid in active_uids}
            non_zero = [(uid, w) for uid, w in final_weights.items() if w > 0.0 and np.isfinite(w)]
            if not non_zero:
                burn_uid = self._get_burn_uid()
                bt.logging.warning("No valid weights, defaulting to burn UID.")
                non_zero = [(burn_uid, 1.0)]

            uids, weights = zip(*non_zero)
            self.subtensor.set_weights(
                netuid=self.config.netuid,
                uids=np.array(uids, dtype=np.int64),
                weights=np.array(weights, dtype=np.float32),
                wait_for_inclusion=False,
                wait_for_finalization=False,
                version_key=bt.__version_as_int__
            )
            bt.logging.info(f"Set weights for {len(uids)} UIDs at block {block}.")
        except Exception as e:
            bt.logging.error(f"Failed to set weights: {e}\n{traceback.format_exc()}")

        if not self.metrics_logger.disabled:
            wandb_data = {
                "block": block,
                "uids_evaluated": len(uid_states),
                "avg_crps": np.nanmean([s.score for s in uid_states.values() if np.isfinite(s.score)]),
                "ema_spread": np.nanstd([s.ema for s in uid_states.values() if np.isfinite(s.ema)]),
                "submitted_weights": len(uids),
                "weight_sum": float(np.sum(weights)),
            }
            for uid, state in uid_states.items():
                wandb_data[f"uid_{uid}/ema"] = state.ema
                wandb_data[f"uid_{uid}/val"] = state.validated
                wandb_data[f"uid_{uid}/weight"] = weights_dict.get(uid, 0.0)
            self.metrics_logger.log(wandb_data, step=block)

        self._save_state()
        self.previous_block = block
        self.global_step += 1

    async def try_run_step(self, ttl: int):
        async def _try_run_step():
            self.forward()
        try:
            await asyncio.wait_for(_try_run_step(), timeout=ttl)
        except asyncio.TimeoutError:
            bt.logging.error("Timeout running validator step.")

    async def run(self):
        while True:
            try:
                await self.try_run_step(ttl=600)
            except KeyboardInterrupt:
                bt.logging.info("Validator stopped.")
                break
            except Exception as e:
                bt.logging.error(f"Error in run loop: {e}\n{traceback.format_exc()}")


if __name__ == "__main__":
    asyncio.run(EpochorValidator(bt.config()).run())
