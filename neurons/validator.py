"""
Main validator script.

This orchestrator wires together the validator services that sync miner
submissions, execute validator-owned training, score the resulting checkpoints,
and submit incentives on-chain. Its core responsibilities include:
- Initializing Bittensor objects (wallet, subtensor, metagraph).
- Setting up and starting all background services (ModelManager, WeightSetter).
- Running the main evaluation loop (`run_step`), which coordinates submission
  training, evaluation, scoring, and weighting.
"""
import os
import time
import asyncio
import threading
import traceback
import logging
import random
import math
import datetime as dt
from rich.console import Console
from rich.table import Table

import bittensor as bt
import torch
import wandb
from retry import retry

from neurons import config
from taoverse.metagraph.metagraph_syncer import MetagraphSyncer
import constants
from competitions import competitions
from epochor.utils import metagraph_utils
from epochor.utils import logging as epochor_logging
from epochor.utils.miner_iterator import MinerIterator
from epochor.model.model_updater import ModelUpdater
from epochor.model.storage.disk_model_store import DiskModelStore
from epochor.model.storage.hf_model_store import HuggingFaceModelStore
from epochor.model.storage.metadata_model_store import ChainModelMetadataStore
from epochor.utils.competition_utils import get_competition_schedule_for_block

# Import all components from the new sub-package
from neurons.validator import (
    ValidatorState,
    ModelManager,
    WeightSetter,
    CompetitionManager,
    EvaluationService,
    ScoringService,
    PerUIDEvalState,
)


class Validator:
    """The main orchestrator class for the Bittensor validator."""

    def __init__(self):
        """Initializes all necessary components and services for the validator."""
        self.config = config.validator_config()
        self._configure_logging()

        # Bittensor objects
        try:
            self.wallet = bt.wallet(config=self.config) if self.config.wallet.name and self.config.wallet.hotkey else None
        except Exception:
            self.wallet = None
        self.subtensor = bt.subtensor(config=self.config)
        self.weights_subtensor = bt.subtensor(config=self.config)
        self.dendrite = bt.dendrite(wallet=self.wallet)

        self.metagraph_lock = threading.RLock()
        self._setup_metagraph_syncer()
        if not self.config.offline:
            self.uid = metagraph_utils.assert_registered(self.wallet, self.metagraph)
        if self.config.wandb.on and not self.config.offline:
            self._new_wandb_run()

        # Core state and operational objects
        self.global_step = 0
        self.miner_iterator = MinerIterator(self.metagraph.uids.tolist())
        S_tensor = torch.from_numpy(self.metagraph.S)
        self.weights = torch.zeros_like(S_tensor, dtype=torch.float32)

        # Storage and State Management
        self.local_store = DiskModelStore(base_dir=self.config.model_dir)
        state_dir = os.path.join(self.config.model_dir, "vali-state")
        self.state = ValidatorState(self.metagraph, state_dir, self.metagraph_lock)
        self.state.load()

        # Services and Managers
        self._init_services()

        # Initialize weights from state for the current schedule
        competition_schedule = get_competition_schedule_for_block(
            block=self._get_current_block(),
            schedule_by_block=competitions.COMPETITION_SCHEDULE_BY_BLOCK,
        )
        self.weights = self.state.ema_tracker.subnet_weights(competition_schedule)

    # ---------------------------
    # Wire services
    # ---------------------------
    def _init_services(self):
        """Initializes and wires up all the service classes."""
        metadata_store = ChainModelMetadataStore(self.subtensor, self.config.netuid, self.wallet)
        remote_store = HuggingFaceModelStore()
        model_updater = ModelUpdater(metadata_store, remote_store, self.local_store, self.state.model_tracker)

        self.model_manager = ModelManager(
            model_updater=model_updater,
            model_tracker=self.state.model_tracker,            # ← required in your signature
            miner_iterator=self.miner_iterator,
            metagraph=self.metagraph,
            state=self.state,
            metagraph_lock=self.metagraph_lock,
            local_store=self.local_store,
            get_current_block_fn=self._get_current_block,      # ← required too
        )
        self.weight_setter = WeightSetter(
            self.weights_subtensor, self.wallet, self.config.netuid, self.metagraph,
            self.weights, self.metagraph_lock
        )
        self.competition_manager = CompetitionManager(self.state)
        sandbox_config = getattr(self.config, "sandbox", None)
        self.evaluation_service = EvaluationService(
            self.state,
            self.metagraph,
            self.local_store,
            self.config.device,
            self.metagraph_lock,
            sandbox_config=sandbox_config,
        )
        self.scoring_service = ScoringService(self.state, self.metagraph, self.config)

    # ---------------------------
    # Main step
    # ---------------------------
    async def run_step(self):
        """Executes one full validation step, orchestrating the services."""
        cur_block = self._get_current_block()
        logging.info(f"Current block: {cur_block}")

        # Determine active schedule and this step’s competition
        competition_schedule = get_competition_schedule_for_block(
            block=cur_block, schedule_by_block=competitions.COMPETITION_SCHEDULE_BY_BLOCK
        )
        if not competition_schedule:
            logging.warning("No active competitions. Waiting.")
            await asyncio.sleep(300)
            return
        competition = competition_schedule[self.global_step % len(competition_schedule)]

        # Move pending UIDs → active for this competition
        uids_to_eval = self.competition_manager.prepare_uids_for_eval(competition.id)
        if not uids_to_eval:
            logging.debug(f"No UIDs to eval for {competition.id}. Sleeping 5 minutes.")
            await asyncio.sleep(300)
            return

        # Load data for competition
        seed = self._get_seed()
        eval_tasks, samples = self.competition_manager.load_data_for_competition(competition, seed)
        if not samples:
            logging.warning(f"No data for {competition.id}. Skipping step.")
            return

        # Evaluate models
        logging.info(f"Evaluating {len(uids_to_eval)} UIDs for competition: {competition.id}")
        uid_to_state = self.evaluation_service.evaluate_uids(uids_to_eval, competition, samples, eval_tasks, seed)

        # Score and update weights
        scoring_metrics, models_to_keep = self.scoring_service.process_scores_and_update_weights(
            uids_to_eval, uid_to_state, competition, cur_block
        )

        # Finalize state for the next step
        active_comps = {c.id for c in competition_schedule}
        self.state.update_uids_to_eval(competition.id, models_to_keep, active_comps)
        self.state.reset_ema_competitions(active_comps)  # keep EMA clean w.r.t. schedule
        self.weights = self.state.ema_tracker.subnet_weights(competition_schedule)
        self.state.save()

        self.log_step(competition, uids_to_eval, uid_to_state, scoring_metrics, seed)
        self.global_step += 1

    # ---------------------------
    # Background threads
    # ---------------------------
    def run_in_background(self):
        """Starts all background threads."""
        self.model_manager.start()
        if not self.config.offline and not self.config.dont_set_weights:
            self.weight_setter.start()

    def __enter__(self):
        self.run_in_background()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.model_manager.stop()
        if not self.config.offline and not self.config.dont_set_weights:
            self.weight_setter.stop()
        if self.config.wandb.on and not self.config.offline:
            wandb.finish()

    # ---------------------------
    # Helpers
    # ---------------------------
    def _configure_logging(self):
        epochor_logging.reinitialize_logging()
        epochor_logging.configure_logging(self.config)

    def _setup_metagraph_syncer(self):
        syncer_subtensor = bt.subtensor(config=self.config)
        self.subnet_metagraph_syncer = MetagraphSyncer(
            syncer_subtensor,
            config={self.config.netuid: dt.timedelta(minutes=20).total_seconds()},
            lite=False
        )
        self.subnet_metagraph_syncer.do_initial_sync()
        self.metagraph = self.subnet_metagraph_syncer.get_metagraph(self.config.netuid)
        self.subnet_metagraph_syncer.register_listener(
            self._on_subnet_metagraph_updated, netuids=[self.config.netuid]
        )
        self.subnet_metagraph_syncer.start()
        self.known_hotkeys = set(self.metagraph.hotkeys)

    def _on_subnet_metagraph_updated(self, metagraph, netuid):
        if netuid != self.config.netuid:
            return
        with self.metagraph_lock:
            # Re-init EMA tracker when neuron count changes
            if len(getattr(self, "metagraph", metagraph).uids) != len(metagraph.uids):
                self.state.ema_tracker = self.state.ema_tracker.__class__(num_neurons=len(metagraph.uids))
                logging.info("Re-initialized CompetitionEMATracker due to neuron-count change.")

            # Reset scores for any newly added hotkeys
            old_hotkeys = set(getattr(self, "metagraph", metagraph).hotkeys if hasattr(self, "metagraph") else [])
            new_hotkeys = set(metagraph.hotkeys)
            added = new_hotkeys - old_hotkeys
            for hk in added:
                self.state.reset_ema_hotkey_score(hotkey=hk)

            self.metagraph = metagraph
            self.miner_iterator.set_miner_uids(metagraph.uids.tolist())

    @retry(tries=3, delay=1)
    def _get_current_block(self) -> int:
        return self.subtensor.block

    def _get_seed(self):
        # Simplified fallback
        try:
            from epochor.utils import metagraph_utils as mgu
            return mgu.get_hash_of_sync_block(self.subtensor, constants.SYNC_BLOCK_CADENCE)
        except Exception:
            return random.randint(0, 2**32 - 1)

    def _new_wandb_run(self):
        # Simplified wandb setup
        self.wandb_run = wandb.init(project="epochor-subnet")

    def log_step(self, competition, uids, uid_to_state, scoring_metrics, seed):
        console = Console()
        table = Table(title=f"Step {self.global_step} | Competition: {competition.id}")
        table.add_column("UID")
        table.add_column("Seed")
        table.add_column("HF Repo")
        table.add_column("Raw Score")
        table.add_column("Final Score")
        table.add_column("Weight")

        final_scores = scoring_metrics.get("final_scores_dict", {})
        ema_scores = self.state.ema_tracker.get(competition.id)  # {uid: ema_score}
        sub_comp_weights = self.state.ema_tracker.get_competition_weights(competition.id)

        for uid in uids:
            final_score = float(final_scores.get(uid, math.inf))
            ema_val = float(ema_scores.get(uid, 0.0))
            weight_val = float(sub_comp_weights[uid].item()) if uid < len(sub_comp_weights) else 0.0
            table.add_row(
                str(uid),
                str(seed),
                uid_to_state[uid].repo_name,
                f"{uid_to_state[uid].score:.4f}" if math.isfinite(uid_to_state[uid].score) else "inf",
                f"{final_score:.4f}" if math.isfinite(final_score) else "inf",
                f"{ema_val:.4f}" if math.isfinite(ema_val) else "inf",
                f"{weight_val:.4f}",
            )
        console.print(table)


async def main():
    """The main entry point for the validator."""
    with Validator() as validator:
        while True:
            try:
                await asyncio.wait_for(validator.run_step(), timeout=60 * 20)
            except asyncio.TimeoutError:
                logging.warning("run_step timed out. Proceeding to next step.")
            except KeyboardInterrupt:
                logging.info("Shutting down validator.")
                break
            except Exception as e:
                logging.error(f"Error in main loop: {e}\n{traceback.format_exc()}")
                await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())
