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
from collections import defaultdict
import datetime as dt
import logging

import bittensor as bt
import torch
import wandb
from retry import retry

from template.base.validator import BaseValidatorNeuron
from epochor.competition.data import Competition
from epochor.competition import utils as competition_utils
from neurons import config
from neurons.validator2 import ValidatorState, ModelManager, WeightSetter
from epochor.model.model_updater import ModelUpdater
from epochor.model.storage.disk_model_store import DiskModelStore
from epochor.model.storage.hf_model_store import HuggingFaceModelStore
from epochor.model.storage.metadata_model_store import ChainModelMetadataStore
from epochor.utils import metagraph_utils
from taoverse.metagraph.metagraph_syncer import MetagraphSyncer
from taoverse.metagraph.miner_iterator import MinerIterator
import constants

class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super().__init__(config=config)
        self.config = config.validator_config() if config else config.validator_config()

        # === Bittensor objects ====
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.weights_subtensor = bt.subtensor(config=self.config)
        self.dendrite = bt.dendrite(wallet=self.wallet)

        # Setup metagraph syncer for the subnet based on config. This is non-lite for getting weights by vali.
        syncer_subtensor = bt.subtensor(config=self.config)
        self.subnet_metagraph_syncer = MetagraphSyncer(
            syncer_subtensor,
            config={
                self.config.netuid: dt.timedelta(minutes=20).total_seconds(),
            },
            lite=False,
        )
        # Perform an initial sync of all tracked metagraphs.
        self.subnet_metagraph_syncer.do_initial_sync()
        self.subnet_metagraph_syncer.start()
        # Get initial metagraphs.
        self.metagraph: bt.metagraph = self.subnet_metagraph_syncer.get_metagraph(
            self.config.netuid
        )

        # Register a listener for metagraph updates.
        self.subnet_metagraph_syncer.register_listener(
            self._on_subnet_metagraph_updated,
            netuids=[self.config.netuid],
        )

        torch.backends.cudnn.benchmark = True

        # Dont check registration status if offline.
        if not self.config.offline:
            self.uid = metagraph_utils.assert_registered(self.wallet, self.metagraph)

        # Track how may run_steps this validator has completed.
        self.run_step_count = 0

        # Dont log to wandb if offline.
        if not self.config.offline and self.config.wandb.on:
            self._new_wandb_run()

        # === Running args ===
        self.weight_lock = threading.RLock()
        self.weights = torch.zeros_like(torch.from_numpy(self.metagraph.S))
        self.global_step = 0

        self.uids_to_eval: typing.Dict[int, typing.Set] = defaultdict(set)

        # Create a set of newly added uids that should be evaluated on the next loop.
        self.pending_uids_to_eval_lock = threading.RLock()
        self.pending_uids_to_eval: typing.Dict[int, typing.Set] = defaultdict(
            set
        )

         # Setup a miner iterator to ensure we update all miners.
        # This subnet does not differentiate between miner and validators so this is passed all uids.
        self.miner_iterator = MinerIterator(self.metagraph.uids.tolist())

        # Setup a ModelMetadataStore
        chain_store_subtensor = bt.subtensor(config=self.config)
        self.metadata_store = ChainModelMetadataStore(
            subtensor=chain_store_subtensor,
            subnet_uid=self.config.netuid,
            wallet=self.wallet,
        )

        # Setup a RemoteModelStore
        self.remote_store = HuggingFaceModelStore()

        # Setup a LocalModelStore
        self.local_store = DiskModelStore(base_dir=self.config.model_dir)

        # Setup a model updater to download models as needed to match the latest provided miner metadata.
        self.model_updater = ModelUpdater(
            metadata_store=self.metadata_store,
            remote_store=self.remote_store,
            local_store=self.local_store,
            model_tracker=None, #self.state.model_tracker,
        )


        # Construct the filepaths to save/load state.
        state_dir = os.path.join(self.config.model_dir, "vali-state")
        os.makedirs(state_dir, exist_ok=True)

        # Initialize the helper classes
        self.state = ValidatorState(base_dir=state_dir)
        self.state.load()

        self.model_manager = ModelManager(
            model_updater=self.model_updater,  # Replace with actual model_updater
            model_tracker=self.state.model_tracker,
            miner_iterator=self.miner_iterator,  # Replace with actual miner_iterator
            metagraph=self.metagraph,
        )

        self.weight_setter = WeightSetter(
            subtensor=self.weights_subtensor,
            wallet=self.wallet,
            netuid=self.config.netuid,
            weights=self.weights,
        )


        # Also update our internal weights based on the tracker.
        cur_block = self._get_current_block()

        # Get the competition schedule for the current block.
        # This is a list of competitions
        competition_schedule: typing.List[Competition] = (
            competition_utils.get_competition_schedule_for_block(
                block=cur_block,
                schedule_by_block=constants.COMPETITION_SCHEDULE_BY_BLOCK,
            )
        )

    def run_in_background_thread(self):
        self.model_manager.start()
        self.weight_setter.start()

    def __enter__(self):
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.model_manager.stop()
        self.weight_setter.stop()

    async def run_step(self):
        """
        Executes a step in the evaluation process of models.
        """
        # Implement your core validation logic here
        print("Running validation step...")
        time.sleep(1)

        # Save state
        self.state.save(self.uids_to_eval, self.pending_uids_to_eval)

    def _get_current_block(self) -> int:
        """Returns the current block."""

        @retry(tries=5, delay=1, backoff=2)
        def _get_block_with_retry():
            return self.subtensor.block

        try:
            return _get_block_with_retry()
        except:
            logging.debug(
                "Failed to get the latest block from the chain. Using the block from the cached metagraph."
            )
            # Network call failed. Fallback to using the block from the metagraph,
            # even though it'll be a little stale.
            #with self.metagraph_lock:
            return 1 #self.metagraph.block.item()

    def _on_subnet_metagraph_updated(
        self, metagraph: bt.metagraph, netuid: int
    ):
        if netuid != self.config.netuid:
            return

        #with self.metagraph_lock:
        #    old = self.known_hotkeys
        #    new = set(metagraph.hotkeys)
        #    added = new - old

        #    if added:
        #        logging.info(f"New hotkeys registered: {added}")
        #        for hk in added:
        #            # you may not yet know its competition_id,
        #            # but you can reset globally or per‐competition once you do
        #            self.ema_tracker.reset_score_for_hotkey(hotkey=hk)

        #    self.known_hotkeys = new
        #    self.metagraph = copy.deepcopy(metagraph)
        #    self.miner_iterator.set_miner_uids(list(new))
        #    self.model_tracker.on_hotkeys_updated(new)
        pass

    def _new_wandb_run(self):
        """Creates a new wandb run to save information to."""

        # Create a unique run id for this run.
        run_id = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = "validator-" + str(self.uid) + "-" + run_id
        self.wandb_run = wandb.init(
            name=name,
            project=self.config.wandb_project,
            entity="macrocosmos",
            config={
                "uid": self.uid,
                "hotkey": self.wallet.hotkey.ss58_address,
                "run_name": run_id,
                "version": constants.__version__,
                "validator version": constants.__validator_version__,
                "type": "validator",
            },
            allow_val_change=True,
        )

        logging.debug(f"Started a new wandb run: {name}")


if __name__ == "__main__":
    # Set an output width explicitly for rich table output (affects the pm2 tables that we use).
    try:
        width = os.get_terminal_size().columns
    except:
        width = 0
    os.environ["COLUMNS"] = str(max(200, width))

    with Validator() as validator:
        while True:
            asyncio.run(validator.run_step())
            time.sleep(1)
