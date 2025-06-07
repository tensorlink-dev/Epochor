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

        # Trackers - Using EMATracker for scores and weights, removing CompetitionTracker
        self.model_tracker = ModelTracker()
        self.ema_tracker = EMATracker(alpha=constants.alpha)

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
                # Load state into the EMATracker, assuming it's the new standard
                self.ema_tracker.load_state(self.competition_tracker_filepath)
            except Exception as e:
                logging.error(f"Failed to load EMA tracker state: {e}")

        self.uids_to_eval = defaultdict(set)
        self.pending_uids_to_eval = defaultdict(set)
        if os.path.exists(self.uids_filepath):
            with open(self.uids_filepath, "rb") as f:
                self.uids_to_eval = pickle.load(f)
                self.pending_uids_to_eval = pickle.load(f)

        # Use itertools.cycle for a correct and robust iterator
        self.miner_iterator = cycle(range(self.metagraph.n.item()))

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
        self.pending_uids_to_eval_lock = threading.RLock()
        self.global_step = 0

        self.stop_event = threading.Event()
        self.update_thread = threading.Thread(target=self.update_models, daemon=True)
        self.update_thread.start()
        self.clean_thread = threading.Thread(target=self.clean_models, daemon=True)
        self.clean_thread.start()
        if not self.config.offline and not self.config.dont_set_weights:
            self.weight_thread = threading.Thread(target=self.set_weights, daemon=True)
            self.weight_thread.start()
