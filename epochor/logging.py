# epochor/logging.py

import logging
import sys

# Store the initial logging configuration to allow reinitialization
_initial_log_level = logging.INFO
_log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Keep track of whether basicConfig has been called by this module
_configured = False

def reinitialize(log_level: int = None, force_reinit: bool = False) -> logging.Logger:
    """
    Configures and returns a logger for the 'epochor' namespace.
    It ensures that basicConfig is called only once unless forced.

    Args:
        log_level: The logging level (e.g., logging.INFO, logging.DEBUG).
                   Defaults to the module's _initial_log_level.
        force_reinit: If True, forces re-configuration even if already configured.
                      This is generally not recommended if other parts of the
                      application also configure logging.

    Returns:
        A logging.Logger instance.
    """
    global _configured, _initial_log_level

    chosen_log_level = log_level if log_level is not None else _initial_log_level

    # Python's logging.basicConfig() can only be effectively called once.
    # Subsequent calls are no-ops unless force=True is used (Python 3.8+).
    # For broader compatibility and to avoid issues with other modules configuring logging,
    # we try to be careful here.
    
    # If force_reinit, and Python version supports it, try to remove handlers and reconfigure.
    # This is a bit more involved and can have side effects.
    # A simpler approach is to just set the level of the root logger or specific loggers.

    if not _configured or force_reinit:
        # For force_reinit, if using Python 3.8+, can do:
        # logging.basicConfig(level=chosen_log_level, format=_log_format, force=True)
        # However, 'force' is not available in older Pythons.
        # A more compatible way if re-configuring: remove existing handlers.
        if force_reinit:
            # This removes handlers from the root logger.
            # Be cautious if other libraries add handlers you want to keep.
            root = logging.getLogger()
            for handler in root.handlers[:]:
                root.removeHandler(handler)
            logging.basicConfig(level=chosen_log_level, format=_log_format, stream=sys.stdout)
        else:
             logging.basicConfig(level=chosen_log_level, format=_log_format, stream=sys.stdout)
        
        _initial_log_level = chosen_log_level # Store the level that was used
        _configured = True
        logging.getLogger(__name__).info(f"Logging reinitialized to level {logging.getLevelName(chosen_log_level)}")

    # Get a logger specific to the 'epochor' namespace or the calling module's namespace.
    # Using 'epochor' as a root namespace for the project is a good practice.
    logger = logging.getLogger("epochor")
    logger.setLevel(chosen_log_level) # Ensure this specific logger instance has the desired level
    
    # If no handlers are configured for this logger specifically, and basicConfig was called,
    # it will propagate to the root logger's handlers.
    # If you want specific handlers (e.g., file output) for 'epochor', add them here.

    return logger

# Initialize once with default settings when module is imported
# if not _configured:
#    reinitialize() # Or defer to first explicit call


def log_step(
    self,
    competition_id: CompetitionId,
    competition_epsilon_func: EpsilonFunc,
    eval_tasks: typing.List[EvalTask],
    current_block: int,
    uids: typing.List[int],
    uid_to_state: typing.Dict[int, PerUIDEvalState],
    uid_to_competition_id: typing.Dict[int, typing.Optional[int]],
    data_loaders: typing.List[SubsetLoader],
    logging_metrics: typing.Dict[int, typing.Dict[str, float]], # Added logging_metrics
    load_model_perf: PerfMonitor,
    compute_loss_perf: PerfMonitor,
    load_data_perf: PerfMonitor,
):
    """Logs the results of the step to the console and wandb (if enabled)."""
    # Get pages from each data loader
    pages = []
    for loader in data_loaders:
        for page_name in loader.get_page_names():
            pages.append(f"{loader.name}:{loader.config}:{page_name}")

    # Build step log
    step_log = {
        "timestamp": time.time(),
        "competition_id": competition_id,
        "pages": pages,
        "uids": uids,
        "uid_data": {},
    }

    # Get sub-competition weights from the tracker
    # Ensure competition_id exists in ema_tracker before accessing
    if competition_id in self.ema_tracker.competition_weights:
            sub_competition_weights = self.ema_tracker.get_competition_weights(competition_id)
    else:
            logging.warning(f"Competition ID {competition_id} not found in EMA tracker competition weights.")
            sub_competition_weights = torch.zeros_like(self.metagraph.S, dtype=torch.float32)


    # Get a copy of global weights to print.
    with self.weight_lock:
        log_weights = self.weights

    # All uids in the competition step log are from the same competition.
    for uid in uids:
        metrics = logging_metrics.get(uid, {}) # Get metrics for this UID, default to empty dict
        
        step_log["uid_data"][str(uid)] = {
            "uid": uid,
            "block": uid_to_state[uid].block,
            "hf": uid_to_state[uid].repo_name,
            "competition_id": int(competition_id),
            "raw_score": metrics.get("raw_score", math.inf), # Use metrics dict for raw_score
            "ema_score": metrics.get("ema_score", math.inf), # Use metrics dict for ema_score
            "raw_gap_score": metrics.get("raw_gap_score", math.inf), # Add raw_gap_score
            "epsilon_adv": competition_epsilon_func.compute_epsilon(
                current_block, uid_to_state[uid].block
            ),
            "weight": log_weights[uid].item() if uid < len(log_weights) else 0.0,
            "norm_weight": sub_competition_weights[uid].item() if uid < len(sub_competition_weights) else 0.0,
            "dataset_perf": {},
        }

        for task in eval_tasks:
            # Use .get to handle missing tasks safely in uid_to_state[uid].score_details
            score_details = uid_to_state[uid].score_details.get(task.name, ScoreDetails())
            step_log["uid_data"][str(uid)][f"{task.name}.raw_score"] = score_details.raw_score
            step_log["uid_data"][str(uid)][f"{task.name}.norm_score"] = score_details.norm_score
            step_log["uid_data"][str(uid)][f"{task.name}.weighted_norm_score"] = score_details.weighted_norm_score

            # Also log in this older format to avoid breaking the leaderboards.
            task_to_dataset_name = {
                "FALCON": "tiiuae/falcon-refinedweb",
                "FINEWEB": "HuggingFaceFW/fineweb",
                "FINEWEB_EDU2": "HuggingFaceFW/fineweb-edu-score-2",
                "STACKV2_DEDUP": "bigcode/the-stack-v2-dedup",
                "PES2OX": "laion/Pes2oX-fulltext",
                "FINEMATH_3P": "HuggingFaceTB/finemath:finemath-3p",
                "INFIWEBMATH_3P": "HuggingFaceTB/finemath:infiwebmath-3p",
                "PPL_SPEECH": "MLCommons/peoples_speech"
            }
            dataset_name = task_to_dataset_name.get(task.name, "DatasetNameNotFound")

            step_log["uid_data"][str(uid)]["dataset_perf"][f"{dataset_name}"] = {
                "average_loss": score_details.raw_score # Older format logs raw_score here
            }


    table = Table(title="Step", expand=True)
    table.add_column("uid", justify="right", style="cyan", no_wrap=True)
    table.add_column("hf", style="magenta", overflow="fold")
    table.add_column("raw_score", style="magenta", overflow="fold") # Updated header
    table.add_column("ema_score", style="magenta", overflow="fold") # Added EMA score
    table.add_column("raw_gap_score", style="magenta", overflow="fold") # Added raw gap score
    table.add_column("epsilon_adv", style="magenta", overflow="fold")
    table.add_column("total_weight", style="magenta", overflow="fold")
    table.add_column("comp_weight", style="magenta", overflow="fold")
    table.add_column("block", style="magenta", overflow="fold")
    table.add_column("comp", style="magenta", overflow="fold")
    for uid in uids:
        try:
            uid_data = step_log["uid_data"][str(uid)]
            table.add_row(
                str(uid),
                str(uid_data["hf"]),
                str(round(uid_data["raw_score"], 4)),
                str(round(uid_data["ema_score"], 4)),
                str(round(uid_data["raw_gap_score"], 4)),
                str(round(uid_data["epsilon_adv"], 4)),
                str(round(uid_data["weight"], 4)),
                str(round(uid_data["norm_weight"], 4)),
                str(uid_data["block"]),
                str(uid_data["competition_id"]),
            )
        except Exception:
                logging.exception(f"Error logging data for UID {uid} to Rich table.")


    console = Console()
    console.print(table)

    ws, ui = self.weights.topk(min(len(self.weights), len(self.metagraph.uids)))
    table = Table(title="Weights > 0.001")
    table.add_column("uid", justify="right", style="cyan", no_wrap=True)
    table.add_column("weight", style="magenta")
    table.add_column("comp", style="magenta")

    uid_to_current_competition = self._get_uids_to_competition_ids()

    for index, weight in zip(ui.tolist(), ws.tolist()):
        if weight > 0.001:
                if index in uid_to_current_competition:
                table.add_row(
                    str(index),
                    str(round(weight, 4)),
                    str(uid_to_current_competition[index]),
                )
                else:
                    logging.warning(f"UID {index} not found in uid_to_competition_id map for logging weights table.")

    console.print(table)

    logging.info(f"Step results: {step_log}")

    if self.config.wandb.on and not self.config.offline:
        if (
            self.run_step_count
            and self.run_step_count % constants.MAX_RUN_STEPS_PER_WANDB_RUN == 0
        ):
            logging.trace(
                f"Validator has completed {self.run_step_count} run steps. Creating a new wandb run."
            )
            self.wandb_run.finish()
            self._new_wandb_run()

        logged_uids = step_log["uids"]
        logged_uid_data = step_log["uid_data"]

        graphed_data = {
            "time": time.time(),
            "step_competition_id": competition_id,
            "block": current_block,
            "raw_score_data": {
                str(uid): logged_uid_data[str(uid)].get("raw_score", math.inf) for uid in logged_uids
            },
                "ema_score_data": {
                str(uid): logged_uid_data[str(uid)].get("ema_score", math.inf) for uid in logged_uids
            },
            "raw_gap_score_data": { # Added raw gap score data for Wandb
                str(uid): logged_uid_data[str(uid)].get("raw_gap_score", math.inf) for uid in logged_uids
            },
            "uid_epsilon_adv": {
                str(uid): logged_uid_data[str(uid)].get("epsilon_adv", 0.0) for uid in logged_uids
            },
            "weight_data": {str(uid): logged_uid_data[str(uid)].get("weight", 0.0) for uid in logged_uids},
            "competition_weight_data": {
                str(uid): logged_uid_data[str(uid)].get("norm_weight", 0.0) for uid in logged_uids
            },
            "competition_id": {str(uid): int(competition_id) for uid in logged_uids},
            "load_model_perf": {
                "min": load_model_perf.min(),
                "median": load_model_perf.median(),
                "max": load_model_perf.max(),
                "P90": load_model_perf.percentile(90),
            },
            "compute_model_perf": {
                "min": compute_loss_perf.min(),
                "median": compute_loss_perf.median(),
                "max": compute_loss_perf.max(),
                "P90": compute_loss_perf.percentile(90),
            },
            "load_data_perf": {
                "min": load_data_perf.min(),
                "median": load_data_perf.median(),
                "max": load_data_perf.max(),
                "P90": load_data_perf.percentile(90),
            },
        }
        for task in eval_tasks:
                task_raw_scores = {}
                task_norm_scores = {}
                task_weighted_norm_scores = {}
                for uid in logged_uids:
                    uid_data = logged_uid_data[str(uid)]
                    # Get from the primary score_details logged in step_log["uid_data"]
                    raw_score_detail = uid_data.get(f"{task.name}.raw_score", math.inf) # Get raw_score directly
                    norm_score_detail = uid_data.get(f"{task.name}.norm_score", 0.0) # Get norm_score
                    weighted_norm_score_detail = uid_data.get(f"{task.name}.weighted_norm_score", 0.0) # Get weighted_norm_score

                    task_raw_scores[str(uid)] = raw_score_detail
                    task_norm_scores[str(uid)] = norm_score_detail
                    task_weighted_norm_scores[str(uid)] = weighted_norm_score_detail


                graphed_data[f"{task.name}.raw_score"] = task_raw_scores
                graphed_data[f"{task.name}.norm_score"] = task_norm_scores
                graphed_data[f"{task.name}.weighted_norm_score"] = task_weighted_norm_scores


        logging.trace("Logging to Wandb")
        self.wandb_run.log(
            {**graphed_data, "original_format_json": json.dumps(step_log, indent=4)}
        )

        self.run_step_count += 1
