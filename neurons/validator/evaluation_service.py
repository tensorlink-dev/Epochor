"""Core evaluation engine for scoring miner models against prepared datasets."""

import dataclasses
import math
import logging
import traceback
import functools
import typing
from collections import defaultdict
import threading

import bittensor as bt
from epochor.utils import misc as utils, model_utils
from epochor.validation.validation import score_time_series_model, ScoreDetails
from epochor.model.storage.disk_model_store import DiskModelStore
from epochor.model.model_constraints import MODEL_CONSTRAINTS_BY_COMPETITION_ID, Competition

from .state import ValidatorState


@dataclasses.dataclass
class PerUIDEvalState:
    """State tracked per UID during a single evaluation run."""
    block: int = math.inf
    hotkey: str = "Unknown"
    repo_name: str = "Unknown"
    score: float = math.inf
    score_details: typing.Dict[str, ScoreDetails] = dataclasses.field(default_factory=dict)


class EvaluationService:
    """
    Acts as the core evaluation engine. It takes a list of UIDs and prepared data,
    retrieves the corresponding models, and executes the scoring function for each.
    """
    def __init__(self, state: ValidatorState, metagraph: "bt.metagraph", local_store: DiskModelStore, device: str, metagraph_lock: threading.RLock):
        """Initializes the EvaluationService."""
        self.state = state
        self.metagraph = metagraph
        self.local_store = local_store
        self.device = device
        self.metagraph_lock = metagraph_lock

    def evaluate_uids(self, uids: list[int], competition: Competition, samples: list, eval_tasks: list, seed: int) -> dict[int, PerUIDEvalState]:
        """Evaluates a list of UIDs, returning their raw scores and performance details."""
        uid_to_state: dict[int, PerUIDEvalState] = defaultdict(PerUIDEvalState)
        for uid in uids:
            with self.metagraph_lock:
                hotkey = self.metagraph.hotkeys[uid]
            uid_to_state[uid].hotkey = hotkey

            model_metadata = self.state.model_tracker.get_model_metadata_for_miner_hotkey(hotkey)
            if model_metadata and model_metadata.id.competition_id == competition.id:
                try:
                    uid_to_state[uid].block = model_metadata.block
                    uid_to_state[uid].repo_name = model_utils.get_hf_repo_name(model_metadata)

                    model_i = self.local_store.retrieve_model(
                        hotkey, model_metadata.id,
                        model_constraints=MODEL_CONSTRAINTS_BY_COMPETITION_ID[competition.id]
                    )

                    score, score_details = utils.run_in_subprocess(
                        functools.partial(score_time_series_model, model_i.model, samples, eval_tasks, self.device, seed),
                        ttl=1800, mode="spawn"
                    )
                    uid_to_state[uid].score = score
                    uid_to_state[uid].score_details = score_details
                    del model_i
                except Exception:
                    logging.error(f"Error during eval for UID {uid}: {traceback.format_exc()}")
                    uid_to_state[uid].score = math.inf

        return uid_to_state
