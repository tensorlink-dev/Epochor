"""Manages the 'what' and 'when' of evaluations: selecting competitions and loading data."""

import typing
import logging
import random

from competitions import competitions
from epochor.utils import competition_utils
from epochor.datasets.dataloaders import DatasetLoaderFactory
from epochor.model.model_constraints import Competition

from .state import ValidatorState


class CompetitionManager:
    """
    Manages the selection of competitions and the loading of their associated data.
    """
    def __init__(self, state: ValidatorState):
        """Initializes the CompetitionManager."""
        self.state = state

    def get_next_competition(self, global_step: int, block: int) -> typing.Optional[Competition]:
        """Determines the next competition to evaluate based on the current block and schedule."""
        schedule = competition_utils.get_competition_schedule_for_block(
            block=block,
            schedule_by_block=competitions.COMPETITION_SCHEDULE_BY_BLOCK,
        )
        return schedule[global_step % len(schedule)] if schedule else None
    
    def prepare_uids_for_eval(self, competition_id: int) -> typing.List[int]:
        """Moves pending UIDs to the active evaluation queue for a competition."""
        with self.state.pending_uids_to_eval_lock:
            pending = self.state.pending_uids_to_eval.get(competition_id, set())
            self.state.uids_to_eval[competition_id].update(pending)
            self.state.pending_uids_to_eval[competition_id].clear()
        return list(self.state.uids_to_eval.get(competition_id, set()))

    def load_data_for_competition(self, competition: Competition, seed: int) -> tuple[list, list]:
        """Loads and prepares all necessary data batches for a given competition."""
        eval_tasks = []
        all_samples = []
        for eval_task in competition.eval_tasks:
            try:
                loader = DatasetLoaderFactory.get_loader(
                    dataset_id=eval_task.dataset_id,
                    dataset_kwargs=eval_task.dataset_kwargs,
                    seed=seed,
                )
                batches = list(loader)
                if batches:
                    random.Random(seed).shuffle(batches)
                    eval_tasks.append(eval_task)
                    all_samples.append(batches)
            except Exception as e:
                logging.error(f"Error loading data for task {eval_task.name}: {e}")
        
        return eval_tasks, all_samples
