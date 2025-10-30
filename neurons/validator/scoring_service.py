"""Transforms raw evaluation scores into ranked weights and determines which models to keep."""

import torch
import math
import typing

import bittensor as bt
import constants
from epochor.model.model_data import EvalResult
from epochor.validation.validation import compute_scores, ScoreDetails
from epochor.model.model_constraints import Competition

from .state import ValidatorState
from .evaluation_service import PerUIDEvalState


class ScoringService:
    """
    Functions as the 'judge' that turns raw performance into ranked scores.
    It applies ranking logic, updates EMA trackers, and calculates final weights.
    """
    def __init__(self, state: ValidatorState, metagraph: "bt.metagraph", config):
        """Initializes the ScoringService."""
        self.state = state
        self.metagraph = metagraph
        self.config = config

    def process_scores_and_update_weights(
        self,
        uids: list[int],
        uid_to_state: dict[int, PerUIDEvalState],
        competition: Competition,
        cur_block: int
    ):
        """
        Takes raw scores, calculates weights, updates trackers, and determines which models to keep.
        
        Returns:
            A tuple containing (scoring_metrics, models_to_keep).
        """
        uid_to_raw_losses = {
            uid: state.score_details.get("flat_evaluation", ScoreDetails()).raw_score
            for uid, state in uid_to_state.items()
        }
        uid_to_raw_losses = {uid: losses for uid, losses in uid_to_raw_losses.items() if losses is not None and len(losses) > 0}

        scoring_metrics = compute_scores(uids, uid_to_raw_losses)

        scores_for_ema = {uid: scoring_metrics["final_scores_dict"][uid] for uid in uids}
        uid_to_hotkey = {uid: self.metagraph.hotkeys[uid] for uid in uids}
        self.state.update_ema_scores(scores_for_ema, competition.id, cur_block, uid_to_hotkey)

        scores = self.state.ema_tracker.get(competition.id)
        top_uid = max(scores, key=scores.get, default=None)
        if top_uid is not None:
            self._record_eval_results(top_uid, cur_block, uid_to_state, competition.id)

        model_weights = torch.tensor([scores.get(uid, 0) for uid in range(self.metagraph.n)], dtype=torch.float32)
        step_weights = torch.softmax(model_weights / constants.temperature, dim=0)
        self.state.ema_tracker.record_competition_weights(competition.id, step_weights)

        models_to_keep = self._determine_models_to_keep(uids, uid_to_state, scoring_metrics, competition.id)
        return scoring_metrics, models_to_keep

    def _record_eval_results(self, top_uid: int, curr_block: int, uid_to_state: dict[int, PerUIDEvalState], competition_id: int):
        """Records the outcome of an evaluation for each participating model."""
        top_model_score = uid_to_state[top_uid].score
        for _, state in uid_to_state.items():
            eval_result = EvalResult(
                block=curr_block, score=state.score,
                winning_model_block=uid_to_state[top_uid].block, winning_model_score=top_model_score
            )
            self.state.model_tracker.on_model_evaluated(state.hotkey, competition_id, eval_result)
    
    def _determine_models_to_keep(self, uids, uid_to_state, scoring_metrics, competition_id):
        """Determines which UIDs should be kept for the next evaluation round based on performance."""
        win_rate = scoring_metrics.get('win_rate_dict', {})
        tracker_weights = self.state.ema_tracker.get_competition_weights(competition_id)
        
        model_prioritization = {
            uid: (1 + tracker_weights[uid].item()) if uid < len(tracker_weights) and tracker_weights[uid].item() >= 0.001 else win_rate.get(uid, 0.0)
            for uid in uids
        }
        
        models_to_keep = set(sorted(model_prioritization, key=model_prioritization.get, reverse=True)[: self.config.sample_min])
        
        if len(models_to_keep) < self.config.sample_min:
            uid_to_avg_score = {uid: state.score for uid, state in uid_to_state.items()}
            for uid in sorted(uid_to_avg_score, key=uid_to_avg_score.get):
                if len(models_to_keep) >= self.config.sample_min:
                    break
                models_to_keep.add(uid)

        return models_to_keep
