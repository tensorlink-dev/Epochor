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

# Tools for performing validation over models.
from typing import Any, Dict, List, Tuple

import dataclasses
import math
import typing
import numpy as np

from epochor.utils import logging
import torch
from competitions.epsilon import EpsilonFunc
from epochor.model.model_data import Model
from epochor.evaluation.eval_task import EvalTask
from epochor.config import EPOCHOR_CONFIG
from epochor.evaluation.evaluation import BaseEvaluator
from epochor.validation.statistics import (
    compute_overall_win_rate,
    compute_ci_bounds,
    compute_aggregate_gap,
    normalize_gap_scores,
)
from epochor.evaluation.evaluation import EVALUATION_BY_COMPETITION 



@dataclasses.dataclass
class ScoreDetails:
    """Details of the score for a model."""

    raw_score: typing.Optional[float] = None
    norm_score: typing.Optional[float] = None
    weighted_norm_score: typing.Optional[float] = None
    num_samples: int = 0



def compute_scores(
    uids: typing.List[int],
    uid_to_score: typing.Dict[int, float],
) -> typing.Dict[str, typing.Dict[int, float]]:  # Changed return type hint
    """
    Computes the wins and win rate for each model based on loss comparison.

    Parameters:
        uids (list): A list of uids to compare.
        uid_to_score (dict): A dictionary of scores for each uid.

    Returns:
        tuple: A tuple containing two dictionaries, one for wins and one for win rates.
    """

    mat = np.stack(list(uid_to_score.values()), axis=0)

    # Compute components
    try:
        win_rate_arr = compute_overall_win_rate(mat)
        ci_lo, ci_hi = compute_ci_bounds(
            mat, B=EPOCHOR_CONFIG.bootstrap_samples, alpha=EPOCHOR_CONFIG.ci_alpha
        )
        agg_gap_arr = compute_aggregate_gap(ci_lo, ci_hi) 
        sep_score_arr = normalize_gap_scores(agg_gap_arr)
    except Exception as e:
        logging.error(f"Error computing scores: {e}")
        return  {
            "final_scores_dict": {uid: 0.0 for uid in uids},
            "win_rate_dict": {uid: 0.0 for uid in uids},
            "agg_gap_dict": {uid: np.nan for uid in uids},
            "sep_score_dict": {uid: np.nan for uid in uids},
            "raw_composite_score_dict": {uid: np.nan for uid in uids},
        }

    # Composite score
    raw_composite_score_arr = win_rate_arr * sep_score_arr

    # Clean up NaNs for final scores
    final_cleaned_score_arr = np.nan_to_num(raw_composite_score_arr, nan=0.0)

    # Prepare dictionaries for return
    final_scores_dict = {uid: float(final_cleaned_score_arr[i]) for i, uid in enumerate(uids)}
    win_rate_dict = {uid: float(win_rate_arr[i]) if np.isfinite(win_rate_arr[i]) else np.nan for i, uid in enumerate(uids)}
    agg_gap_dict = {uid: float(agg_gap_arr[i]) if np.isfinite(agg_gap_arr[i]) else np.nan for i, uid in enumerate(uids)}
    sep_score_dict = {uid: float(sep_score_arr[i]) if np.isfinite(sep_score_arr[i]) else np.nan for i, uid in enumerate(uids)}
    raw_composite_score_dict = {uid: float(raw_composite_score_arr[i]) if np.isfinite(raw_composite_score_arr[i]) else np.nan for i, uid in enumerate(uids)}

    return  {
        "final_scores_dict": final_scores_dict,
        "win_rate_dict": win_rate_dict,
        "agg_gap_dict": agg_gap_dict,
        "sep_score_dict": sep_score_dict,
        "raw_composite_score_dict": raw_composite_score_dict,
    }

def score_time_series_model(
    model: Any,
    samples: List[List[Dict[str, Any]]],  # List of batches per task
    eval_tasks: List[Any],  # List of EvalTask objects
    device: str,
    task_or_seed: Any,
) -> Tuple[float, Dict[str, "ScoreDetails"]]:  # Assuming ScoreDetails is defined elsewhere
    model.to(device)
    model.eval()

    all_losses = []
    score_details = {}

    # Correctly unpack both eval_task and corresponding task_batches
    for eval_task, task_batches in zip(eval_tasks, samples):
        EvaluatorClass = EVALUATION_BY_COMPETITION[eval_task.method_id.value]
        evaluator = EvaluatorClass()

        for batch in task_batches:  # Not enumerate here, unless you use `i`
            inputs = batch["inputs_padded"].to(device)
            targets = batch["targets_padded"].to(device)
            forecast_len = batch['actual_target_lengths'][0]
            targets = targets[:forecast_len]
            logging.debug(f'forecast len : {forecast_len}')

            with torch.inference_mode():
                preds = model.forecast(
                    inputs=batch["inputs_padded"].unsqueeze(-1),
                    prediction_length=forecast_len,
                    attention_mask=batch["attention_mask"],
                )
                losses = evaluator.evaluate(targets.cpu().numpy(), preds.cpu().numpy())  # [B] or [B, T]
                all_losses.append(losses)

    # Flatten and compute mean
    all_losses_flat = (
        np.concatenate(all_losses)
        if all_losses and isinstance(all_losses[0], np.ndarray)
        else np.array(all_losses)
    )
    mean_score = float(np.mean(all_losses_flat))

    score_details["flat_evaluation"] = ScoreDetails(
        raw_score=all_losses_flat,
        norm_score=None,
        weighted_norm_score=None,
        num_samples=len(all_losses_flat),
    )

    return mean_score, score_details


def compute_competitive_uids(
    uid_to_score: typing.Dict[int, float],
    uid_to_block: typing.Dict[int, int],
    epsilon_func: EpsilonFunc,
) -> typing.List[int]:
    """
    Computes the list of any uids that may at one point be the top model.

    Parameters:
        uid_to_score (dict): A dictionary of score for each uid over all batches.
        uid_to_block (dict): A dictionary of blocks for each uid.
        epsilon_func (EpsilonFunc): Function that determines how much advantage to give to the earlier block.

    Returns:
        list: A list of uids that may at one point be the top model.
    """
    # Get fully decayed loss for every model.
    fully_decayed_epsilon = 1 - epsilon_func.compute_epsilon(
        current_block=math.inf, model_block=0
    )
    fully_decayed_scores = {
        uid: uid_to_score[uid] * fully_decayed_epsilon for uid in uid_to_block
    }

    competitive_uids = []
    for uid, loss in uid_to_score.items():
        earlier_uids = [
            i
            for i, block in uid_to_block.items()
            if i != uid and block <= uid_to_block[uid]
        ]
        if all(loss < fully_decayed_scores[uid_other] for uid_other in earlier_uids):
            competitive_uids.append(uid)

    return competitive_uids
