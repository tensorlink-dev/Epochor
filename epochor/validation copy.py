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

import dataclasses
import math
import typing

import taoverse.utilities.logging as logging
import torch
from taoverse.model.competition.epsilon import EpsilonFunc
from taoverse.model.data import Model
from taoverse.model.eval.normalization import normalize_score
from taoverse.model.eval.task import EvalTask

from pretrain.eval.method import (
    EvalMethodId,
    compute_text_loss,
    compute_wer
)
from pretrain.eval.sample import EvalSample



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
 #   uid_to_block: typing.Dict[int, int],
 #   current_block: int,
) -> typing.Tuple[typing.Dict[int, int], typing.Dict[int, float]]:
    """
    Computes the wins and win rate for each model based on loss comparison.

    Parameters:
        uids (list): A list of uids to compare.
        uid_to_score (dict): A dictionary of scores for each uid.
        uid_to_block (dict): A dictionary of blocks for each uid.
        epsilon_func (EpsilonFunc): Function that determines how much advantage to give to the earlier block.
        current_block: The current block.

    Returns:
        tuple: A tuple containing two dictionaries, one for wins and one for win rates.
    """

    mat = np.stack( uid_to_score , axis=0)
  #  block_vec = np.stack(uid_to_block,axis=0 )
  #  diffs = current_block - block_vec    # shape: (N,)



    # Compute components
    try:
        win_rate_arr = compute_overall_win_rate(mat)
        ci_lo, ci_hi = compute_ci_bounds(
            mat, B=EPOCHOR_CONFIG.bootstrap_samples, alpha=EPOCHOR_CONFIG.ci_alpha
        )
        # Assuming compute_aggregate_gap can handle NaNs from ci_bounds if mat had NaNs
        agg_gap_arr = compute_aggregate_gap(ci_lo, ci_hi) 
        sep_score_arr = normalize_gap_scores(agg_gap_arr)
    except Exception as e:
        # If score component computation fails
        # error_message = f"Score components failed: {e}. UIDs: {uids}"
        # Consider logging this error_message
        empty_scores = {uid: 0.0 for uid in uids}
        empty_details = {uid: np.nan for uid in uids}
        return empty_scores, empty_details, empty_details, empty_details, empty_details

    # Composite score
    raw_composite_score_arr = win_rate_arr * sep_score_arr

    # Clean up NaNs for final scores
    final_cleaned_score_arr = np.nan_to_num(raw_composite_score_arr, nan=0.0)

    # Prepare dictionaries for return
    final_scores_dict = {uid: float(final_cleaned_score_arr[i]) for i, uid in enumerate(uids)}
    win_rate_dict = {uid: float(win_rate_arr[i]) if np.isfinite(win_rate_arr[i]) else np.nan for i, uid in enumerate(uids)}
    # Ensure agg_gap_arr is an array before indexing, it might be a tuple if return_raw_matrix was True
    # However, in this context, compute_aggregate_gap is called without return_raw_matrix=True
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
    model: torch.nn.Module,
    series: typing.Union[np.ndarray, torch.Tensor],
    evaluator: CRPSEvaluator,
    device: str,
) -> float:
    """
    Runs a time-series model on the given input and returns its loss.

    Args:
        model (torch.nn.Module): The time-series model (e.g., a PyTorch nn.Module) 
            that takes `series` as input and returns predictions of the same shape.
        series (np.ndarray or torch.Tensor): The input time-series batch. 
            Should already be in the format the model expects (e.g., shape [batch, features, ...]).
        evaluator (CRPSEvaluator): An instance that knows how to compute loss between
            the true series and the model’s predictions.
        device (str): Device identifier, e.g. "cpu" or "cuda:0".

    Returns:
        float: The computed loss (e.g., CRPS) for this forward pass.
    """
    # Move model to the correct device and set eval mode
    model.to(device)
    model.eval()

    # Convert series to a torch.Tensor on `device` if not already
    if not isinstance(series, torch.Tensor):
        series_tensor = torch.tensor(series, dtype=torch.float32, device=device)
    else:
        series_tensor = series.to(device)

    with torch.inference_mode():
        # Forward pass
        preds = model(series_tensor)
        # Compute loss via the evaluator
        loss = evaluator[task.name](series_tensor, preds) 
        # replace
        score_details[task.name] = ScoreDetails(
                raw_score= loss ,
                norm_score=None,
                weighted_norm_score=None,
                num_samples=len(series),
            )

    return loss.mean(),score_details 

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

    # Iterate through the models and only keep models who's loss is better than
    # all models uploaded at an earlier block, after they've fully decayed.
    # If the model cannot, then there exists at least one model at an earlier block which
    # will always have a better epislon adjusted loss, thus it will never be the top model.
    competitive_uids = []
    for uid, loss in uid_to_score.items():
        # Check if the current UID beats all earlier (or same block) models at full decay.
        # all([]) is true so we always keep the earliest model.
        earlier_uids = [
            i
            for i, block in uid_to_block.items()
            if i != uid and block <= uid_to_block[uid]
        ]
        if all(loss < fully_decayed_scores[uid_other] for uid_other in earlier_uids):
            competitive_uids.append(uid)

    return competitive_uids