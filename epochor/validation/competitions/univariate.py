# epochor/validation.py

"""
Validation logic for Epochor subnet.

Computes composite score = win_rate × normalized CI-gap.
"""

import numpy as np
from typing import Dict, List, Tuple # Added Tuple
from epochor.config import EPOCHOR_CONFIG
from epochor.statistics import (
    compute_overall_win_rate,
    compute_ci_bounds,
    compute_aggregate_gap,
    normalize_gap_scores,
)

# Define a type alias for the return structure for clarity
ValidationResult = Tuple[
    Dict[int, float],  # Final cleaned scores
    Dict[int, float],  # Raw win rates
    Dict[int, float],  # Raw aggregate gap (before normalization)
    Dict[int, float],  # Normalized separation score (CI-gap based)
    Dict[int, float]   # Raw composite score (win_rate * sep_score, before cleaning)
]

def validate(loss_history: Dict[int, List[float]]) -> ValidationResult:
    """
    Given a dictionary of uid → list of past losses,
    compute composite scores using round-robin win rate and CI-gap separation.

    Returns:
        A tuple containing five dictionaries, each mapping UID to:
        1. Final cleaned score.
        2. Raw win rate.
        3. Raw aggregate gap score (before normalization).
        4. Normalized separation score.
        5. Raw composite score (before NaN cleaning).
    """
    uids = list(loss_history.keys())
    if not uids:
        empty_dict = {}
        return empty_dict, empty_dict, empty_dict, empty_dict, empty_dict

    histories = [loss_history[uid] for uid in uids]
    
    # Determine max_len carefully, handle empty histories if any uid has one
    valid_histories = [h for h in histories if h] # Filter out empty lists
    if not valid_histories: # All histories are empty or uids had no history
        empty_scores = {uid: 0.0 for uid in uids}
        empty_details = {uid: np.nan for uid in uids} # Use NaN for raw unavailable data
        return empty_scores, empty_details, empty_details, empty_details, empty_details

    max_len = max(len(h) for h in valid_histories) if valid_histories else 0
    
    padded_histories = []
    for h in histories:
        if not h: # If original history was empty
            padded_histories.append([0] * max_len if max_len > 0 else []) # Pad with NaNs
        else:
            padded_histories.append(h + [np.nanmean(h)] * (max_len - len(h)))
            
    try:
        # Ensure all lists in padded_histories have the same length max_len
        # If max_len is 0 (e.g. all histories were empty and valid_histories was empty)
        # np.stack will fail.
        if max_len == 0 and uids: # if uids exist but no valid history to form a matrix
             # This case implies all UIDs had empty histories. Return NaNs for scores.
            empty_scores = {uid: 0.0 for uid in uids}
            empty_details = {uid: np.nan for uid in uids}
            return empty_scores, empty_details, empty_details, empty_details, empty_details
        elif not uids: # Should have been caught earlier, but as a safeguard
            empty_dict = {}
            return empty_dict, empty_dict, empty_dict, empty_dict, empty_dict
            
        mat = np.stack(padded_histories, axis=0)
    except Exception as e:
        # If matrix building fails, return default/empty values for all UIDs
        # error_message = f"Failed to build loss matrix: {e}. UIDs: {uids}, Histories: {histories}"
        # Consider logging this error_message
        empty_scores = {uid: 0.0 for uid in uids}
        empty_details = {uid: np.nan for uid in uids}
        return empty_scores, empty_details, empty_details, empty_details, empty_details

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

    return (
        final_scores_dict,
        win_rate_dict,
        agg_gap_dict,
        sep_score_dict,
        raw_composite_score_dict,
    )
