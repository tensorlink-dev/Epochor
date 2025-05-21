"""
Statistical utilities for Epochor scoring.

Includes win-rate, bootstrap CI, CI-gap scoring, and normalization.
"""

import numpy as np
from typing import Tuple, Union # Added Union for the new return type

def compute_per_round_win_rate(losses: np.ndarray) -> np.ndarray:
    """
    Calculate round-robin win rate for each miner per round.

    Args:
        losses: array of shape (N_miners, T_rounds)

    Returns:
        win_matrix: shape (N_miners, T_rounds), each value ∈ [0, 1]
    """
    N, T = losses.shape
    win_matrix = np.empty((N, T), dtype=float)

    for t in range(T):
        col = losses[:, t]
        matrix = (col[:, None] < col[None, :]).astype(float)
        # Avoid division by zero if N=1 (should not happen in practice with multiple miners)
        win_matrix[:, t] = matrix.sum(axis=1) / (N - 1) if N > 1 else 0.0


    return win_matrix


def compute_overall_win_rate(losses: np.ndarray) -> np.ndarray:
    """
    Average round-robin win rate over all rounds.

    Args:
        losses: shape (N, T)

    Returns:
        overall win rate vector, shape (N,)
    """
    return compute_per_round_win_rate(losses).mean(axis=1)


def bootstrap_ci(data: np.ndarray, B: int = 2000, alpha: float = 0.05) -> Tuple[float, float, float]:
    """
    Estimate bootstrap confidence interval for the mean.

    Args:
        data: 1D array of length T
        B: number of bootstrap replicates
        alpha: significance level (default 5%)

    Returns:
        (mean, lower, upper)
    """
    T = data.shape[0]
    if T < 2:
        # If only one sample, mean is the sample itself, CI is undefined or point estimate.
        # Returning mean and NaNs for bounds or mean itself for bounds might be alternatives.
        # For now, raising an error as CI is not well-defined.
        raise ValueError("Bootstrap requires at least 2 samples.")
    idx = np.random.randint(0, T, size=(B, T))
    samples = data[idx]
    means = samples.mean(axis=1)
    lower = np.percentile(means, 100 * alpha / 2)
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return data.mean(), lower, upper


def compute_ci_bounds(loss_matrix: np.ndarray, B: int = 2000, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap CI bounds for each miner.

    Returns:
        ci_lo, ci_hi: each of shape (N,)
    """
    N, T = loss_matrix.shape
    lo = np.empty(N, dtype=float)
    hi = np.empty(N, dtype=float)

    for i in range(N):
        if T < 2: # Handle cases where a miner might have less than 2 data points
            lo[i] = np.nanmean(loss_matrix[i]) if T > 0 else np.nan
            hi[i] = np.nanmean(loss_matrix[i]) if T > 0 else np.nan
        else:
            _, l, h = bootstrap_ci(loss_matrix[i], B=B, alpha=alpha)
            lo[i] = l
            hi[i] = h
    return lo, hi


def compute_aggregate_gap(ci_lo: np.ndarray, ci_hi: np.ndarray, return_raw_matrix: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute aggregate gap score (how much better each miner's worst-case
    is compared to others' best-case).

    Args:
        ci_lo: array of shape (N,), lower confidence interval bounds.
        ci_hi: array of shape (N,), upper confidence interval bounds.
        return_raw_matrix: If True, returns the raw gap matrix along with aggregate scores.

    Returns:
        agg_gap: array of shape (N,).
        (optional) raw_gap_matrix: array of shape (N, N) if return_raw_matrix is True.
    """
    N = ci_lo.shape[0]
    if N == 0:
        return np.array([]) if not return_raw_matrix else (np.array([]), np.array([[]]))

    # ci_hi[:, None] creates a column vector (N,1)
    # ci_lo[None, :] creates a row vector (1,N)
    # Broadcasting results in an (N,N) matrix where gap[i, j] = ci_hi[i] - ci_lo[j]
    # This represents how much miner i's upper bound exceeds miner j's lower bound.
    # We are interested in the opposite: how much miner i's lower bound (worst case for i)
    # exceeds other miners' upper bounds (best case for others).
    # So, we want ci_lo_i - ci_hi_j. Or, more intuitively, how separated is miner i from j.
    # A positive value in gap_matrix[i,j] would mean miner i's worst case (ci_lo[i])
    # is better than miner j's best case (ci_hi[j]).
    # The sum of these positive differences indicates a strong separation.

    # The provided code was: gap = ci_hi[:, None] - ci_lo[None, :]
    # This calculates gap[i,j] = ci_hi[i] - ci_lo[j].
    # Let's re-evaluate what "aggregate gap" aims to measure.
    # If it's about how much a miner's CI *does not overlap* with others,
    # for miner `i`, we want to see how much `ci_lo[i]` is above `ci_hi[j]` for all `j != i`.
    # Or, how much `ci_hi[i]` is below `ci_lo[j]` for all `j != i`.

    # The current sum `gap.sum(axis=1)` with `gap = ci_hi[:, None] - ci_lo[None, :]` means:
    # For miner `k`, sum_{j} (ci_hi[k] - ci_lo[j]). This doesn't seem right for "separation".

    # Let's consider the definition from the prompt: "how much better each miner's worst-case
    # (ci_lo) is compared to others' best-case (ci_hi)".
    # For a given miner `i`, its worst case is `ci_lo[i]`.
    # For any other miner `j`, its best case is `ci_hi[j]`.
    # The "gap" for miner `i` against miner `j` is `ci_lo[i] - ci_hi[j]`.
    # A larger positive value is better.
    # The raw gap matrix would then be `gap_matrix[i,j] = ci_lo[i] - ci_hi[j]`.
    # And `agg_gap[i] = sum_{j!=i} (ci_lo[i] - ci_hi[j])`.

    raw_gap_matrix = ci_lo[:, None] - ci_hi[None, :]
    np.fill_diagonal(raw_gap_matrix, 0.0) # Set self-gap to 0
    
    # Aggregate gap: sum of how much a miner's lower bound exceeds others' upper bounds.
    # We should probably only sum positive gaps, or handle the interpretation carefully.
    # The existing code `gap.sum(axis=1)` for `gap = ci_hi[:, None] - ci_lo[None, :]`
    # and then `(agg_gap - hi) / (lo - hi)` for normalization (where more negative agg_gap became 1)
    # implies that a more negative sum was better.
    # If `agg_gap = sum(ci_hi_k - ci_lo_j)`, then a smaller sum is better if ci_hi_k is smaller (better loss)
    # and ci_lo_j is larger. This is confusing.

    # Let's stick to the structure of the existing code's computation of `gap` and its sum,
    # as the normalization function `normalize_gap_scores` expects `agg_gap` where "More negative = better".
    # Original: gap = ci_hi[:, None] - ci_lo[None, :]
    # This means gap[row_k, col_j] = ci_hi[k] - ci_lo[j]
    # agg_gap_k = sum_j (ci_hi[k] - ci_lo[j])
    # If miner k is good, its ci_hi[k] is low. If other miners j are bad, their ci_lo[j] are high.
    # So, (low_value - high_value) = very negative. This matches "More negative = better".

    gap_matrix_for_sum = ci_hi[:, None] - ci_lo[None, :]
    np.fill_diagonal(gap_matrix_for_sum, 0.0) # As per original logic
    agg_gap_scores = gap_matrix_for_sum.sum(axis=1)

    if return_raw_matrix:
        # The "raw gap matrix" for debug/visualization should intuitively show miner i vs miner j.
        # Let's define raw_gap_debug[i,j] as ci_lo[i] - ci_hi[j] (how much i's worst is better than j's best)
        # This seems more intuitive for a "gap matrix" display.
        # However, to be consistent with how agg_gap is used for normalization,
        # perhaps the matrix that *led* to agg_gap_scores is more appropriate if "raw" means "intermediate".
        # The prompt says "return raw gap matrix". Let's assume it's the matrix whose sum is `agg_gap_scores`.
        return agg_gap_scores, gap_matrix_for_sum
    else:
        return agg_gap_scores


def normalize_gap_scores(agg_gap: np.ndarray) -> np.ndarray:
    """
    Normalize gap scores into [0, 1].

    More negative = better = closer to 1 after normalization.
    This means that if agg_gap_i is much lower than agg_gap_j, miner i is better.

    Args:
        agg_gap: shape (N,)

    Returns:
        normalized: shape (N,)
    """
    if agg_gap.size == 0:
        return np.array([])
        
    lo, hi = agg_gap.min(), agg_gap.max()
    if np.isclose(lo, hi): # Handles case with one miner or all identical agg_gaps
        return np.ones_like(agg_gap)
    # (agg_gap - hi) makes best scores (most negative) become (most_negative - hi), which is even more negative.
    # (lo - hi) is (smallest_actual_gap - largest_actual_gap), which is negative.
    # Division makes it positive. Largest value becomes (hi - hi) / (lo - hi) = 0.
    # Smallest value becomes (lo - hi) / (lo - hi) = 1.
    # This correctly maps most negative original agg_gap (best) to 1.
    return (agg_gap - hi) / (lo - hi)


__all__ = [
    "compute_overall_win_rate",
    "compute_ci_bounds",
    "compute_aggregate_gap",
    "normalize_gap_scores",
    "compute_per_round_win_rate" # Added this as it was defined but not in __all__
]
