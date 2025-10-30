# """
# Statistical utilities for Epochor scoring.

# Includes win-rate, bootstrap CI, CI-gap scoring, and normalization.
# """

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
    # Initialize with NaNs to clearly distinguish from a zero win rate.
    win_matrix = np.full((N, T), np.nan, dtype=float)

    for t in range(T):
        col = losses[:, t]
        # NaN-awareness: Create a mask for valid (non-NaN) scores in the current round.
        valid_indices = ~np.isnan(col)

        # Only proceed if there are at least two miners with valid scores to compare.
        if np.sum(valid_indices) < 2:
            win_matrix[valid_indices, t] = 0.0  # If no one to compete against, win rate is 0.
            continue
            
        valid_scores = col[valid_indices]
        
        # Perform pairwise comparison only on the valid scores.
        matrix = (valid_scores[:, None] < valid_scores[None, :]).astype(float)
        
        wins = matrix.sum(axis=1)
        num_competitors = len(valid_scores) - 1
        
        # Assign the calculated win rate back to the original positions in the win_matrix.
        win_matrix[valid_indices, t] = wins / num_competitors if num_competitors > 0 else 0.0

    return win_matrix


def compute_overall_win_rate(losses: np.ndarray) -> np.ndarray:
    """
    Average round-robin win rate over all rounds.

    Args:
        losses: shape (N, T)

    Returns:
        overall win rate vector, shape (N,)
    """
    per_round_wins = compute_per_round_win_rate(losses)
    # NaN-awareness: Use np.nanmean to average only the valid (non-NaN) rounds for each miner.
    with np.errstate(invalid='ignore'): # Suppress warnings for rows that are all NaN
        overall_wins = np.nanmean(per_round_wins, axis=1)
    # If a miner had no valid rounds, their nanmean will be NaN. Default this to 0.0.
    return np.nan_to_num(overall_wins, nan=0.0)


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
    # NaN-awareness: Filter out NaNs from the data before performing bootstrapping.
    valid_data = data[~np.isnan(data)]
    T = valid_data.shape[0]

    if T < 2:
        # Instead of raising ValueError, return NaNs for CI bounds
        mean_val = np.mean(valid_data) if T > 0 else np.nan
        return mean_val, np.nan, np.nan # Return mean, nan, nan for consistency
        
    # Sample with replacement from only the valid data points.
    idx = np.random.randint(0, T, size=(B, T))
    samples = valid_data[idx]
    
    means = np.mean(samples, axis=1)
    
    lower = np.percentile(means, 100 * alpha / 2)
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return np.mean(valid_data), lower, upper


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
        # Now bootstrap_ci handles T < 2 and NaNs, no need for separate checks here.
        # Just directly unpack.
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

    # This calculation remains the same, as it defines the gap metric.
    # A good miner 'i' has a low ci_hi[i]. A bad miner 'j' has a high ci_lo[j].
    # The difference ci_hi[i] - ci_lo[j] will be very negative for good miners.
    # This matches the expectation that "More negative = better" for the normalization function.
    gap_matrix_for_sum = ci_hi[:, None] - ci_lo[None, :]
    
    # NaN-awareness: Ignore self-comparison by setting diagonal to NaN before averaging.
    np.fill_diagonal(gap_matrix_for_sum, np.nan)
    
    # Use nanmean to correctly average the gaps, ignoring NaN values from self-comparison
    # and any potential NaNs from the CI bounds themselves.
    with np.errstate(invalid='ignore'):
        agg_gap_scores = np.nanmean(gap_matrix_for_sum, axis=1)
    
    # Replace any resulting NaNs (e.g., if a miner had no one to compare against) with 0.
    final_scores = np.nan_to_num(agg_gap_scores, nan=0.0)

    if return_raw_matrix:
        return final_scores, gap_matrix_for_sum
    else:
        return final_scores


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
    # Handle empty or single-miner cases.
    if agg_gap.size < 2:
        return np.ones_like(agg_gap)
        
    # NaN-awareness: Use nanmin and nanmax to find the range, ignoring any NaNs.
    lo, hi = np.nanmin(agg_gap), np.nanmax(agg_gap)
    
    # If all values are the same (or all NaN), return 1s.
    if np.isclose(lo, hi) or np.isnan(lo):
        return np.ones_like(agg_gap)
        
    # (agg_gap - hi) / (lo - hi) maps the range [lo, hi] to [1, 0].
    # This correctly maps the most negative original agg_gap (best) to 1.
    norm_scores = (agg_gap - hi) / (lo - hi)
    
    # Convert any remaining NaNs (e.g., from miners who had no valid scores) to 0 (the worst score).
    return np.nan_to_num(norm_scores, nan=0.0)


__all__ = [
    "compute_overall_win_rate",
    "compute_ci_bounds",
    "compute_aggregate_gap",
    "normalize_gap_scores",
    "compute_per_round_win_rate" # Added this as it was defined but not in __all__
]