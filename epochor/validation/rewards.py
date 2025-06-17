# epochor/rewards.py

"""
Convert composite scores into normalized on-chain weights.

Supports exponentiation, first-place boost, and NaN protection.
"""

import numpy as np
from typing import Dict, Callable
from epochor.config import EPOCHOR_CONFIG
from epochor.utils.logging import reinitialize

logger = reinitialize()


def softmax_rewards(scores: Dict[int, float], temperature: float) -> Dict[int, float]:
    """Calculates weights using a temperature-scaled softmax."""
    uids = list(scores.keys())
    vals = np.array([scores[uid] for uid in uids], dtype=float)
    
    if temperature == 0: # Avoid division by zero, treat as argmax
        weights = np.zeros_like(vals)
        weights[np.argmax(vals)] = 1.0
        return {uid: float(w) for uid, w in zip(uids, weights)}

    vals_scaled = vals / temperature
    exp_scores = np.exp(vals_scaled - np.max(vals_scaled))  # Subtract max for numerical stability
    weights = exp_scores / np.sum(exp_scores)
    return {uid: float(w) for uid, w in zip(uids, weights)}

def linear_rewards(scores: Dict[int, float]) -> Dict[int, float]:
    """Calculates weights directly proportional to scores."""
    uids = list(scores.keys())
    vals = np.array([scores[uid] for uid in uids], dtype=float)
    total_score = np.sum(vals)
    if total_score == 0:
        num_uids = len(uids)
        return {uid: 1.0 / num_uids if num_uids > 0 else 0 for uid in uids}
    weights = vals / total_score
    return {uid: float(w) for uid, w in zip(uids, weights)}

def ranked_decay_rewards(scores: Dict[int, float], decay_factor: float = 0.9) -> Dict[int, float]:
    """Calculates weights based on rank with a decay factor."""
    if not scores:
        return {}
    
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    uids = [item[0] for item in sorted_scores]
    
    ranks = np.arange(len(uids))
    decayed_values = decay_factor ** ranks
    
    total_decayed_value = np.sum(decayed_values)
    if total_decayed_value == 0: # Should not happen with decay_factor > 0
        num_uids = len(uids)
        return {uid: 1.0 / num_uids if num_uids > 0 else 0 for uid in uids}

    weights = decayed_values / total_decayed_value
    return {uid: float(w) for uid, w in zip(uids, weights)}


REWARD_STRATEGIES: Dict[str, Callable[[Dict[int, float]], Dict[int, float]]] = {
    "softmax": lambda s: softmax_rewards(s, EPOCHOR_CONFIG.reward_temperature),
    "linear": linear_rewards,
    "ranked_decay": lambda s: ranked_decay_rewards(s), # Add decay_factor to config if needed
}


def allocate_rewards(scores: Dict[int, float]) -> Dict[int, float]:
    """
    Args:
        scores: uid → score dict

    Returns:
        uid → weight dict
    """
    uids = list(scores.keys())
    raw_vals = np.array([scores[uid] for uid in uids], dtype=float)

    # 1) Replace NaNs and clip negatives
    processed_vals = np.nan_to_num(raw_vals, nan=0.0)
    processed_vals = np.clip(processed_vals, 0.0, None)
    
    processed_scores = {uid: float(v) for uid, v in zip(uids, processed_vals)}

    # 2) Select reward strategy
    strategy_func = REWARD_STRATEGIES.get(EPOCHOR_CONFIG.reward_strategy)
    if not strategy_func:
        logger.warning(f"Unknown reward strategy: {EPOCHOR_CONFIG.reward_strategy}. Defaulting to softmax.")
        strategy_func = lambda s: softmax_rewards(s, EPOCHOR_CONFIG.reward_temperature)
    
    weights_dict = strategy_func(processed_scores)
    
    weights_uids = list(weights_dict.keys())
    weights_vals = np.array([weights_dict[uid] for uid in weights_uids], dtype=float)

    # 3) Optional exponentiation (can be applied before or after strategy, depending on desired effect)
    # Applying *after* strategy, on the weights themselves.
    if EPOCHOR_CONFIG.reward_exponent != 1.0:
        weights_vals = np.power(weights_vals, EPOCHOR_CONFIG.reward_exponent)
        # Renormalize after exponentiation
        total_weight = np.sum(weights_vals)
        if total_weight > 0:
            weights_vals = weights_vals / total_weight
        else: # Handle case where all weights become zero (e.g. exponentiating zeros)
            logger.warning("All weights became zero after exponentiation. Defaulting to uniform weights.")
            num_weights = len(weights_vals)
            weights_vals = np.ones(num_weights) / num_weights if num_weights > 0 else np.array([])


    # 4) Optional first-place boost (applied to the weights from the strategy)
    if EPOCHOR_CONFIG.first_place_boost > 1.0 and weights_vals.size > 0:
        # Find the UID that had the max *original* score to boost its *current* weight
        if processed_vals.size > 0:
            original_max_score_idx = int(np.nanargmax(processed_vals)) # Index in processed_vals
            boost_uid = uids[original_max_score_idx] # UID to boost

            if boost_uid in weights_dict:
                # Find the index of this UID in the current weights_vals array
                try:
                    boost_idx = weights_uids.index(boost_uid)
                    logger.info(f"Boosting UID {boost_uid} (original max score) by {EPOCHOR_CONFIG.first_place_boost}x")
                    weights_vals[boost_idx] *= EPOCHOR_CONFIG.first_place_boost
                    
                    # Renormalize after boost
                    total_weight = np.sum(weights_vals)
                    if total_weight > 0:
                        weights_vals = weights_vals / total_weight
                    else:
                        logger.warning("Total weight is zero after boost. Cannot normalize.")
                except ValueError:
                    logger.warning(f"UID {boost_uid} not found in weights_dict for boosting. This shouldn't happen.")
            else:
                logger.warning(f"UID {boost_uid} for boosting not found in current weights dictionary.")


    # 5) Apply min weight thresholding and re-normalize
    if EPOCHOR_CONFIG.min_weight_threshold > 0 and weights_vals.size > 0:
        below_threshold_mask = weights_vals < EPOCHOR_CONFIG.min_weight_threshold
        weights_vals[below_threshold_mask] = EPOCHOR_CONFIG.min_weight_threshold
        
        # Re-normalize so that the sum of weights is 1
        total_weight = np.sum(weights_vals)
        if total_weight > 0:
            weights_vals = weights_vals / total_weight
        else:
            logger.warning("Total weight is zero after applying min_weight_threshold. Cannot normalize.")


    # 6) Final check for normalization and create final dict
    final_total = np.sum(weights_vals)
    if not np.isclose(final_total, 1.0) and weights_vals.size > 0 :
        logger.warning(f"Final weights do not sum to 1 (sum={final_total}). Re-normalizing as a final step.")
        if final_total > 0 :
             weights_vals = weights_vals / final_total
        else: # Fallback to uniform if sum is still zero
            logger.error("Sum of weights is still zero. Defaulting to uniform weights for safety.")
            num_weights = len(weights_vals)
            weights_vals = np.ones(num_weights) / num_weights if num_weights > 0 else np.array([])


    return {uid: float(w) for uid, w in zip(weights_uids, weights_vals)}
