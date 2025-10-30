import json
from typing import Dict, Any

def get_arch_dict(config) -> Dict[str, Any]:
    """Creates a canonical dictionary representing the model architecture."""
    # 1) Convert to a dict
    cfg_dict = config.to_dict()
    # 2) Drop non-architecture fields
    for k in ("type", "model_type", "output_attentions", "output_hidden_states", "use_cache"):
        cfg_dict.pop(k, None)
    return cfg_dict

def get_architecture_diff_score(config_a, config_b) -> int:
    """
    Calculates the number of differing architectural parameters between two models.
    A score of 0 means the architectures are identical.

    Returns:
        int: The count of parameters that are different.
    """
    dict_a = get_arch_dict(config_a)
    dict_b = get_arch_dict(config_b)

    # Find all unique keys from both dictionaries
    all_keys = set(dict_a.keys()) | set(dict_b.keys())
    
    diff_count = 0
    for key in all_keys:
        # If values are different or a key is missing in one, increment diff count
        if dict_a.get(key) != dict_b.get(key):
            diff_count += 1
            
    return diff_count


    import numpy as np

def apply_copy_penalty(
    base_score: np.ndarray,   # shape (N,)
    time_lt:    np.ndarray,   # lower-triangular age-diff matrix (age_i – age_j for i>j)
    sim_lt:     np.ndarray,   # lower-triangular similarity matrix (0 ⇒ identical, >0 ⇒ different)
    P:          float         # flat penalty factor for any copy, e.g. 0.5
) -> np.ndarray:
    """
    - Any model that is a younger copy (identical sim==0 and age_i < age_j)
      gets penalty = P.
    - All originals & unique models get penalty = 1.
    Final score = base_score * penalty.
    """
    N = base_score.shape[0]

    # Reconstruct full matrices
    time_full = time_lt - time_lt.T
    sim_full  = sim_lt + sim_lt.T

    # Detect identical off-diagonal pairs
    eye       = np.eye(N, dtype=bool)
    identical = (sim_full == 0) & (~eye)

    # Flag any model that’s the younger half of an identical pair
    is_copy = (identical & (time_full < 0)).any(axis=1)

    # Build penalty vector
    penalty = np.where(is_copy, P, 1.0)

    # Apply
    return base_score * penalty
