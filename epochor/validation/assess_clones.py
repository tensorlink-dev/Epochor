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