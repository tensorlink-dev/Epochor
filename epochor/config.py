# epochor/config.py

"""
Global configuration schema for Epochor subnet.

Tunable parameters used by validators, evaluators, reward allocators.
"""

class EPOCHOR_CONFIG:
    # Evaluation
    series_length = 500
    bootstrap_samples = 2000
    ci_alpha = 0.05

    # Generator sampling weights for each category
    samplers = []
    registries = []
    category_weights = [0.5, 0.3, 0.2, 0.0]

    # WandB logging
    use_wandb = True
    wandb_project = "epochor"
    wandb_entity = "tensorlink"

    # Subnet info
    netuid = 999

    # Reward shaping
    reward_exponent = 2.0           # sharpens reward distribution
    first_place_boost = 1.2         # 20% boost for top scorer
    reward_temperature = 0.5        # Scales scores before softmax
    min_weight_threshold = 0.001    # Minimum normalized weight
    reward_strategy = "softmax"     # "softmax", "linear", "ranked_decay"

    # EMA smoothing (if used)
    ema_span = 10                   # decay window for score smoothing

    # Batch limits (optional)
    max_models_per_round = 256