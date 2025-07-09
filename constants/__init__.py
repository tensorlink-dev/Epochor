
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const

# Fomula for converting TAO to an integer:
#
# old_balance = self.subtensor.get_balance( self.wallet.coldkey.ss58_address )
# new_balance = old_balance.tao * 10**9

import os
import torch
from typing import List, Tuple
from keylimiter import KeyLimiter
from datetime import timedelta
from enum import IntEnum

class CompetitionId(IntEnum):
    """
    Defines the different competition tracks available in the Epochor subnet.
    Validators and miners can use this to tailor their behavior based on the
    active competition.
    """
    UNIVARIATE = 0
    UNIVARIATE_COVARS = 1
    MULTIVARIATE =  2

    def __repr__(self) -> str:
        return f"{self.value}"

# Local development mode.
IS_LOCAL_DEVELOPMENT_MODE = bool(os.environ.get("IS_LOCAL_DEVELOPMENT_MODE", False))
LOCAL_MODE_NEURONS_COUNT = int(os.environ.get("LOCAL_MODE_NEURONS_COUNT", 10))

# The minimum stake required to be sync'd with the network.
MIN_SYNC_STAKE = 10_000

# The number of models to keep in the cache.
MAX_MODELS_TO_CACHE = 10

# URL of the weights repo.
WEIGHTS_REPO_URL = "https://huggingface.co/makemoret/model_weights"

# Filename for the weights.
WEIGHTS_FILENAME = "distilbert-base-uncased.pt"

# The version of the validator.
weights_version_key = 2002

# The number of neurons in the subnet.
SUBNET_NEURONS = 256

# The number of runs in a pass on the network.
PASS_THROUGH_RUNS = 2

# The batch size for processing data.
BATCH_SIZE = 32

# The sequence length of the data.
SEQUENCE_LENGTH = 1024

# The time (in seconds) to wait before giving up on a validation request.
VALIDATION_REQUEST_TIMEOUT = 10

# The time (in seconds) to wait before giving up on a ranking request.
RANKING_REQUEST_TIMEOUT = 10

# The number of top miners to sync with.
WEIGHT_SYNC_MINER_MIN_PERCENT = 0.8
WEIGHT_SYNC_VALI_MIN_STAKE = 10_000

# The cadence at which to check for chain updates.
chain_update_cadence = timedelta(minutes=1)

# The cadence at which to check the top models on the network.
scan_top_model_cadence = timedelta(minutes=30)

# The cadence at which to retry a model that was previously discarded.
model_retry_cadence = 7200  # 1 hour.

alpha = 0.05
# The limit on the number of updated models to hold in memory.
updated_models_limit = 10

def get_list_of_uids(
    pass_through_config: int, self_uid: int, metagraph: "bt.metagraph.Metagraph"
) -> List[int]:
    """
    In the event that the pass through config is set to 0, we want to query all neurons.
    Otherwise, we want to query a subset of the neurons.

    Args:
        pass_through_config (int): The number of neurons to query.
        self_uid (int): The UID of the validator.
        metagraph (bt.metagraph.Metagraph): The metagraph of the subnet.

    Returns:
        List[int]: The list of UIDs to query.
    """
    if pass_through_config == 0:
        return [uid for uid in range(len(metagraph.hotkeys))]
    else:
        # Create a list of UIDs to query, respecting the pass_through_config.
        # This will query in sequence and round-robin through the UIDs.
        # Note that this is not parallel, but sequential.
        # This is to avoid overloading the machine with too many requests at once.
        # The order of the UIDs is determined by the last time they were queried.
        # This is to ensure that all UIDs are queried equally over time.
        return [
            uid
            for uid in range(len(metagraph.hotkeys))
            if uid % pass_through_config == self_uid % pass_through_config
        ]
import os

# Base project settings
WANDB_PROJECT         = "epocher-validator"       # your wandb project name
SUBNET_UID            = 2                    # your Bittensor subnet UID

# Validator-specific defaults
sample_min            =  10                       # min UIDs per eval round
updated_models_limit  = 100                       # max concurrent evals

# Filesystem
ROOT_DIR              = os.path.dirname(os.path.abspath(__file__))


# Add these to your constants.py

import datetime as dt

# EMA smoothing factor
alpha: float = 0.3

# Spec version of the validator (bump on incompatible changes)
__spec_version__: str = "0.1.0"

# ─── Cadences & Timeouts ───────────────────────────────────────────────────────
# How often to scan top-miner weights (wall-clock)
scan_top_model_cadence: dt.timedelta = dt.timedelta(minutes=60)

# Minimum time between sequential chain-update checks
chain_update_cadence: dt.timedelta = dt.timedelta(minutes=5)

# How often to push weights on-chain
set_weights_cadence: dt.timedelta = dt.timedelta(hours=1.5)

# ─── Retry thresholds ───────────────────────────────────────────────────────────
# Max blocks before retrying a previously-seen model
model_retry_cadence: int = 1_000

# ─── Weight-sync thresholds ────────────────────────────────────────────────────
# Stake threshold for considering a miner “top”
WEIGHT_SYNC_VALI_MIN_STAKE: float = 1_000.0

# Percent-of-total-weight threshold
WEIGHT_SYNC_MINER_MIN_PERCENT: float = 0.01

# ─── On-chain versioning ────────────────────────────────────────────────────────
# Version key passed when setting weights
weights_version_key: str = "v1"

__validator_version__ = "0.0.1"
version_split = __validator_version__.split(".")
__spec_version__ = (
    (1000 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)
