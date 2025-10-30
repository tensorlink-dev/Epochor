import os
import torch
from typing import List, Tuple
from keylimiter import KeyLimiter
from datetime import timedelta
import datetime as dt

# Local development mode.
IS_LOCAL_DEVELOPMENT_MODE = bool(os.environ.get("IS_LOCAL_DEVELOPMENT_MODE", False))
LOCAL_MODE_NEURONS_COUNT = int(os.environ.get("LOCAL_MODE_NEURONS_COUNT", 10))
SYNC_BLOCK_CADENCE = 150

# The minimum stake required to be sync'd with the network.
MIN_SYNC_STAKE = 10_000

# The number of models to keep in the cache.
MAX_MODELS_TO_CACHE = 10

# URL of the weights repo.
WEIGHTS_REPO_URL = "https://huggingface.co/tensor-link/model_weights"

# Filename for the weights.
WEIGHTS_FILENAME = "distilbert-base-uncased.pt"

temperature = 0.1
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

# Base project settings
WANDB_PROJECT         = "epocher-validator"       # your wandb project name
SUBNET_UID            = 2                    # your Bittensor subnet UID

# Validator-specific defaults
sample_min            =  10                       # min UIDs per eval round
updated_models_limit  = 100                       # max concurrent evals

# Filesystem
ROOT_DIR              = os.path.dirname(os.path.abspath(__file__))

# EMA smoothing factor
alpha: float = 0.3

# Spec version of the validator (bump on incompatible changes)

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
weights_version_key: int =1 

__validator_version__ = "0.0.1"
version_split = __validator_version__.split(".")
__spec_version__ = (
    (1000 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)
