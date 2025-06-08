import datetime
from competition.competitions import Competition, CompetitionId, ModelConstraints
from competition.epsilon import EpsilonFunc

# Release
__version__ = "1.0.0"

# Validator schema version
__validator_version__ = "1.0.0"
version_split = __validator_version__.split(".")
__spec_version__ = (
    (1000 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)

# EMA alpha for blended scores
alpha = 0.1

# How often to scan the metagraph for top performing models to give them a chance to be re-evaluated.
scan_top_model_cadence = datetime.timedelta(minutes=30)

# How often to check for new models from the same UID.
chain_update_cadence = datetime.timedelta(minutes=5)

# How many blocks to wait before retrying a model.
model_retry_cadence = 7200 # Roughly 1 day of blocks

# The minimum stake a validator must have to be considered for setting weights on a miner.
WEIGHT_SYNC_VALI_MIN_STAKE = 10000
ROOT_DIR = Path(__file__).parent.parent

# The minimum percentage of weight a validator must have on a miner for it to be considered a top miner.
WEIGHT_SYNC_MINER_MIN_PERCENT = 0.05

# Version key for weights
weights_version_key = 1

# How often to sync the random seed for evaluations.
# Synchronize on blocks roughly every 30 minutes.
SYNC_BLOCK_CADENCE = 150
# Delay at least as long as the sync block cadence with an additional buffer.
EVAL_BLOCK_DELAY = SYNC_BLOCK_CADENCE + 100

# The maximum number of batches to evaluate from a dataset.
MAX_BATCHES_PER_DATASET = 50 

# Temperature for softmax when calculating weights.
temperature = 0.1

# The maximum number of run steps to log to a single wandb run.
MAX_RUN_STEPS_PER_WANDB_RUN = 100

# The current spec version of the validator.



@dataclass
class Competition:
    """Defines a competition."""

    # Unique ID for this competition.
    # Recommend making an IntEnum for use in the subnet codebase.
    id: int

    # All restrictions on models allowed in this competition.
    constraints: ModelConstraints

    # Percentage of emissions dedicated to this competition.
    reward_percentage: float

    # The set of tasks used to evaluate models in this competition.
    eval_tasks: List[EvalTask] = dataclasses.field(default_factory=list)

# Competition schedule.
# Schedule of competitions by block.
COMPETITION_SCHEDULE_BY_BLOCK: List[Tuple[int, List[Competition]]] = [
    (
        0,
        [
            Competition(
                id = CompetitionId.UNIVARIATE,
                constraints = MODEL_CONSTRAINTS_BY_COMPETITION_ID_TMP[CompetitionId.UNIVARIATE],
                reward_percentage= 0.3,
                eval_tasks=[
                    EvalTask(
                        name="SYNTHETIC-V1",
                        method_id=EvalMethodId.CRPS_LOSS,
                        dataset_id=DatasetId.UNIVARIATE_SYNTHETIC,
                        normalization_id=NormalizationId.NONE,
                        dataset_kwargs={
                            "batch_size": BATCH_SIZE,
                            "num_pages": PAGES_PER_EVAL_FINEWEB,
                        },
                        weight=1.00 ,
                    )
            )
        ]
    )