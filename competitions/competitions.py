
from typing import List, Tuple
from epochor.constants import Competition, CompetitionId, MODEL_CONSTRAINTS_BY_COMPETITION_ID, EvalTask, EvalMethodId, DatasetId, NormalizationId, BATCH_SIZE, PAGES_PER_EVAL_FINEWEB

COMPETITION_SCHEDULE_BY_BLOCK: List[Tuple[int, List[Competition]]] = [
    (
        0,
        [
            Competition(
                id = CompetitionId.UNIVARIATE,
                constraints = MODEL_CONSTRAINTS_BY_COMPETITION_ID[CompetitionId.UNIVARIATE],
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
]

def get_current_competition(block: int) -> dict:
    return {}
