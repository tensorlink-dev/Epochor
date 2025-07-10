# The MIT License (MIT)
# Copyright © 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

from typing import List, Tuple
from constants import CompetitionId, BATCH_SIZE
from epochor.model.model_constraints import Competition, EvalTask, EvalMethodId, DatasetId, NormalizationId, MODEL_CONSTRAINTS_BY_COMPETITION_ID

# BATCH_SIZE and PAGES_PER_EVAL_FINEWEB are not defined in constants, so I will define them here.
PAGES_PER_EVAL_UNIV = 1
COMPETITION_SCHEDULE_BY_BLOCK: Dict[int, List[Competition]] = {
    0: [
        Competition(
            id=CompetitionId.UNIVARIATE,
            constraints=MODEL_CONSTRAINTS_BY_COMPETITION_ID[CompetitionId.UNIVARIATE],
            reward_percentage=0.3,
            eval_tasks=[
                EvalTask(
                    name="SYNTHETIC-V1",
                    method_id=EvalMethodId.CRPS_LOSS,
                    dataset_id=DatasetId.UNIVARIATE_SYNTHETIC,
                    normalization_id=NormalizationId.NONE,
                    dataset_kwargs={
                        "batch_size": BATCH_SIZE,
                        "num_pages": PAGES_PER_EVAL_UNIV,
                    },
                    weight=1.00,
                )
            ],
        )
    ],
    # … add more block keys here …
}


def get_current_competition(block: int) -> dict:
    return {}
