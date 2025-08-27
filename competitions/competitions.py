# The MIT License (MIT)
# Copyright © 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

from typing import List, Tuple, Dict
from epochor.model.model_constraints import Competition, EvalTask, EvalMethodId, NormalizationId, MODEL_CONSTRAINTS_BY_COMPETITION_ID
from competitions import CompetitionId # Corrected import
from epochor.datasets.ids import DatasetId # New import

# BATCH_SIZE and PAGES_PER_EVAL_FINEWEB are not defined in constants, so I will define them here.
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
                    dataset_id=DatasetId.UNIVARIATE_SYNTHETIC, # Changed from 0
                    normalization_id=NormalizationId.NONE,
                    quantiles=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
                    dataset_kwargs={
                        "batch_size": 32,
                        "n_series": 1000,
                        "length" : 1024,
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
