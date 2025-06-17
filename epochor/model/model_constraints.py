from pydantic import BaseModel, Field
from typing import Type, Dict, List
from constants import CompetitionId
from dataclasses import dataclass
from temporal.models.base_model import  BaseTemporalModel as model_cls 
from temporal.configs.transformer_config import  TransformerTimeSeriesConfig as config_cls
from enum import IntEnum

class ModelConstraints(BaseModel):
    """
    Defines the constraints for a model.
    """
    max_model_size_bytes: int = Field(
        default=10 * 1024 * 1024 * 1024,
        description="Maximum size of the model in bytes. Default is 10GB."
    )
    max_model_parameters: int = Field(
        default=5_000_000_000,
        description="Maximum number of parameters in the model. Default is 1 million."
    )
    # Placeholder for the model and config classes from the temporal package
    model_cls: Type = Field(default=model_cls)
    config_cls: Type = Field(default=config_cls)

class EvalMethodId(IntEnum):
    CRPS_LOSS = 0

class DatasetId(IntEnum):
    UNIVARIATE_SYNTHETIC = 0

class NormalizationId(IntEnum):
    NONE = 0

@dataclass
class EvalTask:
    name: str
    method_id: EvalMethodId
    dataset_id: DatasetId
    normalization_id: NormalizationId
    dataset_kwargs: Dict
    weight: float

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

# Mapping from CompetitionId to the constraints for each competition.
MODEL_CONSTRAINTS_BY_COMPETITION_ID: Dict[CompetitionId, ModelConstraints] = {
    CompetitionId.UNIVARIATE: ModelConstraints(),
    # CompetitionId.CUSTOM_TRACK_1: ModelConstraints(max_model_size_bytes=20 * 1024 * 1024),
}
