from pydantic import BaseModel, Field
from typing import Type, Dict, List
from dataclasses import dataclass, field
from temporal.models.builder import build_time_series_transformer as model_cls
from temporal.configs.transformer_config import  TransformerTimeSeriesConfig as config_cls
from temporal.models.base_model import  BaseTemporalModel as model_type

from competitions import CompetitionId # Updated import to be from competitions package directly
from enum import IntEnum # Keep IntEnum for other enums if they are still local
from competitions.epsilon import EpsilonFunc, FixedEpsilon # Import EpsilonFunc and FixedEpsilon

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
    model_type: Type=  Field(default=model_type)
    epsilon_func: Type[EpsilonFunc] = Field(default_factory=lambda: FixedEpsilon(epsilon=0.0))

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
    id: CompetitionId # Changed type hint to use imported CompetitionId

    # All restrictions on models allowed in this competition.
    constraints: ModelConstraints

    # Percentage of emissions dedicated to this competition.
    reward_percentage: float

    # The set of tasks used to evaluate models in this competition.
    eval_tasks: List[EvalTask] = field(default_factory=list)

# Mapping from CompetitionId to the constraints for each competition.
MODEL_CONSTRAINTS_BY_COMPETITION_ID: Dict[CompetitionId, ModelConstraints] = {
    CompetitionId.UNIVARIATE: ModelConstraints(),
    # CompetitionId.CUSTOM_TRACK_1: ModelConstraints(max_model_size_bytes=20 * 1024 * 1024),
}
