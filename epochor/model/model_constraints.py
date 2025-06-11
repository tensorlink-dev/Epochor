from pydantic import BaseModel, Field
from typing import Type, Dict
from epochor.constants import CompetitionId
from competition.competitions import ModelConstraints
from temporal import TransformerTemporalModel as model_cls 
from temporal.configs import  TransformerTimeSeriesConfig as config_cls


class ModelConstraintsUnivariate(BaseModel):
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

# Mapping from CompetitionId to the constraints for each competition.
COMPETITION_CONSTRAINTS: Dict[CompetitionId, ModelConstraints] = {
    CompetitionId.UNIVARIATE: ModelConstraintsUnivariate(),
    # CompetitionId.CUSTOM_TRACK_1: ModelConstraints(max_model_size_bytes=20 * 1024 * 1024),
}
