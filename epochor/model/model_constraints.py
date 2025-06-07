from pydantic import BaseModel, Field
from typing import Type, Dict
from epochor.constants import CompetitionId

# Placeholder for the temporal package
class TemporalModel:
    pass

class TemporalConfig:
    pass

class ModelConstraints(BaseModel):
    """
    Defines the constraints for a model.
    """
    max_model_size_bytes: int = Field(
        default=10 * 1024 * 1024,
        description="Maximum size of the model in bytes. Default is 10MB."
    )
    max_model_parameters: int = Field(
        default=1000000,
        description="Maximum number of parameters in the model. Default is 1 million."
    )
    # Placeholder for the model and config classes from the temporal package
    model_cls: Type = Field(default=TemporalModel)
    config_cls: Type = Field(default=TemporalConfig)

# Mapping from CompetitionId to the constraints for each competition.
COMPETITION_CONSTRAINTS: Dict[CompetitionId, ModelConstraints] = {
    CompetitionId.BASELINE: ModelConstraints(),
    # Add other competitions here, for example:
    # CompetitionId.CUSTOM_TRACK_1: ModelConstraints(max_model_size_bytes=20 * 1024 * 1024),
}
