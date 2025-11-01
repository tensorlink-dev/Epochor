from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Sequence, Type

import torch
from pydantic import BaseModel, Field
from torch import nn

from competitions import CompetitionId  # Updated import to be from competitions package directly
from competitions.epsilon import EpsilonFunc, FixedEpsilon  # Import EpsilonFunc and FixedEpsilon
from epochor.model.base import (
    AutoregressiveGenerationMixin,
    BaseTemporalModel,
    TemporalModelOutput,
)


@dataclass
class SimpleTimeSeriesConfig:
    context_length: int = 24
    prediction_length: int = 1
    input_dim: int = 1
    output_dim: int = 1
    hidden_size: int = 16
    num_layers: int = 1
    dropout: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimpleTimeSeriesConfig":
        return cls(**data)


class SimpleAutoregressiveModel(BaseTemporalModel, AutoregressiveGenerationMixin):
    """Tiny GRU-based autoregressive model for smoke tests and defaults."""

    def __init__(self, config: SimpleTimeSeriesConfig | None = None):
        config = config or SimpleTimeSeriesConfig()
        super().__init__(config=config)
        input_dim = getattr(config, "input_dim", 1)
        hidden_size = getattr(config, "hidden_size", 16)
        output_dim = getattr(config, "output_dim", 1)

        self.encoder = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.projection = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor, **_: Any) -> TemporalModelOutput:  # type: ignore[override]
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        outputs, hidden = self.encoder(x)
        predictions = self.projection(outputs)
        return self._to_output({"predictions": predictions, "state": hidden})

    def init_generation_state(self, inputs: torch.Tensor, **_: Any) -> Dict[str, Any]:
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(-1)
        _, hidden = self.encoder(inputs)
        return {"hidden": hidden}

    def generation_step(
        self, step_input: torch.Tensor, state: Dict[str, Any], **_: Any
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        if step_input.dim() == 2:
            step_input = step_input.unsqueeze(-1)
        outputs, new_hidden = self.encoder(step_input, state.get("hidden"))
        predictions = self.projection(outputs)
        return predictions, {"hidden": new_hidden}

    def select_next_token(self, step_outputs: torch.Tensor, **_: Any) -> torch.Tensor:
        return step_outputs


def build_default_autoregressive_model(
    config: SimpleTimeSeriesConfig,
) -> BaseTemporalModel:
    return SimpleAutoregressiveModel(config=config)

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
    # Placeholder for the default miner model and config
    model_cls: Callable[[Any], BaseTemporalModel] = Field(
        default=build_default_autoregressive_model
    )
    config_cls: Type = Field(default=SimpleTimeSeriesConfig)
    model_type: Type[BaseTemporalModel] = Field(default=BaseTemporalModel)
    epsilon_func: Type[EpsilonFunc] = Field(default_factory=lambda: FixedEpsilon(epsilon=0.0))

class EvalMethodId(IntEnum):
    CRPS_LOSS = 0

class DatasetId(IntEnum):
    UNIVARIATE_SYNTHETIC = 0

class NormalizationId(IntEnum):
    NONE = 0


@dataclass
class EvalTask:
    """Represents a task to evaluate a model on.

    Args:
        name: Friendly task name.
        method_id: Which evaluation method to use.
        dataset_id: Identifier of the dataset to evaluate on.
        normalization_id: Normalization strategy (default NONE).
        normalization_kwargs: Extra args for normalization (ignored for NONE).
        quantiles: Target quantiles in (0, 1). Accepts a single float or an iterable.
                   Defaults to [0.5].
        dataset_kwargs: Extra args for the dataset loader.
        weight: Positive weight applied to the normalized score.
    """

    # Required
    name: str
    method_id: EvalMethodId
    dataset_id: int

    # Options
    normalization_id: NormalizationId = NormalizationId.NONE
    normalization_kwargs: dict[str, Any] = field(default_factory=dict)

    # Ensure correct type + default 0.5
    quantiles: Sequence[float] = field(default_factory=lambda: [0.5])

    dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0

    def __post_init__(self):
        # --- weight guard ---
        if self.weight <= 0:
            raise ValueError("Weight must be positive.")

        # --- quantiles coercion/validation ---
        q = self.quantiles
        # Allow a single float/int
        if isinstance(q, (float, int)):
            q_list = [float(q)]
        else:
            # Try iterating and casting to floats
            try:
                q_list = [float(x) for x in q]
            except TypeError as e:
                raise TypeError(
                    "quantiles must be a float or an iterable of floats."
                ) from e

        # Default to [0.5] if empty or None-like
        if not q_list:
            q_list = [0.5]

        # Validate open interval (0,1)
        for x in q_list:
            if not (0.0 < x < 1.0):
                raise ValueError(
                    f"Invalid quantile {x}. Each quantile must satisfy 0 < q < 1."
                )

        # Deduplicate and sort for stability
        q_list = sorted(set(q_list))
        self.quantiles = q_list

        # --- normalization rules ---
        match self.normalization_id:
            case NormalizationId.NONE:
                if self.normalization_kwargs:
                    raise ValueError(
                        "Normalization kwargs should not be provided for NONE normalization."
                    )

            case NormalizationId.INVERSE_EXPONENTIAL:
                # Allow missing or None ceiling; if missing, set to None.
                ceiling = self.normalization_kwargs.get("ceiling", None)
                if ceiling is None:
                    # "allow none also" → default ceiling to None explicitly
                    self.normalization_kwargs["ceiling"] = None
                else:
                    if not isinstance(ceiling, (int, float)) or ceiling <= 0:
                        raise ValueError(
                            "Normalization 'ceiling' must be a positive number or None."
                        )

            case _:
                # Other normalization strategies: no-op here (extend as needed)
                pass


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
