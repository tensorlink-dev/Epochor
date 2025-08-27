import dataclasses
from typing import Any, Sequence

from epochor.model.model_constraints import NormalizationId
from epochor.evaluation.method import EvalMethodId


@dataclasses.dataclass
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
    normalization_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    # Ensure correct type + default 0.5
    quantiles: Sequence[float] = dataclasses.field(default_factory=lambda: [0.5])

    dataset_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
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
