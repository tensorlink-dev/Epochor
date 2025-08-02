"""
Defines unique identifiers for different evaluation methodologies or benchmarks.
"""
from enum import Enum # Standard library imports

class EvalMethodId(Enum):
    """
    Enumeration of available evaluation method or benchmark identifiers.

    Each member represents a distinct way an evaluation can be performed
    or a specific benchmark that can be run.
    """
    CRPS_LOSS = 0 


class NormalizationId(Enum):
    """
    Defines unique identifiers for different normalization methods.
    """
    NONE = 0
    INVERSE_EXPONENTIAL = 1

__all__ = ["EvalMethodId", "NormalizationId"]
