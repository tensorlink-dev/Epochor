"""
Defines unique identifiers for different evaluation methodologies or benchmarks.
"""
from enum import Enum, auto # Standard library imports

class EvalMethodId(Enum):
    """
    Enumeration of available evaluation method or benchmark identifiers.

    Each member represents a distinct way an evaluation can be performed
    or a specific benchmark that can be run.
    """
    # If there were existing entries, they would go here.
    # Example:
    # SOME_EXISTING_METHOD = auto()
    
    SYNTHETIC_BENCHMARK = auto()
    # Example of how other methods might be added:
    # REAL_WORLD_SCENARIO_A = auto()

__all__ = ["EvalMethodId"]
