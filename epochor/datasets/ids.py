
from enum import Enum, auto

class EvalMethodId(Enum):
    # existing entries…
    SYNTHETIC_BENCHMARK = auto()
    
class DatasetId(Enum):
    """
    Enumeration of available dataset identifiers.

    Each member represents a distinct dataset that can be loaded or used
    within the epochor framework.
    """
    # Add any existing entries here if this enum is being expanded.
    # For now, only SYNTHETIC is added as per the prompt.
    SYNTHETIC = auto()
    # Example of how other datasets might be added:
    # SOME_OTHER_DATASET = auto()
    # YET_ANOTHER_ONE = auto()

__all__ = ["DatasetId"]
