
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
    UNIVARIATE_SYNTHETIC = 0 



__all__ = ["DatasetId"]
