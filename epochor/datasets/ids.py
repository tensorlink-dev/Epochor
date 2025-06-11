
from enum import Enum

class DatasetId(Enum):
    """
    Enumeration of available dataset identifiers.

    Each member represents a distinct dataset that can be loaded or used
    within the epochor framework.
    """
    UNIVARIATE_SYNTHETIC = 0 



__all__ = ["DatasetId"]
