"""
Defines project-wide constants for epochor, including competition IDs,
schedules, and model constraints.
"""
from enum import Enum, auto # Standard library imports
from datetime import timedelta # Standard library imports
from typing import Dict, Any # Standard library imports

class CompetitionId(Enum):
    """
    Enumeration of unique identifiers for different competitions.
    """
    # If there were existing IDs, they would go here.
    # Example:
    # SOME_EXISTING_COMPETITION = auto()

    EPOCHOR_SYNTHETIC = auto()

COMPETITION_SCHEDULE_BY_ID: Dict[CompetitionId, timedelta] = {
    # If there were existing schedules, they would go here.
    # Example:
    # CompetitionId.SOME_EXISTING_COMPETITION: timedelta(days=1),

    CompetitionId.EPOCHOR_SYNTHETIC: timedelta(hours=2),
}

MODEL_CONSTRAINTS_BY_COMPETITION_ID: Dict[CompetitionId, Dict[str, Any]] = {
    # If there were existing constraints, they would go here.
    # Example:
    # CompetitionId.SOME_EXISTING_COMPETITION: {"max_parameters": 100_000_000, "allowed_libraries": ["numpy", "pandas"]},
    
    CompetitionId.EPOCHOR_SYNTHETIC: {"max_length": 1024}, # Example constraint
}

# It's good practice to define __all__ for constants modules as well,
# especially if you have many constants and enums.
__all__ = [
    "CompetitionId",
    "COMPETITION_SCHEDULE_BY_ID",
    "MODEL_CONSTRAINTS_BY_COMPETITION_ID",
]
