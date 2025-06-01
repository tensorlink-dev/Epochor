# epochor/validation/base_validator.py

import abc
from typing import Dict, List, Any

class BaseValidator(abc.ABC):
    """
    Completely abstract base class for any competition’s validator. 

    Subclasses MUST implement:
        def validate(
            self,
            uid_to_losses: Dict[int, List[float]],
            uid_to_metadata: Dict[int, Any] = None
        ) -> Dict[int, float]

    where:
      - uid_to_losses: each miner → “list of past loss(es)” or any numeric history.
      - uid_to_metadata: optional extra data (e.g. chain blocks, registration timestamps, model IDs, etc.).
      - return value: a dict mapping each miner‐UID to a float score (higher = better).

    The manager simply dispatches to the appropriate subclass’s validate() method.
    """
    @abc.abstractmethod
    def validate(
        self,
        uid_to_losses: Dict[int, List[float]],
        uid_to_metadata: Dict[int, Any] = None
    ) -> Dict[int, float]:
        """
        Compute a final score for each UID, given whatever “loss histories” or context is needed.
        If a UID is omitted or any error occurs for that UID, that UID’s score should be 0.0.

        Returns:
            { uid: float_score, … }
        """
        pass
