# epochor/validation/validation_data.py

from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class ValidationData:
    """
    A very flexible container for any per-UID intermediate metrics
    that a validator computes.

    - metrics: maps uid → a dict of arbitrary intermediate values
       (e.g. "losses", "win_rate", "ci_lo", "ci_hi", "aggregate_gap",
       "sep_score", "raw_composite", or anything else).
    """
    metrics: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    def add_metric(self, uid: int, key: str, value: Any):
        """
        Add one intermediate metric under `metrics[uid][key] = value`.
        """
        if uid not in self.metrics:
            self.metrics[uid] = {}
        self.metrics[uid][key] = value


# epochor/validation/validation_results.py

from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class FinalResults:
    """
    A strict container for “final” validation output:

    - uids         : List of all UIDs validated this round.
    - final_scores : Dict[uid → float], cleaned composite score (NaN→0).
    - metadata     : Dict[uid → Any], any per-UID metadata carried forward
                     (e.g. ModelMetadata, block numbers, etc.).
    """
    uids: List[int] = field(default_factory=list)
    final_scores: Dict[int, float] = field(default_factory=dict)
    metadata: Dict[int, Any] = field(default_factory=dict)

@dataclass
class ValidationResults:
    """
    Single object returned by every validator:
      - data    : a ValidationData instance (flexible intermediate metrics)
      - results : a FinalResults instance (strict final scores + metadata)
    """
    data: ValidationData = field(default_factory=ValidationData)
    results: FinalResults = field(default_factory=FinalResults)
