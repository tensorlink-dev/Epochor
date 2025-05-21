# epochor/competition.py

"""
Competition scheduling and blending logic for Epochor subnet.

Allows evaluation across multiple competitions and smoothly blends weights (epsilons)
as defined in COMPETITION_SCHEDULE.
"""

from typing import Dict
from epochor.constants import COMPETITION_SCHEDULE


class EpsilonFunc:
    """
    Blends between 1.0 and 0.0 as a function of block height (linear decay).
    If decay_end == decay_start, epsilon is fixed to 1.0 before that point, then 0.0 after.
    """
    def __init__(self, decay_start: int, decay_end: int):
        self.start = decay_start
        self.end = decay_end

    def __call__(self, block: int) -> float:
        if block < self.start:
            return 1.0
        elif block >= self.end:
            return 0.0
        else:
            progress = (block - self.start) / (self.end - self.start)
            return 1.0 - progress


class CompetitionTracker:
    """
    Returns the active competition weights (epsilons) for a given block.
    COMPETITION_SCHEDULE maps block -> {competition_id: epsilon_func or float}.
    """
    def __init__(self, schedule: Dict[int, Dict[int, float]]):
        # Convert any floats to constant EpsilonFuncs
        self.schedule = {
            block: {
                cid: (ef if callable(ef) else ConstantEpsilonFunc(ef))
                for cid, ef in competitions.items()
            }
            for block, competitions in sorted(schedule.items())
        }

    def get(self, block: int) -> Dict[int, float]:
        latest_block = max([b for b in self.schedule if b <= block], default=None)
        if latest_block is None:
            return {}
        active = self.schedule[latest_block]
        return {cid: float(epsilon(block)) for cid, epsilon in active.items()}


class ConstantEpsilonFunc:
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def __call__(self, block: int) -> float:
        return self.epsilon


# Global instance
tracker = CompetitionTracker(COMPETITION_SCHEDULE)


def get_current_competition(block: int) -> Dict[int, float]:
    """
    Returns the current competition ID weights active at the given block.

    Returns:
        Dict[competition_id, weight] — normalized weights
    """
    weights = tracker.get(block)
    total = sum(weights.values())
    if total == 0:
        return {cid: 1.0 for cid in weights}  # fallback to equal weights
    return {cid: w / total for cid, w in weights.items()}
