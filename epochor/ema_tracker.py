import collections
from typing import Dict, Any, Optional


class EMATracker:
    """
    Tracks an Exponential Moving Average (EMA) for scores associated with UIDs.
    """

    def __init__(self, alpha: float = 0.2):
        """
        Initializes the EMATracker.

        Args:
            alpha: The smoothing factor for the EMA, between 0 and 1.
                   A higher alpha gives more weight to recent scores.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha must be between 0 and 1.")
        self.alpha = alpha
        self.ema_scores = collections.defaultdict(float)     # UID -> EMA score
        self.initialized = collections.defaultdict(bool)     # UID -> whether EMA has been initialized

    def update(self, uid: int, score: float):
        """
        Updates the EMA score for a given UID.

        Args:
            uid: The unique identifier.
            score: The new score to incorporate into the EMA.
        """
        if not self.initialized[uid]:
            # First time seeing this UID: initialize with raw score
            self.ema_scores[uid] = score
            self.initialized[uid] = True
        else:
            # Standard EMA update
            self.ema_scores[uid] = (self.alpha * score) + ((1.0 - self.alpha) * self.ema_scores[uid])

    def get(self, uid: int) -> float:
        """
        Retrieves the current EMA score for a given UID.

        Args:
            uid: The unique identifier.

        Returns:
            The EMA score for the UID. Returns 0.0 if the UID has not been updated yet.
        """
        return self.ema_scores[uid]

    def get_all_scores(self) -> Dict[int, float]:
        """
        Retrieves all current EMA scores for this tracker.

        Returns:
            A dictionary mapping UIDs to their EMA scores.
        """
        return dict(self.ema_scores)

    def set_alpha(self, new_alpha: float):
        """
        Allows modifying the alpha (smoothing factor) after initialization.

        Args:
            new_alpha: The new alpha value between 0 and 1.
        """
        if not 0.0 <= new_alpha <= 1.0:
            raise ValueError("Alpha must be between 0 and 1.")
        self.alpha = new_alpha


class CompetitionEMATracker:
    """
    Manages a separate EMATracker instance for each competition.

    Usage:
        comp_tracker = CompetitionEMATracker(default_alpha=0.2)
        comp_tracker.update(comp_id=5, uid=42, score=0.75)
        all_scores_for_comp5 = comp_tracker.get(comp_id=5)
        specific_score = comp_tracker.get(comp_id=5, uid=42)
        comp_tracker.set_alpha(comp_id=5, new_alpha=0.1)
    """

    def __init__(self, default_alpha: float = 0.2):
        """
        Initializes the CompetitionEMATracker.

        Args:
            default_alpha: The alpha to use when creating a new EMATracker for a competition.
        """
        self.default_alpha = default_alpha
        self.trackers: Dict[Any, EMATracker] = {}  # competition_id -> EMATracker

    def _get_or_create_tracker(self, competition_id: Any) -> EMATracker:
        """
        Retrieves the EMATracker for the given competition_id, creating one if it does not exist.

        Args:
            competition_id: Identifier for the competition.

        Returns:
            The EMATracker instance associated with competition_id.
        """
        if competition_id not in self.trackers:
            self.trackers[competition_id] = EMATracker(alpha=self.default_alpha)
        return self.trackers[competition_id]

    def update(self, competition_id: Any, uid: int, score: float):
        """
        Updates the EMA score for a given UID under the specified competition.

        Args:
            competition_id: Identifier for the competition.
            uid: The unique identifier.
            score: The new score to incorporate into the EMA.
        """
        tracker = self._get_or_create_tracker(competition_id)
        tracker.update(uid, score)

    def get(self, competition_id: Any, uid: Optional[int] = None) -> Any:
        """
        Retrieves EMA information under the specified competition.

        If `uid` is provided, returns that UID’s EMA score (0.0 if not present).
        If `uid` is None, returns the entire dictionary of {uid: ema_score} for that competition.

        Args:
            competition_id: Identifier for the competition.
            uid: (Optional) The unique identifier. If omitted, return all scores.

        Returns:
            If uid is not None:
                float → the EMA score for that UID (or 0.0 if missing).
            Else:
                Dict[int, float] → mapping UIDs to their EMA scores.
        """
        tracker = self._get_or_create_tracker(competition_id)
        if uid is not None:
            return tracker.get(uid)
        else:
            return tracker.get_all_scores()

    def set_alpha(self, competition_id: Any, new_alpha: float):
        """
        Modifies the alpha value for a specific competition’s EMA tracker.

        Args:
            competition_id: Identifier for the competition.
            new_alpha: The new alpha value between 0 and 1.
        """
        tracker = self._get_or_create_tracker(competition_id)
        tracker.set_alpha(new_alpha)

    def remove_competition(self, competition_id: Any):
        """
        Deletes the EMATracker for a specific competition, if it exists.

        Args:
            competition_id: Identifier for the competition to remove.
        """
        if competition_id in self.trackers:
            del self.trackers[competition_id]

    def clear_all(self):
        """
        Clears all competition‐specific trackers.
        """
        self.trackers.clear()
