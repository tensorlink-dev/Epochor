import collections
from typing import Dict, Any, Optional

from epochor.competition import get_current_competition

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

    def reset_score(self, uid: int):
        """Resets the score for a specific UID to 0."""
        self.ema_scores[uid] = 0.0


class CompetitionEMATracker:
    """
    Manages a separate EMATracker instance for each competition.
    It also handles combining scores based on the current competition schedule
    and manages UID to hotkey mappings.
    """

    def __init__(self, default_alpha: float = 0.2):
        self.default_alpha = default_alpha
        self.trackers: Dict[Any, EMATracker] = {}  # competition_id -> EMATracker
        self.uid_to_hotkey: Dict[int, str] = {}
        self.hotkey_to_uid: Dict[str, int] = {}

    def _get_or_create_tracker(self, competition_id: Any) -> EMATracker:
        if competition_id not in self.trackers:
            self.trackers[competition_id] = EMATracker(alpha=self.default_alpha)
        return self.trackers[competition_id]

    def update(self, competition_id: Any, uid: int, score: float, hotkey: Optional[str] = None):
        """
        Updates the EMA score for a given UID and its hotkey under the specified competition.
        """
        tracker = self._get_or_create_tracker(competition_id)
        tracker.update(uid, score)
        # Update mapping if hotkey is provided
        if hotkey:
            if uid not in self.uid_to_hotkey or self.uid_to_hotkey[uid] != hotkey:
                # Remove old hotkey mapping if it exists and is different
                if self.uid_to_hotkey.get(uid) in self.hotkey_to_uid:
                    del self.hotkey_to_uid[self.uid_to_hotkey[uid]]
                self.uid_to_hotkey[uid] = hotkey
                self.hotkey_to_uid[hotkey] = uid


    def get(self, competition_id: Any, uid: Optional[int] = None) -> Any:
        tracker = self._get_or_create_tracker(competition_id)
        if uid is not None:
            return tracker.get(uid)
        else:
            return tracker.get_all_scores()

    def set_alpha(self, competition_id: Any, new_alpha: float):
        tracker = self._get_or_create_tracker(competition_id)
        tracker.set_alpha(new_alpha)

    def remove_competition(self, competition_id: Any):
        if competition_id in self.trackers:
            del self.trackers[competition_id]

    def clear_all(self):
        self.trackers.clear()
        self.uid_to_hotkey.clear()
        self.hotkey_to_uid.clear()

    def reset_score_for_hotkey(self, hotkey: str):
        """
        Resets the score for a given hotkey to 0 across all competitions.
        This should be called when a model is updated or re-registered.
        """
        if hotkey in self.hotkey_to_uid:
            uid = self.hotkey_to_uid[hotkey]
            for competition_id in self.trackers:
                self.trackers[competition_id].reset_score(uid)

    def get_combined_scores(self, block: int, uids: list[int]) -> Dict[int, float]:
        """
        Calculates the combined scores for a set of UIDs based on the current
        competition weights at a given block.
        """
        # Get the current competition weights {comp_id: weight}
        competition_weights = get_current_competition(block)

        combined_scores = collections.defaultdict(float)

        if not competition_weights:
            # If no competitions are active, return 0 for all uids.
            return {uid: 0.0 for uid in uids}

        # Get all unique UIDs that have scores in any of the active competitions
        all_scored_uids = set()
        active_competition_ids = competition_weights.keys()
        for comp_id in active_competition_ids:
            if comp_id in self.trackers:
                all_scored_uids.update(self.trackers[comp_id].get_all_scores().keys())

        # Ensure we are considering all provided UIDs, even if they have no scores yet.
        all_uids_to_process = set(uids).union(all_scored_uids)

        for uid in all_uids_to_process:
            weighted_score = 0.0
            for comp_id, weight in competition_weights.items():
                if weight > 0 and comp_id in self.trackers:
                    # Get score for this uid in this competition, default to 0.0
                    score = self.trackers[comp_id].get(uid)
                    weighted_score += weight * score

            combined_scores[uid] = weighted_score

        return dict(combined_scores)
