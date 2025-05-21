import collections

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
        self.ema_scores = collections.defaultdict(float) # Stores UID -> EMA score
        self.initialized = collections.defaultdict(bool) # Stores UID -> whether EMA has been initialized

    def update(self, uid: int, score: float):
        """
        Updates the EMA score for a given UID.

        Args:
            uid: The unique identifier.
            score: The new score to incorporate into the EMA.
        """
        if not self.initialized[uid]:
            self.ema_scores[uid] = score
            self.initialized[uid] = True
        else:
            self.ema_scores[uid] = (self.alpha * score) + ((1 - self.alpha) * self.ema_scores[uid])

    def get(self, uid: int) -> float:
        """
        Retrieves the current EMA score for a given UID.

        Args:
            uid: The unique identifier.

        Returns:
            The EMA score for the UID. Returns 0.0 if the UID has not been updated yet.
        """
        return self.ema_scores[uid]

    def get_all_scores(self) -> dict:
        """
        Retrieves all current EMA scores.

        Returns:
            A dictionary mapping UIDs to their EMA scores.
        """
        return dict(self.ema_scores)
