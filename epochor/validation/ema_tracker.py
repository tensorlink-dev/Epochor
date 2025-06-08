import collections
import pickle
import typing
from typing import Dict, Any, Optional

from epochor.competition import get_current_competition


class EMATracker:
    """
    Tracks an Exponential Moving Average (EMA) for scores associated with UIDs.
    """

    def __init__(self, alpha: float = 0.2):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha must be between 0 and 1.")
        self.alpha = alpha
        self.ema_scores: Dict[int, float] = collections.defaultdict(float)
        self.initialized: Dict[int, bool] = collections.defaultdict(bool)

    def update(self, uid: int, score: float):
        if not self.initialized[uid]:
            # first observation
            self.ema_scores[uid] = score
            self.initialized[uid] = True
        else:
            self.ema_scores[uid] = (
                self.alpha * score + (1.0 - self.alpha) * self.ema_scores[uid]
            )

    def get(self, uid: int) -> float:
        return self.ema_scores.get(uid, 0.0)

    def get_all_scores(self) -> Dict[int, float]:
        return dict(self.ema_scores)

    def set_alpha(self, new_alpha: float):
        if not 0.0 <= new_alpha <= 1.0:
            raise ValueError("Alpha must be between 0 and 1.")
        self.alpha = new_alpha

    def reset_score(self, uid: int):
        self.ema_scores[uid] = 0.0


class CompetitionEMATracker:
    """
    Manages one EMATracker per competition, plus raw-score history and hotkey mappings.
    """

    def __init__(self, default_alpha: float = 0.2):
        self.default_alpha = default_alpha
        # comp_id -> EMATracker
        self.trackers: Dict[Any, EMATracker] = {}
        # comp_id -> { uid -> last raw score }
        self.raw_scores: Dict[Any, Dict[int, float]] = collections.defaultdict(dict)
        # UID <-> hotkey
        self.uid_to_hotkey: Dict[int, str] = {}
        self.hotkey_to_uid: Dict[str, int] = {}

    def _get_or_create_tracker(self, competition_id: Any) -> EMATracker:
        if competition_id not in self.trackers:
            self.trackers[competition_id] = EMATracker(alpha=self.default_alpha)
        return self.trackers[competition_id]

    def update(
        self,
        competition_id: Any,
        uid: int,
        score: float,
        hotkey: Optional[str] = None
    ):
        """
        1) Update the EMA for this (comp_id, uid).
        2) Store the raw score for diagnostics.
        3) Update UID↔hotkey mapping if provided.
        """
        # EMA update
        tracker = self._get_or_create_tracker(competition_id)
        tracker.update(uid, score)

        # raw score
        self.raw_scores[competition_id][uid] = score

        # mapping
        if hotkey:
            old = self.uid_to_hotkey.get(uid)
            if old and old != hotkey:
                del self.hotkey_to_uid[old]
            self.uid_to_hotkey[uid] = hotkey
            self.hotkey_to_uid[hotkey] = uid

    def get(
        self,
        competition_id: Any,
        uid: Optional[int] = None
    ) -> Any:
        """
        If uid is None, returns EMA dict; otherwise EMA for that uid.
        """
        tracker = self._get_or_create_tracker(competition_id)
        return tracker.get(uid) if uid is not None else tracker.get_all_scores()

    def get_raw_scores(
        self,
        competition_id: Any,
        uid: Optional[int] = None
    ) -> Any:
        """
        If uid is None, returns raw-score dict; otherwise raw score for that uid.
        """
        comp = self.raw_scores.get(competition_id, {})
        return comp.get(uid, 0.0) if uid is not None else dict(comp)

    def set_alpha(self, competition_id: Any, new_alpha: float):
        tracker = self._get_or_create_tracker(competition_id)
        tracker.set_alpha(new_alpha)

    def remove_competition(self, competition_id: Any):
        """Wipe all data for that competition."""
        self.trackers.pop(competition_id, None)
        self.raw_scores.pop(competition_id, None)

    def clear_all(self):
        """Wipe every competition and mapping."""
        self.trackers.clear()
        self.raw_scores.clear()
        self.uid_to_hotkey.clear()
        self.hotkey_to_uid.clear()

    def reset_competitions(self, competition_ids: typing.Set[Any]):
        """Resets tracked competitions to only those identified.

        Args:
            competition_ids (typing.Set[Any]): Competition ids to continue tracking.
        """
        # Make a list to avoid issues deleting from within the dictionary iterator.
        for key in list(self.trackers.keys()):
            if key not in competition_ids:
                del self.trackers[key]
        
        for key in list(self.raw_scores.keys()):
            if key not in competition_ids:
                del self.raw_scores[key]

    def reset_score_for_hotkey(self, hotkey: str):
        """
        Zero out both EMA and raw for the model behind this hotkey.
        """
        uid = self.hotkey_to_uid.get(hotkey)
        if uid is None:
            return

        # EMA reset
        for tracker in self.trackers.values():
            tracker.reset_score(uid)

        # raw reset
        for comp_dict in self.raw_scores.values():
            comp_dict.pop(uid, None)

    def get_combined_scores(self, block: int, uids: list[int]) -> Dict[int, float]:
        """
        Block‐aware aggregation of raw scores:
          score_i = sum_over_active_comps(weight(comp) * raw_score_i)
        """
        competition_weights = get_current_competition(block)
        combined = collections.defaultdict(float)

        for uid in uids:
            total = 0.0
            for comp_id, w in competition_weights.items():
                if comp_id in self.raw_scores:
                    total += w * self.raw_scores[comp_id].get(uid, 0.0)
            combined[uid] = total

        return dict(combined)

    def get_uid(self, hotkey: str) -> Optional[int]:
        return self.hotkey_to_uid.get(hotkey)

    def get_hotkey(self, uid: int) -> Optional[str]:
        return self.uid_to_hotkey.get(uid)
        
    def get_subnet_weights(
            self,
            competitions: List[Competition],
            min_comp_weight_threshold: float = 0.0,
        ) -> torch.Tensor:
            """Aggregate tensor‐based weights across competitions (with optional thresholding)."""
            subnet_weights = torch.zeros(self.num_neurons, dtype=torch.float32)

            for comp in competitions:
                comp_id = comp.id
                if comp_id not in self.trackers:
                    continue

                # fetch the EMA‐derived tensor and normalize
                comp_weights = self.trackers[comp_id].get_all_scores()
                tensor = torch.tensor(
                    [comp_weights.get(uid, 0.0) for uid in range(self.num_neurons)],
                    dtype=torch.float32
                )
                tensor /= tensor.sum() if tensor.sum() > 0 else 1.0
                tensor = tensor.nan_to_num(0.0)

                if min_comp_weight_threshold > 0:
                    mask = tensor < min_comp_weight_threshold
                    tensor[mask] = 0.0
                    tensor /= tensor.sum() if tensor.sum() > 0 else 1.0
                    tensor = tensor.nan_to_num(0.0)

                subnet_weights += tensor * comp.reward_percentage

            subnet_weights /= subnet_weights.sum() if subnet_weights.sum() > 0 else 1.0
            return subnet_weights.nan_to_num(0.0)

    def get_competition_weights(self, competition_id: int) -> torch.Tensor:
        """Return the EMA‐derived weight tensor for one competition."""
        tracker = self._get_or_create_tracker(competition_id)
        raw = tracker.get_all_scores()
        return torch.tensor(
            [raw.get(uid, 0.0) for uid in range(self.num_neurons)],
            dtype=torch.float32
        )

    def save_state(self, filepath: str):
        """
        Serialize entire tracker state to disk.
        """
        state = {
            "default_alpha":      self.default_alpha,
            "trackers":           self.trackers,
            "raw_scores":         dict(self.raw_scores),
            "uid_to_hotkey":      self.uid_to_hotkey,
            "hotkey_to_uid":      self.hotkey_to_uid,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)

    def load_state(self, filepath: str):
        """
        Load tracker state from disk, replacing all in‐memory data.
        """
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        self.default_alpha     = state["default_alpha"]
        self.trackers          = state["trackers"]
        self.raw_scores        = collections.defaultdict(dict, state["raw_scores"])
        self.uid_to_hotkey     = state["uid_to_hotkey"]
        self.hotkey_to_uid     = state["hotkey_to_uid"]