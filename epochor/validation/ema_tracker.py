from __future__ import annotations

import threading
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, List, Any, Union, Set, Callable, cast

import torch
from epochor.model.model_constraints import Competition
from competitions.competitions import get_current_competition


class WeightSource(Protocol):
    """Returns a map of competition_id -> weight for a given block."""
    def __call__(self, block: int) -> Dict[int, float]:
        ...


@dataclass
class RawData:
    score: Union[float, Dict[str, float], Any]
    block: int


@dataclass
class EMATracker:
    """
    Per-UID EMA with:
      - alpha:       used for subsequent updates
      - alpha_init:  used for the first update (prev is None)
      - default_baseline: baseline value when initializing the EMA
    First update: EMA0 = (1 - alpha_init) * baseline + alpha_init * score
    Next updates: EMA  = alpha * score + (1 - alpha) * EMA_prev
    """
    alpha: float = 0.2
    alpha_init: float = 0.2
    default_baseline: float = 0.0
    _scores: Dict[int, float] = field(default_factory=dict)

    def update(self, uid: int, score: float, baseline: Optional[float] = None) -> None:
        prev = self._scores.get(uid)
        if prev is None:
            a = float(self.alpha_init)
            b = float(self.default_baseline if baseline is None else baseline)
            self._scores[uid] = (1.0 - a) * b + a * float(score)
        else:
            self._scores[uid] = float(self.alpha) * float(score) + (1.0 - float(self.alpha)) * float(prev)

    def get(self, uid: int) -> float:
        return self._scores.get(uid, 0.0)

    def get_all(self) -> Dict[int, float]:
        return dict(self._scores)

    def reset(self, uid: int) -> None:
        self._scores.pop(uid, None)

    def set_alpha(self, new_alpha: float) -> None:
        if not (0.0 <= new_alpha <= 1.0):
            raise ValueError("Alpha must be between 0 and 1.")
        self.alpha = float(new_alpha)

    def set_alpha_init(self, new_alpha_init: float) -> None:
        if not (0.0 <= new_alpha_init <= 1.0):
            raise ValueError("alpha_init must be between 0 and 1.")
        self.alpha_init = float(new_alpha_init)

    def set_default_baseline(self, baseline: float) -> None:
        self.default_baseline = float(baseline)


@dataclass
class CompetitionData:
    ema: EMATracker = field(default_factory=EMATracker)
    raw: Dict[int, RawData] = field(default_factory=dict)
    weights: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=torch.float32))
    last_updated_block: Optional[int] = None


class CompetitionEMATracker:
    """
    Thread-safe tracker for:
      - Per-competition EMA (per UID)
      - Raw scores (with block)
      - Per-competition weight tensors
      - Hotkey <-> UID mapping

    First EMA for a UID uses a baseline of 0.0: EMA0 = alpha_init * new_score.
    """

    def __init__(self, num_neurons: int, weight_source: WeightSource = get_current_competition):
        self.num_neurons = int(num_neurons)
        self.weight_source: Callable[[int], Dict[int, float]] = cast(Callable[[int], Dict[int, float]], weight_source)

        self._data: Dict[int, CompetitionData] = {}
        self._uid_to_hotkey: Dict[int, str] = {}
        self._hotkey_to_uid: Dict[str, int] = {}

        self._lock = threading.RLock()

    # -------------------------
    # Internal helpers
    # -------------------------
    def _comp(self, comp_id: int) -> CompetitionData:
        return self._data.setdefault(int(comp_id), CompetitionData())

    def get_competition_data(self, comp_id: int) -> Optional[CompetitionData]:
        with self._lock:
            return self._data.get(int(comp_id))

    # -------------------------
    # Public API
    # -------------------------
    def update(
        self,
        comp_id: int,
        uid: int,
        score: Union[float, Dict[str, float], Any],
        block: int,
        hotkey: Optional[str] = None
    ) -> None:
        with self._lock:
            cd = self._comp(comp_id)

            # For scalar score: update EMA. First tick baseline = 0.0 → EMA_init = alpha_init * score.
            if isinstance(score, (int, float)):
                cd.ema.update(int(uid), float(score), baseline=0.0)

            # Always store raw (scalar or dict) with the block.
            cd.raw[int(uid)] = RawData(score=score, block=int(block))
            cd.last_updated_block = int(block)

            # Maintain hotkey ↔ uid mapping.
            if hotkey:
                old = self._uid_to_hotkey.get(int(uid))
                if old and old != hotkey:
                    self._hotkey_to_uid.pop(old, None)
                self._uid_to_hotkey[int(uid)] = hotkey
                self._hotkey_to_uid[hotkey] = int(uid)

    def get(self, comp_id: int, uid: Optional[int] = None) -> Union[float, Dict[int, float]]:
        """Get EMA score(s). If uid is provided, returns that EMA; else returns dict of all EMAs."""
        with self._lock:
            cd = self._data.get(int(comp_id), CompetitionData())
            return cd.ema.get(int(uid)) if uid is not None else cd.ema.get_all()

    def get_raw_scores(self, comp_id: int, uid: Optional[int] = None) -> Union[RawData, Dict[int, RawData], None]:
        """Get raw score(s). If uid is provided, returns that RawData; else returns dict of RawData."""
        with self._lock:
            cd = self._data.get(int(comp_id), CompetitionData())
            if uid is not None:
                return cd.raw.get(int(uid))
            return dict(cd.raw)

    def combined_raw(self, block: int, uids: List[int]) -> Dict[int, Dict[str, float]]:
        """
        Weighted aggregation of nested/dict raw scores across competitions according to weight_source(block).
        If weight_source returns an invalid structure or throws, falls back to empty map (no contribution).
        """
        try:
            wmap = self.weight_source(int(block))
            if not isinstance(wmap, dict):
                wmap = {}
        except Exception:
            wmap = {}

        out: Dict[int, Dict[str, float]] = {}
        with self._lock:
            for uid in uids:
                uid_i = int(uid)
                agg: Dict[str, float] = {}
                for cid, w in wmap.items():
                    rd = self._data.get(int(cid), CompetitionData()).raw.get(uid_i)
                    if rd is None:
                        continue
                    if isinstance(rd.score, dict):
                        for k, v in rd.score.items():
                            try:
                                agg[k] = agg.get(k, 0.0) + float(v) * float(w)
                            except Exception:
                                # skip non-numeric values
                                pass
                    else:
                        try:
                            agg["score"] = agg.get("score", 0.0) + float(rd.score) * float(w)
                        except Exception:
                            pass
                out[uid_i] = agg
        return out

    def get_combined_scores(self, block: int, uids: List[int]) -> Dict[int, float]:
        """Convenience: combined_raw restricted to scalar 'score' field."""
        nested = self.combined_raw(block, uids)
        return {int(uid): float(data.get("score", 0.0)) for uid, data in nested.items()}

    def set_alpha(self, comp_id: int, new_alpha: float) -> None:
        """Set the EMA alpha for subsequent updates (per-competition)."""
        with self._lock:
            self._comp(int(comp_id)).ema.set_alpha(float(new_alpha))

    def set_alpha_init(self, comp_id: int, new_alpha_init: float) -> None:
        """Set the EMA alpha_init for first updates (per-competition)."""
        with self._lock:
            self._comp(int(comp_id)).ema.set_alpha_init(float(new_alpha_init))

    def set_default_baseline(self, comp_id: int, baseline: float) -> None:
        """Set the baseline used when initializing a UID's EMA (per-competition)."""
        with self._lock:
            self._comp(int(comp_id)).ema.set_default_baseline(float(baseline))

    def remove_competition(self, comp_id: int) -> None:
        with self._lock:
            self._data.pop(int(comp_id), None)

    def clear_all(self) -> None:
        with self._lock:
            self._data.clear()
            self._uid_to_hotkey.clear()
            self._hotkey_to_uid.clear()

    def reset_competitions(self, keep: Set[int]) -> None:
        with self._lock:
            keep_ids = {int(x) for x in keep}
            for cid in list(self._data):
                if int(cid) not in keep_ids:
                    del self._data[cid]

    def reset_uid(self, uid: int) -> None:
        """
        Reset only the EMA for this UID across all competitions.
        Keeps raw scores intact (so history remains available if needed).
        """
        with self._lock:
            for cd in self._data.values():
                cd.ema.reset(int(uid))
                # DO NOT zero raw score here; preserving raw aids warm starts and analytics.

    def reset_score_for_hotkey(self, hotkey: str) -> None:
        uid = self.uid_for(hotkey)
        if uid is not None:
            self.reset_uid(int(uid))

    def subnet_weights(
        self,
        competitions: List[Competition],
        min_comp_weight_threshold: float = 0.0
    ) -> torch.Tensor:
        """
        Aggregate per-competition normalized weights into a single subnet weight vector.
        Each comp contributes its recorded 'weights' tensor scaled by comp.reward_percentage.
        If min_comp_weight_threshold > 0, very small weights are zeroed then renormalized.
        """
        out = torch.zeros(self.num_neurons, dtype=torch.float32)
        with self._lock:
            for comp in competitions:
                cid = int(comp.id)
                cd = self.get_competition_data(cid)
                if not cd or cd.weights.numel() == 0:
                    continue
                t = cd.weights.clone().float()
                if min_comp_weight_threshold > 0:
                    mask = t < float(min_comp_weight_threshold)
                    t[mask] = 0.0
                    s = t.sum().item() or 1.0
                    t /= s
                    t = t.nan_to_num(0.0)
                out += t * float(comp.reward_percentage)
        s = out.sum().item() or 1.0
        return (out / s).nan_to_num(0.0)

    def record_competition_weights(self, comp_id: int, weights: torch.Tensor) -> None:
        with self._lock:
            self._comp(int(comp_id)).weights = weights.clone().float()

    def get_competition_weights(self, comp_id: int) -> torch.Tensor:
        with self._lock:
            cd = self.get_competition_data(int(comp_id))
            return cd.weights.clone() if cd and cd.weights.numel() > 0 else torch.zeros(self.num_neurons, dtype=torch.float32)

    def uid_for(self, hotkey: str) -> Optional[int]:
        with self._lock:
            return self._hotkey_to_uid.get(hotkey)

    def hotkey_for(self, uid: int) -> Optional[str]:
        with self._lock:
            return self._uid_to_hotkey.get(int(uid))

    def get_last_updated_block(self, comp_id: int) -> Optional[int]:
        with self._lock:
            cd = self.get_competition_data(int(comp_id))
            return cd.last_updated_block if cd else None

    # -------------------------
    # Persistence
    # -------------------------
    def save(self, path: str) -> None:
        with self._lock:
            state = {
                "num_neurons": int(self.num_neurons),
                "data": {
                    str(cid): {
                        "ema_scores": cd.ema.get_all(),
                        "ema_alpha": float(cd.ema.alpha),
                        "ema_alpha_init": float(cd.ema.alpha_init),
                        "ema_default_baseline": float(cd.ema.default_baseline),
                        "raw": {str(uid): {"score": rd.score, "block": int(rd.block)} for uid, rd in cd.raw.items()},
                        "weights": cd.weights.tolist(),
                        "last_updated_block": cd.last_updated_block,
                    }
                    for cid, cd in self._data.items()
                },
                "hotkey_map": {str(uid): hk for uid, hk in self._uid_to_hotkey.items()},
            }
        Path(path).write_text(json.dumps(state))

    def load(self, path: str) -> None:
        raw = json.loads(Path(path).read_text())
        with self._lock:
            self.num_neurons = int(raw["num_neurons"])
            self.clear_all()
            for cid_str, entry in raw.get("data", {}).items():
                cid = int(cid_str)
                cd = self._comp(cid)

                cd.ema.set_alpha(float(entry.get("ema_alpha", 0.2)))
                if "ema_alpha_init" in entry:
                    cd.ema.set_alpha_init(float(entry["ema_alpha_init"]))
                if "ema_default_baseline" in entry:
                    cd.ema.set_default_baseline(float(entry["ema_default_baseline"]))

                ema_scores = entry.get("ema_scores", {})
                cd.ema._scores = {int(u): float(v) for u, v in ema_scores.items()}

                raw_map = entry.get("raw", {})
                cd.raw = {int(u): RawData(score=e["score"], block=int(e["block"])) for u, e in raw_map.items()}

                weights_list = entry.get("weights", [])
                cd.weights = torch.tensor(weights_list, dtype=torch.float32)

                cd.last_updated_block = entry.get("last_updated_block")

            hk_map = raw.get("hotkey_map", {})
            self._uid_to_hotkey = {int(u): hk for u, hk in hk_map.items()}
            self._hotkey_to_uid = {hk: int(u) for u, hk in hk_map.items()}
