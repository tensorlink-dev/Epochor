from __future__ import annotations
import threading
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, List, Any, Union, Set
import torch
from epochor.model.model_constraints import Competition
from competitions.competitions import get_current_competition

class WeightSource(Protocol):
    def __call__(self, block: int) -> Dict[int, float]:
        ...

@dataclass
class RawData:
    score: Union[float, Dict[str, float], Any]
    block: int

@dataclass
class EMATracker:
    alpha: float = 0.2
    _scores: Dict[int, float] = field(default_factory=dict)

    def update(self, uid: int, score: float) -> None:
        prev = self._scores.get(uid)
        self._scores[uid] = score if prev is None else self.alpha * score + (1 - self.alpha) * prev

    def get(self, uid: int) -> float:
        return self._scores.get(uid, 0.0)

    def get_all(self) -> Dict[int, float]:
        return dict(self._scores)

    def reset(self, uid: int) -> None:
        self._scores.pop(uid, None)

    def set_alpha(self, new_alpha: float) -> None:
        if not (0.0 <= new_alpha <= 1.0):
            raise ValueError("Alpha must be between 0 and 1.")
        self.alpha = new_alpha

@dataclass
class CompetitionData:
    ema: EMATracker = field(default_factory=EMATracker)
    raw: Dict[int, RawData] = field(default_factory=dict)
    weights: torch.Tensor = field(default_factory=lambda: torch.tensor([], dtype=torch.float32))
    last_updated_block: Optional[int] = None

class CompetitionEMATracker:
    """
    Thread-safe replacement for CompetitionEMATracker.
    Tracks per-competition EMA, raw scores (with block), weight tensors, and hotkeys.
    """

    def __init__(self, num_neurons: int, weight_source: WeightSource = get_current_competition):
        self.num_neurons = num_neurons
        self.weight_source = weight_source

        self._data: Dict[int, CompetitionData] = {}
        self._uid_to_hotkey: Dict[int, str] = {}
        self._hotkey_to_uid: Dict[str, int] = {}

        self._lock = threading.RLock()

    def _comp(self, comp_id: int) -> CompetitionData:
        return self._data.setdefault(comp_id, CompetitionData())

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
            if isinstance(score, (int, float)):
                cd.ema.update(uid, float(score))
            cd.raw[uid] = RawData(score=score, block=block)
            cd.last_updated_block = block

            if hotkey:
                old = self._uid_to_hotkey.get(uid)
                if old and old != hotkey:
                    self._hotkey_to_uid.pop(old, None)
                self._uid_to_hotkey[uid] = hotkey
                self._hotkey_to_uid[hotkey] = uid

    def get(self, comp_id: int, uid: Optional[int] = None) -> Union[float, Dict[int, float]]:
        """EMA getter (old .get)."""
        with self._lock:
            cd = self._data.get(comp_id, CompetitionData())
            return cd.ema.get(uid) if uid is not None else cd.ema.get_all()

    def get_raw_scores(self, comp_id: int, uid: Optional[int] = None) -> Union[RawData, Dict[int, RawData], None]:
        """Raw scores getter (old .get_raw_scores)."""
        with self._lock:
            cd = self._data.get(comp_id, CompetitionData())
            if uid is not None:
                return cd.raw.get(uid)
            return dict(cd.raw)

    def combined_raw(self, block: int, uids: List[int]) -> Dict[int, Dict[str, float]]:
        """Weighted aggregation of nested/dict raw scores."""
        wmap = self.weight_source(block)
        out: Dict[int, Dict[str, float]] = {}
        with self._lock:
            for uid in uids:
                agg: Dict[str, float] = {}
                for cid, w in wmap.items():
                    rd = self._data.get(cid, CompetitionData()).raw.get(uid)
                    if rd is None: continue
                    if isinstance(rd.score, dict):
                        for k, v in rd.score.items():
                            agg[k] = agg.get(k, 0.0) + v * w
                    else:
                        agg["score"] = agg.get("score", 0.0) + float(rd.score) * w
                out[uid] = agg
        return out

    def get_combined_scores(self, block: int, uids: List[int]) -> Dict[int, float]:
        """Back-compat name mapping to combined_raw for scalar scores only."""
        nested = self.combined_raw(block, uids)
        return {uid: data.get("score", 0.0) for uid, data in nested.items()}

    def set_alpha(self, comp_id: int, new_alpha: float) -> None:
        """Back-compat setter."""
        with self._lock:
            self._comp(comp_id).ema.set_alpha(new_alpha)

    def remove_competition(self, comp_id: int) -> None:
        with self._lock:
            self._data.pop(comp_id, None)

    def clear_all(self) -> None:
        with self._lock:
            self._data.clear()
            self._uid_to_hotkey.clear()
            self._hotkey_to_uid.clear()

    def reset_competitions(self, keep: Set[int]) -> None:
        with self._lock:
            for cid in list(self._data):
                if cid not in keep:
                    del self._data[cid]

    def reset_uid(self, uid: int) -> None:
        with self._lock:
            for cd in self._data.values():
                cd.ema.reset(uid)
                if uid in cd.raw:
                    cd.raw[uid].score = 0.0

    def reset_score_for_hotkey(self, hotkey: str) -> None:
        uid = self.uid_for(hotkey)
        if uid is not None:
            self.reset_uid(uid)

    def subnet_weights(
        self,
        competitions: List[Competition],
        min_comp_weight_threshold: float = 0.0
    ) -> torch.Tensor:
        """Aggregate EMA-derived weight tensor per comp (old get_subnet_weights)."""
        out = torch.zeros(self.num_neurons, dtype=torch.float32)
        with self._lock:
            for comp in competitions:
                cid = comp.id
                cd = self.get_competition_data(cid)
                if not cd or cd.weights.numel() == 0:
                    continue
                t = cd.weights.clone().float()
                if min_comp_weight_threshold > 0:
                    mask = t < min_comp_weight_threshold
                    t[mask] = 0.0
                    s = t.sum().item() or 1.0
                    t /= s
                    t = t.nan_to_num(0.0)
                out += t * comp.reward_percentage
        s = out.sum().item() or 1.0
        return (out / s).nan_to_num(0.0)

    def record_competition_weights(self, comp_id: int, weights: torch.Tensor) -> None:
        with self._lock:
            self._comp(comp_id).weights = weights.clone().float()

    def get_competition_weights(self, comp_id: int) -> torch.Tensor:
        with self._lock:
            cd = self.get_competition_data(comp_id)
            return cd.weights.clone() if cd else torch.zeros(self.num_neurons, dtype=torch.float32)

    def uid_for(self, hotkey: str) -> Optional[int]:
        with self._lock:
            return self._hotkey_to_uid.get(hotkey)

    def hotkey_for(self, uid: int) -> Optional[str]:
        with self._lock:
            return self._uid_to_hotkey.get(uid)

    def get_last_updated_block(self, comp_id: int) -> Optional[int]:
        with self._lock:
            cd = self.get_competition_data(comp_id)
            return cd.last_updated_block if cd else None

    def save(self, path: str) -> None:
        with self._lock:
            state = {
                "num_neurons": self.num_neurons,
                "data": {
                    str(cid): {
                        "ema_scores": cd.ema.get_all(),
                        "ema_alpha": cd.ema.alpha,
                        "raw": {str(uid): {"score": rd.score, "block": rd.block} for uid, rd in cd.raw.items()},
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
            self.num_neurons = raw["num_neurons"]
            self.clear_all()
            for cid_str, entry in raw["data"].items():
                cid = int(cid_str)
                cd = self._comp(cid)
                cd.ema.set_alpha(entry.get("ema_alpha", 0.2))
                cd.ema._scores = {int(u): float(v) for u, v in entry["ema_scores"].items()}
                cd.raw = {int(u): RawData(score=e["score"], block=e["block"]) for u, e in entry["raw"].items()}
                cd.weights = torch.tensor(entry["weights"], dtype=torch.float32)
                cd.last_updated_block = entry.get("last_updated_block")
            self._uid_to_hotkey = {int(u): hk for u, hk in raw["hotkey_map"].items()}
            self._hotkey_to_uid = {hk: int(u) for u, hk in raw["hotkey_map"].items()}
    
    def get_competition_data(self, comp_id: int) -> Optional[CompetitionData]:
        with self._lock:
            return self._data.get(comp_id)
