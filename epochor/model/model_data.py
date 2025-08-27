import dataclasses
from typing import ClassVar, Optional
import hashlib
import math

from temporal.models.base_model import BaseTemporalModel

# The maximum bytes for metadata on the chain.
MAX_METADATA_BYTES = 128
# The length, in bytes, of a git commit hash.
GIT_COMMIT_LENGTH = 40
# The length, in bytes, of a base64 encoded sha256 hash.
SHA256_BASE_64_LENGTH = 44
# The max length, in characters, of the competition id (as stringified int)
MAX_COMPETITION_ID_LENGTH = 2


@dataclasses.dataclass(frozen=True)
class ModelId:
    """Uniquely identifies a trained model and produces a bounded on-chain commitment."""

    # Budget for "namespace:name" inside the full compressed string.
    MAX_REPO_ID_LENGTH: ClassVar[int] = (
        MAX_METADATA_BYTES
        - GIT_COMMIT_LENGTH
        - SHA256_BASE_64_LENGTH
        - MAX_COMPETITION_ID_LENGTH
        - 4  # separators between fields: ns : name : commit : secure : comp
    )

    # Namespace where the model can be found. ex. Hugging Face username/org.
    namespace: str

    # Name of the model (repo name).
    name: str

    # Identifier for competition (int on-chain)
    competition_id: int

    # Commit must be filled when trying to download from a remote store.
    commit: Optional[str] = dataclasses.field(default=None)

    # Hash is filled automatically when uploading to or downloading from a remote store.
    # (Typically a content hash of the uploaded artifact; not used on-chain directly.)
    hash: Optional[str] = dataclasses.field(default=None)

    # The secure hash that's used for validation. Prefer a fixed-size base64(SHA256)=44 chars.
    secure_hash: Optional[str] = dataclasses.field(default=None)

    # ----------------------------
    # Internal helpers (byte-safe)
    # ----------------------------
    @staticmethod
    def _utf8_len(s: str) -> int:
        return len(s.encode("utf-8"))

    @staticmethod
    def _trim_with_tag(s: str, max_bytes: int, tag_src: bytes) -> str:
        """
        If s fits in max_bytes (UTF-8), return s.
        Else keep as much prefix as fits and append '-<8hex>' (blake2b(tag_src, 4B)).
        Deterministic and collision-resistant enough for display IDs.
        """
        b = s.encode("utf-8")
        if len(b) <= max_bytes:
            return s

        tag = hashlib.blake2b(tag_src, digest_size=4).hexdigest()  # 8 hex chars
        overhead = 1 + len(tag)  # "-" + tag
        if max_bytes <= overhead:
            # No room for prefix; return as much of the tag as fits
            return tag[:max_bytes]

        keep_bytes = max_bytes - overhead
        out, used = [], 0
        for ch in s:
            cb = ch.encode("utf-8")
            if used + len(cb) > keep_bytes:
                break
            out.append(ch)
            used += len(cb)
        return "".join(out) + "-" + tag

    def _shorten_repo(self) -> tuple[str, str]:
        """
        Ensure f"{namespace}:{name}" fits within MAX_REPO_ID_LENGTH bytes.
        If not, trim BOTH sides proportionally (with small floors) and tag them.
        This is UTF-8 byte-aware, so non-ASCII names are safe.
        """
        ns, nm = self.namespace, self.name
        budget = self.MAX_REPO_ID_LENGTH
        raw = f"{ns}:{nm}"
        if self._utf8_len(raw) <= budget:
            return ns, nm

        sep_bytes = 1  # the ':'
        b_ns, b_nm = self._utf8_len(ns), self._utf8_len(nm)
        total = max(1, b_ns + b_nm)
        avail = budget - sep_bytes

        # Keep a little readability on each side
        MIN_NS, MIN_NM = 8, 8
        ns_budget = max(MIN_NS, (avail * b_ns) // total)
        nm_budget = max(MIN_NM, avail - ns_budget)
        if ns_budget + nm_budget > avail:
            nm_budget = avail - ns_budget

        tag_src = f"{ns}:{nm}".encode("utf-8")
        ns2 = self._trim_with_tag(ns, ns_budget, tag_src)
        nm2 = self._trim_with_tag(nm, nm_budget, tag_src)

        assert self._utf8_len(f"{ns2}:{nm2}") <= budget
        return ns2, nm2

    # ----------------------------
    # Public API
    # ----------------------------
    def to_compressed_str(self) -> str:
        """
        Returns a colon-separated string that is ALWAYS ≤ MAX_METADATA_BYTES bytes:
            namespace : name : commit(40 or 'None') : secure_hash(44 or 'None') : competition_id
        """
        ns, nm = self._shorten_repo()
        commit = self.commit or "None"
        secure = self.secure_hash or "None"
        comp = str(self.competition_id)

        s = f"{ns}:{nm}:{commit}:{secure}:{comp}"
        if len(s.encode("utf-8")) <= MAX_METADATA_BYTES:
            return s

        # Ultra-rare fallback: if anything drifts, compress further with a digest tag.
        tag = hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest()  # 32 hex
        s2 = f"{ns}:{tag}:{comp}"
        assert len(s2.encode("utf-8")) <= MAX_METADATA_BYTES
        return s2

    @classmethod
    def from_compressed_str(
        cls, cs: str, default_competition_id: int = 0
    ) -> "ModelId":
        """
        Instantiate from a compressed string representation.
        Backward‐compat: older format lacked explicit competition_id.
        """
        tokens = cs.split(":")

        # Backward-compat (len < 5) implies: ns, name, commit, secure_hash
        if len(tokens) < 5:
            competition_id = default_competition_id
            secure_hash = tokens[3] if len(tokens) > 3 and tokens[3] != "None" else None
        else:
            competition_id = int(tokens[4])
            secure_hash = tokens[3] if tokens[3] != "None" else None

        return cls(
            namespace=tokens[0] if len(tokens) > 0 else "",
            name=tokens[1] if len(tokens) > 1 else "",
            commit=(tokens[2] if len(tokens) > 2 and tokens[2] != "None" else None),
            hash=None,
            secure_hash=secure_hash,
            competition_id=competition_id,
        )


@dataclasses.dataclass
class Model:
    """Represents a pre-trained foundation model."""

    # Identifier for this model.
    id: ModelId

    # The raw model object (e.g. a torch.nn.Module or any other class).
    model: BaseTemporalModel

    # Tokenizer is no longer managed by stores; always None.


@dataclasses.dataclass
class ModelMetadata:
    # Identifier for this trained model.
    id: ModelId

    # Block on which this model was uploaded on the chain.
    block: int


@dataclasses.dataclass
class EvalResult:
    """Records an evaluation result for a model."""

    # The block the model was evaluated at.
    block: int

    # The eval score of this model when it was evaluated.
    # May be math.inf if the model failed to evaluate.
    score: float

    # The block the winning model was submitted.
    winning_model_block: int

    # The score of the winning model when this model was evaluated.
    winning_model_score: float
