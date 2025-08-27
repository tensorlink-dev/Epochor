# add this import near the top with the others
import hashlib
import dataclasses

@dataclasses.dataclass(frozen=True)
class ModelId:
    """Uniquely identifies a trained model"""

    MAX_REPO_ID_LENGTH: ClassVar[int] = (
        MAX_METADATA_BYTES
        - GIT_COMMIT_LENGTH
        - SHA256_BASE_64_LENGTH
        - MAX_COMPETITION_ID_LENGTH
        - 4  # separators
    )

    # Namespace where the model can be found. ex. Hugging Face username/org.
    namespace: str
    # Name of the model.
    name: str
    # Identifier for competition
    competition_id: int

    # Commit must be filled when trying to download from a remote store.
    commit: Optional[str] = dataclasses.field(default=None)
    # Hash is filled automatically when uploading to or downloading from a remote store.
    hash: Optional[str] = dataclasses.field(default=None)
    # The secure hash that's used for validation.
    secure_hash: Optional[str] = dataclasses.field(default=None)

    # ---------- NEW: helpers to keep strings within byte budgets ----------

    @staticmethod
    def _utf8_len(s: str) -> int:
        return len(s.encode("utf-8"))

    @staticmethod
    def _trim_with_tag(s: str, max_bytes: int, tag_src: bytes) -> str:
        """
        If s fits in max_bytes (UTF-8), return s.
        Else keep as much prefix as fits and append '-<8hex>' (blake2b(tag_src, 4B)).
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
        """
        ns, nm = self.namespace, self.name
        budget = self.MAX_REPO_ID_LENGTH
        raw = f"{ns}:{nm}"
        if self._utf8_len(raw) <= budget:
            return ns, nm

        sep_bytes = 1
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

    # ---------- UPDATED: bounded compressed form ----------

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
        """Instantiate from a compressed string representation."""
        tokens = cs.split(":")

        # Backward‐compat: older format lacked explicit competition_id
        if len(tokens) < 5:
            competition_id = default_competition_id
            secure_hash = tokens[3] if tokens[3] != "None" else None
        else:
            competition_id = int(tokens[4])
            secure_hash = tokens[3] if tokens[3] != "None" else None

        return cls(
            namespace=tokens[0],
            name=tokens[1],
            commit=tokens[2] if tokens[2] != "None" else None,
            hash=None,
            secure_hash=secure_hash,
            competition_id=competition_id,
        )
