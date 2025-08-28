import dataclasses
import hashlib
from typing import Optional, Any
from temporal.models.base_model import BaseTemporalModel
import math

# The maximum bytes for metadata on the chain.
MAX_METADATA_BYTES = 128


@dataclasses.dataclass(frozen=True)
class ModelId:
    """
    Uniquely identifies a trained model and produces a byte-bounded string
    for on-chain commitment. It dynamically allocates space to ensure the
    full string never exceeds MAX_METADATA_BYTES.
    """

    # Core identifier fields
    namespace: str
    name: str
    competition_id: int

    # Optional fields that are part of the on-chain commitment
    commit: Optional[str] = dataclasses.field(default=None)
    secure_hash: Optional[str] = dataclasses.field(default=None)

    # Local-only field, not part of the compressed string
    hash: Optional[str] = dataclasses.field(default=None)


    def _shorten_repo_id(self, repo_id: str, budget: int) -> str:
        """
        Ensures the repo ID string fits within the given byte budget. If it's too long,
        it shortens the name part and appends a hash of the full original ID.
        """
        if len(repo_id.encode("utf-8")) <= budget:
            return repo_id

        # The hash tag will be 8 hex characters (4 bytes) plus a hyphen.
        hash_tag = hashlib.blake2b(repo_id.encode("utf-8"), digest_size=4).hexdigest()
        
        # Calculate available space for the name, accounting for namespace, slashes, and the hash tag.
        # We assume the format is "namespace/name"
        namespace, name = self.namespace, self.name
        available_for_name = budget - (len(namespace.encode("utf-8")) + 1 + len(hash_tag) + 1)
        
        if available_for_name <= 0:
            # If there's no space for the name, just use the namespace and hash.
            return f"{namespace}/{hash_tag}"
            
        # Safely slice the original name by characters until it fits the byte budget
        short_name = ""
        for char in name:
            if len((short_name + char).encode('utf-8')) > available_for_name:
                break
            short_name += char

        return f"{namespace}/{short_name}-{hash_tag}"


    def to_compressed_str(self) -> str:
        """
        Returns a compressed string representation that is guaranteed to be
        less than or equal to MAX_METADATA_BYTES.
        Format: "namespace/name:commit:secure_hash:competition_id"
        """
        # Get the variable-length components first.
        commit_str = self.commit or ""
        secure_hash_str = self.secure_hash or ""
        competition_id_str = str(self.competition_id)

        # Calculate the overhead for separators and other components.
        # There are 3 separators.
        overhead = 3 + len(commit_str.encode("utf-8")) + \
                   len(secure_hash_str.encode("utf-8")) + \
                   len(competition_id_str.encode("utf-8"))

        # The budget for the full repo_id ("namespace/name") is what's left.
        repo_id_budget = MAX_METADATA_BYTES - overhead
        
        full_repo_id = f"{self.namespace}/{self.name}"
        
        # Get a repo_id that is guaranteed to fit within the budget.
        repo_id_str = self._shorten_repo_id(full_repo_id, repo_id_budget)

        # Construct the final string.
        final_string = f"{repo_id_str}:{commit_str}:{secure_hash_str}:{competition_id_str}"
        
        # Final safety check to ensure we are under the limit. This should always pass.
        if len(final_string.encode("utf-8")) > MAX_METADATA_BYTES:
             raise ValueError(f"CRITICAL: Compressed string exceeds {MAX_METADATA_BYTES} bytes even after shortening.")
             
        return final_string

    @classmethod
    def from_compressed_str(cls, cs: str, default_competition_id: int = 0) -> "ModelId":
        """Instantiate from a compressed string representation."""
        tokens = cs.split(":")
        
        if len(tokens) < 4:
            raise ValueError(f"Invalid compressed string format: Expected at least 4 parts, got {len(tokens)} from '{cs}'")

        repo_id_str = tokens[0]
        commit = tokens[1] or None
        secure_hash = tokens[2] or None
        competition_id = int(tokens[3]) if tokens[3] else default_competition_id

        # Split the repo_id back into namespace and name.
        if "/" not in repo_id_str:
            raise ValueError(f"Invalid repo_id format in compressed string: '{repo_id_str}'")
        namespace, name = repo_id_str.split("/", 1)

        return cls(
            namespace=namespace,
            name=name,
            commit=commit,
            hash=None,  # Hash is not stored in the compressed string.
            secure_hash=secure_hash,
            competition_id=competition_id,
        )


@dataclasses.dataclass
class Model:
    """Represents a pre-trained foundation model."""
    id: ModelId
    model: BaseTemporalModel


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