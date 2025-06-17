import dataclasses
from typing import ClassVar, Optional, Any
from temporal.models.base_model import BaseTemporalModel
import math

# The maximum bytes for metadata on the chain.
MAX_METADATA_BYTES = 128
# The length, in bytes, of a git commit hash.
GIT_COMMIT_LENGTH = 40
# The length, in bytes, of a base64 encoded sha256 hash.
SHA256_BASE_64_LENGTH = 44
# The max length, in characters, of the competition id
MAX_COMPETITION_ID_LENGTH = 2


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

    def to_compressed_str(self) -> str:
        """Returns a compressed string representation."""
        return f"{self.namespace}:{self.name}:{self.commit}:{self.secure_hash}:{self.competition_id}"

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


@dataclasses.dataclass
class Model:
    """Represents a pre‐trained foundation model."""

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


@dataclasses.dataclass
class ScoreDetails:
    """Additional details about a score."""

    # The score for this task.
    score: float = math.inf
