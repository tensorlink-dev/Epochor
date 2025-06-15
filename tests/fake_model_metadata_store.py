
import asyncio
from typing import Dict, Optional
from epochor.model.data import ModelId, ModelMetadata
from epochor.model.base_metadata_model_store import ModelMetadataStore

class FakeModelMetadataStore(ModelMetadataStore):
    """A fake implementation of the ModelMetadataStore for testing purposes."""

    def __init__(self):
        self.metadata: Dict[str, ModelMetadata] = {}

    async def store_model_metadata(
        self,
        hotkey: str,
        model_id: ModelId,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        ttl: int = 60,
    ):
        """Stores model metadata in the fake store."""
        # In a real implementation, block and confirmation would be handled.
        # Here, we just store it directly.
        self.metadata[hotkey] = ModelMetadata(id=model_id, block=1)

    async def retrieve_model_metadata(self, uid: int, hotkey: str, ttl: int = 60) -> Optional[ModelMetadata]:
        """Retrieves model metadata from the fake store."""
        return self.metadata.get(hotkey)

    def set_metadata(self, hotkey: str, metadata: Optional[ModelMetadata]):
        """A method to manually set metadata for a given hotkey for testing."""
        if metadata:
            self.metadata[hotkey] = metadata
        elif hotkey in self.metadata:
            del self.metadata[hotkey]
