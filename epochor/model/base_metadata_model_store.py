import abc
from typing import Optional
from epochor.model.model_data import ModelId, ModelMetadata


class ModelMetadataStore(abc.ABC):
    """An abstract base class for storing and retrieving model metadata."""

    @abc.abstractmethod
    async def store_model_metadata(
        self,
        hotkey: str,
        model_id: ModelId,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        ttl: int = 60,
    ):
        """Stores model metadata on this subnet for a specific miner."""
        pass

    @abc.abstractmethod
    async def retrieve_model_metadata(self, uid: int, hotkey: str, ttl: int = 60) -> Optional[ModelMetadata]:
        """Retrieves model metadata + block information on this subnet for specific miner, if present"""
        pass