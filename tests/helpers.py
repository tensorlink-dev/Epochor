
import asyncio
from typing import Dict, Optional

import torch
from transformers import PretrainedConfig, PreTrainedModel

from epochor.model.base_hf_model_store import RemoteModelStore
from epochor.model.base_metadata_model_store import ModelMetadataStore
from epochor.model.competition.data import ModelConstraints
from epochor.model.data import Model, ModelId, ModelMetadata


# Dummy model and config for testing
class DummyConfig(PretrainedConfig):
    model_type = "dummy"

    def __init__(self, hidden_size=1, input_dim=1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.input_dim = input_dim


class DummyModel(PreTrainedModel):
    config_class = DummyConfig

    def __init__(self, config):
        super().__init__(config)
        self.linear = torch.nn.Linear(config.hidden_size, 1)

    def forward(self, x):
        return self.linear(x)


class FakeRemoteModelStore(RemoteModelStore):
    """A fake implementation of the RemoteModelStore for testing purposes."""

    def __init__(self, models: dict[str, Model] = None, throw_on_download: bool = False):
        self.models = models if models else {}
        self.throw_on_download = throw_on_download

    async def download_model(
        self, model_id: ModelId, local_path: str, model_constraints: ModelConstraints
    ) -> Model:
        """Retrieves a model from the fake store."""
        if self.throw_on_download:
            raise ValueError("Forced download error")

        model_key = f"{model_id.namespace}:{model_id.name}:{model_id.commit}"
        if model_key in self.models:
            return self.models[model_key]

        # If not found, create a dummy model to return
        config = DummyConfig()
        model = DummyModel(config)
        return Model(id=model_id, model=model)


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
        self.metadata[hotkey] = ModelMetadata(id=model_id, block=1)

    async def retrieve_model_metadata(self, uid: int, hotkey: str, ttl: int = 60) -> Optional[ModelMetadata]:
        """Retrieves model metadata from the fake store."""
        return self.metadata.get(hotkey)
