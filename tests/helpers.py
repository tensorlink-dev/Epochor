from dataclasses import dataclass
from typing import Dict, Optional

import os

import torch

from epochor.model.base import BaseTemporalModel

from epochor.model.base_hf_model_store import RemoteModelStore
from epochor.model.base_metadata_model_store import ModelMetadataStore
from epochor.model.model_constraints import ModelConstraints
from epochor.model.model_data import Model, ModelId, ModelMetadata


@dataclass
class DummyConfig:
    """Lightweight config object used for dummy models."""

    input_dim: int = 1


class DummyModel(BaseTemporalModel):
    config_class = DummyConfig

    def __init__(self, config: Optional[DummyConfig] = None):
        config = config or DummyConfig()
        super().__init__(config)
        input_dim = getattr(config, "input_dim", 1)
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        predictions = self.linear(x)
        return self._to_output({"predictions": predictions})


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
            model = self.models[model_key]
            if model.source_path is None:
                os.makedirs(local_path, exist_ok=True)
                model.source_path = local_path
            return model

        config = DummyConfig()
        model = DummyModel(config)
        os.makedirs(local_path, exist_ok=True)
        return Model(id=model_id, model=model, source_path=local_path)


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
