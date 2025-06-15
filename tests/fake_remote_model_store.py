
import asyncio
from typing import Optional
from epochor.model.data import Model, ModelId
from epochor.model.competition.data import ModelConstraints
from epochor.model.base_hf_model_store import RemoteModelStore
from tests.test_disk_model_store import DummyModel, DummyConfig


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

    def set_model(self, model_id: ModelId, model: Model):
        """A method to manually set a model for a given model_id for testing."""
        model_key = f"{model_id.namespace}:{model_id.name}:{model_id.commit}"
        self.models[model_key] = model

    def set_throw_on_download(self, should_throw: bool):
        """Configure whether download_model should raise an exception."""
        self.throw_on_download = should_throw

