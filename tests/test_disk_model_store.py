
import unittest
import tempfile
import shutil
import os
import torch
from transformers import PretrainedConfig, PreTrainedModel
from epochor.model.storage.disk_model_store import DiskModelStore
from epochor.model.model_data import Model, ModelId
from epochor.model.model_constraints import ModelConstraints

# Dummy model and config for testing
class DummyConfig(PretrainedConfig):
    model_type = "dummy"

    def __init__(self, hidden_size=1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size

class DummyModel(PreTrainedModel):
    config_class = DummyConfig

    def __init__(self, config):
        super().__init__(config)
        self.linear = torch.nn.Linear(config.hidden_size, 1)

    def forward(self, x):
        return self.linear(x)

class TestDiskModelStore(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _create_dummy_model_and_config(self):
        config = DummyConfig()
        model = DummyModel(config)
        return model, config

    def test_store_and_retrieve_model(self):
        store = DiskModelStore(base_dir=self.temp_dir)
        hotkey = "test_hotkey"
        model, config = self._create_dummy_model_and_config()
        model_id = ModelId(namespace="test_namespace", name="test_name", commit="test_commit", hash="test_hash", competition_id=1)

        # Store the model
        stored_model_id = store.store_model(hotkey, Model(id=model_id, model=model))

        # Check if the model is stored
        model_dir = store.get_path(hotkey)
        self.assertTrue(os.path.exists(model_dir))

        # Retrieve the model
        retrieved_model = store.retrieve_model(
            hotkey,
            stored_model_id,
            model_constraints=ModelConstraints(model_cls=DummyModel, config_cls=DummyConfig)
        )

        # Check if the retrieved model is the same as the original model
        self.assertEqual(retrieved_model.id, stored_model_id)
        self.assertTrue(torch.equal(list(retrieved_model.model.parameters())[0], list(model.parameters())[0]))

    def test_retrieve_nonexistent_model(self):
        store = DiskModelStore(base_dir=self.temp_dir)
        hotkey = "test_hotkey"
        model_id = ModelId(namespace="test_namespace", name="test_name", commit="test_commit", hash="nonexistent_hash", competition_id=1)

        with self.assertRaises(Exception):
            store.retrieve_model(
                hotkey,
                model_id,
                model_constraints=ModelConstraints(model_cls=DummyModel, config_cls=DummyConfig)
            )

    def test_delete_unreferenced_models(self):
        store = DiskModelStore(base_dir=self.temp_dir)
        hotkey = "test_hotkey"
        model, config = self._create_dummy_model_and_config()

        # Store two models
        model_id_1 = ModelId(namespace="test_namespace", name="test_name_1", commit="test_commit_1", hash="test_hash_1", competition_id=1)
        stored_model_id_1 = store.store_model(hotkey, Model(id=model_id_1, model=model))

        model_id_2 = ModelId(namespace="test_namespace", name="test_name_2", commit="test_commit_2", hash="test_hash_2", competition_id=1)
        stored_model_id_2 = store.store_model(hotkey, Model(id=model_id_2, model=model))

        # Call delete_unreferenced_models with only the first model as valid
        store.delete_unreferenced_models({hotkey: {stored_model_id_1}}, grace_period_seconds=0)

        # Check that the first model is not deleted
        retrieved_model_1 = store.retrieve_model(
            hotkey,
            stored_model_id_1,
            model_constraints=ModelConstraints(model_cls=DummyModel, config_cls=DummyConfig)
        )
        self.assertIsNotNone(retrieved_model_1)

        # Check that the second model is deleted
        with self.assertRaises(Exception):
            store.retrieve_model(
                hotkey,
                stored_model_id_2,
                model_constraints=ModelConstraints(model_cls=DummyModel, config_cls=DummyConfig)
            )

if __name__ == "__main__":
    unittest.main()
