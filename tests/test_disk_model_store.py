
import os
import torch
import pytest
from epochor.model.storage.disk_model_store import DiskModelStore
from epochor.model.model_data import Model, ModelId
from epochor.model.model_constraints import ModelConstraints
from tests.helpers import DummyModel, DummyConfig

def test_store_and_retrieve_model(temp_dir, dummy_model):
    """Tests that a model can be stored and then retrieved."""
    store = DiskModelStore(base_dir=temp_dir)
    hotkey = "test_hotkey"
    model_id = ModelId(namespace="test_namespace", name="test_name", commit="test_commit", hash="test_hash", competition_id=1)

    # Store the model
    stored_model_id = store.store_model(hotkey, Model(id=model_id, model=dummy_model))

    # Check if the model is stored
    model_dir = store.get_path(hotkey)
    assert os.path.exists(model_dir)

    # Retrieve the model
    retrieved_model = store.retrieve_model(
        hotkey,
        stored_model_id,
        model_constraints=ModelConstraints(model_cls=DummyModel, config_cls=DummyConfig)
    )

    # Check if the retrieved model is the same as the original model
    assert retrieved_model.id == stored_model_id
    assert torch.equal(list(retrieved_model.model.parameters())[0], list(dummy_model.parameters())[0])

def test_retrieve_nonexistent_model(temp_dir):
    """Tests that retrieving a non-existent model raises an exception."""
    store = DiskModelStore(base_dir=temp_dir)
    hotkey = "test_hotkey"
    model_id = ModelId(namespace="test_namespace", name="test_name", commit="test_commit", hash="nonexistent_hash", competition_id=1)

    with pytest.raises(Exception):
        store.retrieve_model(
            hotkey,
            model_id,
            model_constraints=ModelConstraints(model_cls=DummyModel, config_cls=DummyConfig)
        )

def test_delete_unreferenced_models(temp_dir, dummy_model):
    """Tests that unreferenced models are deleted correctly."""
    store = DiskModelStore(base_dir=temp_dir)
    hotkey = "test_hotkey"

    # Store two models
    model_id_1 = ModelId(namespace="test_namespace", name="test_name_1", commit="test_commit_1", hash="test_hash_1", competition_id=1)
    stored_model_id_1 = store.store_model(hotkey, Model(id=model_id_1, model=dummy_model))

    model_id_2 = ModelId(namespace="test_namespace", name="test_name_2", commit="test_commit_2", hash="test_hash_2", competition_id=1)
    stored_model_id_2 = store.store_model(hotkey, Model(id=model_id_2, model=dummy_model))

    # Call delete_unreferenced_models with only the first model as valid
    store.delete_unreferenced_models({hotkey: {stored_model_id_1}}, grace_period_seconds=0)

    # Check that the first model is not deleted
    retrieved_model_1 = store.retrieve_model(
        hotkey,
        stored_model_id_1,
        model_constraints=ModelConstraints(model_cls=DummyModel, config_cls=DummyConfig)
    )
    assert retrieved_model_1 is not None

    # Check that the second model is deleted
    with pytest.raises(Exception):
        store.retrieve_model(
            hotkey,
            stored_model_id_2,
            model_constraints=ModelConstraints(model_cls=DummyModel, config_cls=DummyConfig)
        )
