
import os
import asyncio
from unittest.mock import MagicMock, AsyncMock
import pytest
from epochor import mining
from epochor.model.model_data import ModelId
from tests.helpers import DummyModel
from constants import CompetitionId
import constants

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio

def test_save_and_load_local_model(temp_dir, dummy_model):
    """Tests that a model can be saved and loaded locally."""
    # Save the model
    model_dir = os.path.join(temp_dir, "model_test")
    mining.save(dummy_model, model_dir)

    # Check that the model files were created
    assert os.path.exists(os.path.join(model_dir, "config.json"))
    assert os.path.exists(os.path.join(model_dir, "model.safetensors"))

    # Load the model back
    loaded_model = mining.load_local_model(model_dir, CompetitionId.UNIVARIATE)

    # Check if the loaded model is of the correct type and has the same state
    assert isinstance(loaded_model, DummyModel)
    # Note: A more thorough check would involve comparing the state dicts
    assert dummy_model.state_dict()['linear.weight'].ne(loaded_model.state_dict()['linear.weight']).sum() == 0

async def test_push_model(dummy_model, mock_wallet):
    """Tests the model push process with mocked external stores."""
    # Mock the external dependencies
    mock_metadata_store = AsyncMock()
    mock_remote_store = AsyncMock()
    constants.SUBNET_UID = 1

    # Configure the mock remote store to return a specific ModelId
    test_model_id = ModelId(namespace="test_ns", name="test_repo", commit="test_commit", hash="test_hash", competition_id=CompetitionId.UNIVARIATE)
    mock_remote_store.upload_model.return_value = test_model_id
    
    # Configure the mock metadata store to successfully retrieve the metadata after storing
    async def retrieve_side_effect(uid, hotkey):
        # Simulate that the metadata was stored correctly
        return MagicMock(id=test_model_id)
    mock_metadata_store.retrieve_model_metadata.side_effect = retrieve_side_effect

    # Call the push function
    await mining.push(
        model=dummy_model,
        repo="test_ns/test_repo",
        wallet=mock_wallet,
        competition_id=CompetitionId.UNIVARIATE,
        metadata_store=mock_metadata_store,
        remote_model_store=mock_remote_store
    )

    # Assert that the external stores were called
    mock_remote_store.upload_model.assert_called_once()
    mock_metadata_store.store_model_metadata.assert_called_once()
    mock_metadata_store.retrieve_model_metadata.assert_called_once()
