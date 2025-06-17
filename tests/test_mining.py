
import unittest
import tempfile
import shutil
import os
import asyncio
from unittest.mock import MagicMock, AsyncMock
from epochor import mining
from epochor.model.model_data import Model, ModelId
from tests.test_disk_model_store import DummyModel, DummyConfig
from constants import CompetitionId
import constants

class TestMining(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = DummyConfig()
        self.model = DummyModel(self.config)
        self.mock_wallet = MagicMock()
        self.mock_wallet.hotkey.ss58_address = "fake_hotkey_address"
        constants.SUBNET_UID = 1 


    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_save_and_load_local_model(self):
        # Save the model
        model_dir = os.path.join(self.temp_dir, "model_test")
        mining.save(self.model, model_dir)

        # Check that the model files were created
        self.assertTrue(os.path.exists(os.path.join(model_dir, "config.json")))
        self.assertTrue(os.path.exists(os.path.join(model_dir, "model.safetensors")))

        # Load the model back
        loaded_model = mining.load_local_model(model_dir, DummyModel)

        # Check if the loaded model is of the correct type and has the same state
        self.assertIsInstance(loaded_model, DummyModel)
        # Note: A more thorough check would involve comparing the state dicts
        self.assertEqual(
            self.model.state_dict()['linear.weight'].ne(loaded_model.state_dict()['linear.weight']).sum(),
            0
        )

    def test_push_model(self):
        async def run_test():
            # Mock the external dependencies
            mock_metadata_store = AsyncMock()
            mock_remote_store = AsyncMock()

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
                model=self.model,
                repo="test_ns/test_repo",
                wallet=self.mock_wallet,
                competition_id=CompetitionId.UNIVARIATE,
                metadata_store=mock_metadata_store,
                remote_model_store=mock_remote_store
            )

            # Assert that the external stores were called
            mock_remote_store.upload_model.assert_called_once()
            mock_metadata_store.store_model_metadata.assert_called_once()
            mock_metadata_store.retrieve_model_metadata.assert_called_once()
        
        asyncio.run(run_test())

if __name__ == '__main__':
    unittest.main()
