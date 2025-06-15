
import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from neurons.validator import Validator
from epochor.model.data import ModelId, ModelMetadata, EvalResult, ScoreDetails
from epochor.competition import Competition, CompetitionId
from tests.test_disk_model_store import DummyModel, DummyConfig
import torch

class TestValidator(unittest.TestCase):

    def setUp(self):
        # Mock bittensor objects
        self.mock_wallet = MagicMock()
        self.mock_subtensor = MagicMock()
        self.mock_dendrite = MagicMock()
        self.mock_metagraph = MagicMock()
        self.mock_metagraph.hotkeys = ["hotkey0", "hotkey1", "hotkey2"]
        self.mock_metagraph.uids = [0, 1, 2]
        self.mock_metagraph.n = 3

        # Create a new, isolated event loop for each test
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Patch the Validator's __init__ to avoid real bittensor setup
        with patch('template.base.validator.BaseValidatorNeuron.__init__', MagicMock(return_value=None)):
            with patch('neurons.validator.ValidatorState'):
                 with patch('neurons.validator.ModelManager'):
                    with patch('neurons.validator.WeightSetter'):
                        self.validator = Validator()
        
        # Replace bittensor objects with mocks
        self.validator.wallet = self.mock_wallet
        self.validator.subtensor = self.mock_subtensor
        self.validator.dendrite = self.mock_dendrite
        self.validator.metagraph = self.mock_metagraph
        
        # Mock stores and tracker
        self.validator.metadata_store = AsyncMock()
        self.validator.remote_store = AsyncMock()
        self.validator.local_store = MagicMock()
        self.validator.state = MagicMock()
        self.validator.state.model_tracker = MagicMock()

    def tearDown(self):
        self.loop.close()

    def test_run_step_scoring_and_weight_update(self):
        async def run_test():
            # Setup a fake competition
            competition = Competition(
                id=CompetitionId.SN9_DATETIME,
                eval_tasks=[],
                constraints=MagicMock()
            )
            self.validator.global_step = 0
            
            # Mock the competition schedule to return our fake competition
            with patch('epochor.competition.utils.get_competition_schedule_for_block', return_value=[competition]):
                # Set some UIDs to evaluate
                uids_to_eval = {0, 1}
                self.validator.state.uids_to_eval = {competition.id: uids_to_eval}
                self.validator.state.pending_uids_to_eval = {competition.id: set()}

                # Mock model metadata for the UIDs
                metadata0 = ModelMetadata(id=ModelId(namespace="ns", name="n0", commit="c0", hash="h0", competition_id=competition.id), block=100)
                metadata1 = ModelMetadata(id=ModelId(namespace="ns", name="n1", commit="c1", hash="h1", competition_id=competition.id), block=101)
                self.validator.state.model_tracker.get_model_metadata_for_miner_hotkey.side_effect = lambda hotkey: metadata0 if hotkey == "hotkey0" else metadata1

                # Mock the local store to return models
                model0 = DummyModel(DummyConfig())
                model1 = DummyModel(DummyConfig())
                self.validator.local_store.retrieve_model.side_effect = lambda hotkey, model_id: model0 if hotkey == "hotkey0" else model1

                # Mock the scoring function to return predictable scores
                def mock_score_func(model, *args, **kwargs):
                    if model == model0:
                        return 0.1, {"task1": ScoreDetails(score=0.1)} # Lower score is better
                    else:
                        return 0.2, {"task1": ScoreDetails(score=0.2)}

                with patch('epochor.validation.validation.score_time_series_model', side_effect=mock_score_func):
                    # Mock other necessary methods
                    self.validator._get_current_block = MagicMock(return_value=200)
                    self.validator._get_seed = MagicMock(return_value=123)
                    self.validator.state.ema_tracker = MagicMock()
                    self.validator.state.ema_tracker.get_subnet_weights.return_value = torch.zeros(3)


                    # Run the step
                    await self.validator.run_step()
                    
                    # Assertions
                    # Check that the model tracker was updated with evaluation results
                    self.assertEqual(self.validator.state.model_tracker.on_model_evaluated.call_count, 2)
                    
                    # Check that EMA scores were updated
                    self.validator.state.update_ema_scores.assert_called_once()
                    
                    # Check that competition weights were recorded
                    self.validator.state.ema_tracker.record_competition_weights.assert_called_once()
                    
                    # Check that uids to eval were updated
                    self.validator.state.update_uids_to_eval.assert_called_once()

        self.loop.run_until_complete(run_test())

if __name__ == '__main__':
    unittest.main()
