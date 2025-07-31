
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import pytest
import bittensor as bt
import torch
from neurons.validator import Validator
from epochor.validation.validation import ScoreDetails
from epochor.model.model_data import ModelId, ModelMetadata
from epochor.model.model_constraints import Competition
from competitions.competitions import CompetitionId # Updated import
from tests.helpers import DummyModel, DummyConfig

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio

class ConcreteValidator(Validator):
    """A concrete implementation of the abstract Validator for testing."""
    def __init__(self, config):
        super().__init__(config)

    async def forward(self, synapse: bt.Synapse) -> bt.Synapse:
        pass

@pytest.fixture
def validator(mock_wallet):
    """Provides a mocked Validator instance for testing."""
    mock_config = MagicMock()
    mock_config.validator_config.return_value = mock_config
    mock_config.wallet.name = "mock_wallet"
    mock_config.wallet.hotkey = "mock_hotkey"

    with patch('template.base.validator.BaseValidatorNeuron.__init__', MagicMock(return_value=None)):
        with patch('neurons.validator.ValidatorState'):
            with patch('neurons.validator.ModelManager'):
                with patch('neurons.validator.WeightSetter'):
                    validator = ConcreteValidator(config=mock_config)
    
    validator.wallet = mock_wallet
    validator.subtensor = MagicMock()
    validator.dendrite = MagicMock()
    validator.metagraph = MagicMock()
    validator.metagraph.hotkeys = ["hotkey0", "hotkey1", "hotkey2"]
    validator.metagraph.uids = [0, 1, 2]
    validator.metagraph.n = 3
    
    validator.metadata_store = AsyncMock()
    validator.remote_store = AsyncMock()
    validator.local_store = MagicMock()
    validator.state = MagicMock()
    validator.state.model_tracker = MagicMock()
    validator.config = mock_config
    
    return validator

async def test_run_step_scoring_and_weight_update(validator, dummy_model):
    """Tests the main validator step, including scoring and weight updates."""
    # Setup a fake competition
    competition = Competition(
        id=CompetitionId.UNIVARIATE,
        eval_tasks=[],
        constraints=MagicMock()
    )
    validator.global_step = 0
    
    # Mock the competition schedule to return our fake competition
    with patch('epochor.utils.competition_utils.get_competition_schedule_for_block', return_value=[competition]):
        # Set some UIDs to evaluate
        uids_to_eval = {0, 1}
        validator.state.uids_to_eval = {competition.id: uids_to_eval}
        validator.state.pending_uids_to_eval = {competition.id: set()}

        # Mock model metadata for the UIDs
        metadata0 = ModelMetadata(id=ModelId(namespace="ns", name="n0", commit="c0", hash="h0", competition_id=competition.id), block=100)
        metadata1 = ModelMetadata(id=ModelId(namespace="ns", name="n1", commit="c1", hash="h1", competition_id=competition.id), block=101)
        validator.state.model_tracker.get_model_metadata_for_miner_hotkey.side_effect = lambda hotkey: metadata0 if hotkey == "hotkey0" else metadata1

        # Mock the local store to return models
        model0 = dummy_model
        model1 = DummyModel(DummyConfig()) # A different instance
        validator.local_store.retrieve_model.side_effect = lambda hotkey, model_id, constraints: model0 if hotkey == "hotkey0" else model1

        # Mock the scoring function to return predictable scores
        def mock_score_func(model, *args, **kwargs):
            if model == model0:
                return 0.1, {"task1": ScoreDetails(raw_score=0.1)} # Lower score is better
            else:
                return 0.2, {"task1": ScoreDetails(raw_score=0.2)}

        with patch('epochor.validation.validation.score_time_series_model', side_effect=mock_score_func):
            # Mock other necessary methods
            validator._get_current_block = MagicMock(return_value=200)
            validator._get_seed = MagicMock(return_value=123)
            validator.state.ema_tracker = MagicMock()
            validator.state.ema_tracker.get_subnet_weights.return_value = torch.zeros(3)

            # Run the step
            await validator.run_step()
            
            # Assertions
            # Check that the model tracker was updated with evaluation results
            assert validator.state.model_tracker.on_model_evaluated.call_count == 2
            
            # Check that EMA scores were updated
            validator.state.update_ema_scores.assert_called_once()
            
            # Check that competition weights were recorded
            validator.state.ema_tracker.record_competition_weights.assert_called_once()
            
            # Check that uids to eval were updated
            validator.state.update_uids_to_eval.assert_called_once()
