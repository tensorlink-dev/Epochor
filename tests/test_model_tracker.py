
import os
import pytest
from epochor.model.model_tracker import ModelTracker
from epochor.model.model_data import (
    EvalResult,
    MinerSubmissionSnapshot,
    ModelId,
    TrainingResultRecord,
)

@pytest.fixture
def tracker():
    """Provides a ModelTracker instance pre-populated with some data."""
    tracker = ModelTracker()
    model_id1 = ModelId(namespace="test_ns", name="test_name1", commit="c1", hash="h1", competition_id=1)
    snapshot1 = MinerSubmissionSnapshot(model_id=model_id1, competition_id=1, block=100, snapshot_path="/tmp/path1")
    model_id2 = ModelId(namespace="test_ns", name="test_name2", commit="c2", hash="h2", competition_id=1)
    snapshot2 = MinerSubmissionSnapshot(model_id=model_id2, competition_id=1, block=101, snapshot_path="/tmp/path2")

    tracker.on_submission_updated("hotkey1", snapshot1)
    tracker.on_submission_updated("hotkey2", snapshot2)
    return tracker

def test_on_hotkeys_updated(tracker):
    """Tests that the tracker correctly removes hotkeys that are no longer active."""
    assert tracker.get_submission_for_miner_hotkey("hotkey1") is not None
    assert tracker.get_submission_for_miner_hotkey("hotkey2") is not None

    tracker.on_hotkeys_updated({"hotkey1"})

    assert tracker.get_submission_for_miner_hotkey("hotkey1") is not None
    assert tracker.get_submission_for_miner_hotkey("hotkey2") is None

def test_on_model_updated(tracker):
    """Tests that model updates are correctly stored."""
    submission = tracker.get_submission_for_miner_hotkey("hotkey1")
    assert submission is not None
    assert submission.model_id.namespace == "test_ns"
    assert submission.block == 100

def test_on_model_evaluated_and_history(tracker):
    """Tests that evaluation results are recorded correctly."""
    eval_result1 = EvalResult(block=110, score=0.9, winning_model_block=0, winning_model_score=0)
    tracker.on_model_evaluated("hotkey1", 0, eval_result1)
    
    eval_results = tracker.get_eval_results_for_miner_hotkey("hotkey1", 0)
    assert len(eval_results) == 1
    assert eval_results[0] == eval_result1
    
    eval_result2 = EvalResult(block=120, score=0.95, winning_model_block=0, winning_model_score=0)
    tracker.on_model_evaluated("hotkey1", 0, eval_result2)

    eval_results = tracker.get_eval_results_for_miner_hotkey("hotkey1", 0)
    assert len(eval_results) == 2
    assert eval_results[0] == eval_result1
    assert eval_results[1] == eval_result2

def test_get_block_last_evaluated(tracker):
    """Tests that the last evaluation block is tracked correctly across competitions."""
    assert tracker.get_block_last_evaluated("hotkey1") is None
    
    eval_result1 = EvalResult(block=110, score=0.9, winning_model_block=0, winning_model_score=0)
    tracker.on_model_evaluated("hotkey1", 0, eval_result1)
    
    assert tracker.get_block_last_evaluated("hotkey1") == 110

    eval_result2 = EvalResult(block=120, score=0.95, winning_model_block=0, winning_model_score=0)
    tracker.on_model_evaluated("hotkey1", 1, eval_result2)

    # The block number should be the max across all competitions
    assert tracker.get_block_last_evaluated("hotkey1") == 120
    
    eval_result3 = EvalResult(block=115, score=0.98, winning_model_block=0, winning_model_score=0)
    tracker.on_model_evaluated("hotkey1", 0, eval_result3)

    assert tracker.get_block_last_evaluated("hotkey1") == 120

def test_save_and_load_state(tracker, temp_dir):
    """Tests that the tracker's state can be saved and loaded."""
    filepath = os.path.join(temp_dir, "tracker_state.pkl")
    
    # Add some eval results to save
    eval_result = EvalResult(block=110, score=0.9, winning_model_block=0, winning_model_score=0)
    tracker.on_model_evaluated("hotkey1", 0, eval_result)
    
    train_result = TrainingResultRecord(
        competition_id=1,
        block=100,
        train_metrics={"loss": 0.1},
        val_metrics={"val_loss": 0.2},
        num_steps=10,
        device="cpu",
    )
    tracker.record_training_result("hotkey1", train_result)

    tracker.save_state(filepath)
    assert os.path.exists(filepath)

    new_tracker = ModelTracker()
    new_tracker.load_state(filepath)

    assert new_tracker.get_submission_for_miner_hotkey("hotkey1") == tracker.get_submission_for_miner_hotkey("hotkey1")
    assert len(new_tracker.get_eval_results_for_miner_hotkey("hotkey1", 0)) == 1
    assert new_tracker.get_eval_results_for_miner_hotkey("hotkey1", 0)[0] == eval_result
    assert new_tracker.get_training_result_for_miner_hotkey("hotkey1") == train_result
    # Check that other hotkey is also loaded
    assert new_tracker.get_submission_for_miner_hotkey("hotkey2") == tracker.get_submission_for_miner_hotkey("hotkey2")
    # Check that a non-existent hotkey is still not there
    assert new_tracker.get_submission_for_miner_hotkey("hotkey3") is None


def test_record_training_result(tracker):
    """Ensure training results are stored and retrievable."""
    record = TrainingResultRecord(
        competition_id=1,
        block=100,
        train_metrics={"loss": 1.0},
        val_metrics={"val_loss": 1.5},
        num_steps=5,
        device="cuda",
    )
    tracker.record_training_result("hotkey1", record)
    assert tracker.get_training_result_for_miner_hotkey("hotkey1") == record
