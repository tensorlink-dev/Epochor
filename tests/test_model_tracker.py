
import unittest
import tempfile
import os
import shutil
from epochor.model.model_tracker import ModelTracker
from epochor.model.model_data import ModelId, ModelMetadata, EvalResult

class TestModelTracker(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = ModelTracker()
        self.model_id1 = ModelId(namespace="test_ns", name="test_name1", commit="c1", hash="h1", competition_id=1)
        self.metadata1 = ModelMetadata(id=self.model_id1, block=100)
        self.model_id2 = ModelId(namespace="test_ns", name="test_name2", commit="c2", hash="h2", competition_id=1)
        self.metadata2 = ModelMetadata(id=self.model_id2, block=101)
        
        self.tracker.on_model_updated("hotkey1", self.metadata1)
        self.tracker.on_model_updated("hotkey2", self.metadata2)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_on_hotkeys_updated(self):
        self.assertIsNotNone(self.tracker.get_model_metadata_for_miner_hotkey("hotkey1"))
        self.assertIsNotNone(self.tracker.get_model_metadata_for_miner_hotkey("hotkey2"))
        
        self.tracker.on_hotkeys_updated({"hotkey1"})
        
        self.assertIsNotNone(self.tracker.get_model_metadata_for_miner_hotkey("hotkey1"))
        self.assertIsNone(self.tracker.get_model_metadata_for_miner_hotkey("hotkey2"))

    def test_on_model_updated(self):
        metadata = self.tracker.get_model_metadata_for_miner_hotkey("hotkey1")
        self.assertEqual(metadata, self.metadata1)

    def test_on_model_evaluated_and_history(self):
        eval_result1 = EvalResult(block=110, score=0.9, winning_model_block=0, winning_model_score=0)
        self.tracker.on_model_evaluated("hotkey1", 0, eval_result1)
        
        eval_results = self.tracker.get_eval_results_for_miner_hotkey("hotkey1", 0)
        self.assertEqual(len(eval_results), 1)
        self.assertEqual(eval_results[0], eval_result1)
        
        eval_result2 = EvalResult(block=120, score=0.95, winning_model_block=0, winning_model_score=0)
        self.tracker.on_model_evaluated("hotkey1", 0, eval_result2)

        eval_results = self.tracker.get_eval_results_for_miner_hotkey("hotkey1", 0)
        self.assertEqual(len(eval_results), 2)
        self.assertEqual(eval_results[0], eval_result1)
        self.assertEqual(eval_results[1], eval_result2)

    def test_get_block_last_evaluated(self):
        self.assertIsNone(self.tracker.get_block_last_evaluated("hotkey1"))
        
        eval_result1 = EvalResult(block=110, score=0.9, winning_model_block=0, winning_model_score=0)
        self.tracker.on_model_evaluated("hotkey1", 0, eval_result1)
        
        self.assertEqual(self.tracker.get_block_last_evaluated("hotkey1"), 110)

        eval_result2 = EvalResult(block=120, score=0.95, winning_model_block=0, winning_model_score=0)
        self.tracker.on_model_evaluated("hotkey1", 1, eval_result2)

        # The block number should be the max across all competitions
        self.assertEqual(self.tracker.get_block_last_evaluated("hotkey1"), 120)
        
        eval_result3 = EvalResult(block=115, score=0.98, winning_model_block=0, winning_model_score=0)
        self.tracker.on_model_evaluated("hotkey1", 0, eval_result3)

        self.assertEqual(self.tracker.get_block_last_evaluated("hotkey1"), 120)

    def test_save_and_load_state(self):
        filepath = os.path.join(self.temp_dir, "tracker_state.pkl")
        
        # Add some eval results to save
        eval_result = EvalResult(block=110, score=0.9, winning_model_block=0, winning_model_score=0)
        self.tracker.on_model_evaluated("hotkey1", 0, eval_result)
        
        self.tracker.save_state(filepath)
        self.assertTrue(os.path.exists(filepath))
        
        new_tracker = ModelTracker()
        new_tracker.load_state(filepath)
        
        self.assertEqual(new_tracker.get_model_metadata_for_miner_hotkey("hotkey1"), self.metadata1)
        self.assertEqual(len(new_tracker.get_eval_results_for_miner_hotkey("hotkey1", 0)), 1)
        self.assertEqual(new_tracker.get_eval_results_for_miner_hotkey("hotkey1", 0)[0], eval_result)
        # Check that other hotkey is also loaded
        self.assertEqual(new_tracker.get_model_metadata_for_miner_hotkey("hotkey2"), self.metadata2)
        # Check that a non-existent hotkey is still not there
        self.assertIsNone(new_tracker.get_model_metadata_for_miner_hotkey("hotkey3"))


if __name__ == '__main__':
    unittest.main()
