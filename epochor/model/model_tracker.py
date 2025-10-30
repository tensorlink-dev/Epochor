from typing import Dict, Optional, List
import pickle
import threading
import logging

from epochor.model.model_data import ModelMetadata, EvalResult

class ModelTracker:
    """Tracks the current model metadata for each miner."""

    def __init__(self):
        self.miner_hotkey_to_model_metadata: Dict[str, ModelMetadata] = {}
        self.miner_hotkey_to_eval_results: Dict[str, Dict[int, List[EvalResult]]] = {}
        self.lock = threading.RLock()
        #self.hotkey_to_model_fingerprint : Dict[str, Dict] = {}

    def on_hotkeys_updated(self, hotkeys: set[str]):
        """Called when the hotkeys in the metagraph change."""
        with self.lock:
            # Remove any hotkeys that are no longer in the metagraph.
            for hotkey in list(self.miner_hotkey_to_model_metadata.keys()):
                if hotkey not in hotkeys:
                    del self.miner_hotkey_to_model_metadata[hotkey]
            for hotkey in list(self.miner_hotkey_to_eval_results.keys()):
                if hotkey not in hotkeys:
                    del self.miner_hotkey_to_eval_results[hotkey]

    def on_model_updated(
        self,
        hotkey: str,
        model_metadata: ModelMetadata,
    ) -> None:
        """Notifies the tracker that a miner has had their associated model updated.

        Args:
            hotkey (str): The miner's hotkey.
            model_metadata (ModelMetadata): The latest model metadata of the miner.
        """
        with self.lock:
            prev_metadata = self.miner_hotkey_to_model_metadata.get(hotkey, None)
            self.miner_hotkey_to_model_metadata[hotkey] = model_metadata

            # If the model was updated, clear the evaluation results since they're no
            # longer relevant.
            if prev_metadata != model_metadata:
                if hotkey in self.miner_hotkey_to_eval_results:
                    self.miner_hotkey_to_eval_results[hotkey].clear()


    def on_model_downloaded(self, hotkey: str, metadata: ModelMetadata):
        """Called when a new model is downloaded for a specific miner."""
        with self.lock:
            self.miner_hotkey_to_model_metadata[hotkey] = metadata

    def on_model_evaluated(self, hotkey: str, competition_id: int, eval_result: EvalResult):
        """Called when a model is evaluated."""
        with self.lock:
            if hotkey not in self.miner_hotkey_to_eval_results:
                self.miner_hotkey_to_eval_results[hotkey] = {}
            if competition_id not in self.miner_hotkey_to_eval_results[hotkey]:
                self.miner_hotkey_to_eval_results[hotkey][competition_id] = []
            self.miner_hotkey_to_eval_results[hotkey][competition_id].append(eval_result)

    def get_model_metadata_for_miner_hotkey(self, hotkey: str) -> Optional[ModelMetadata]:
        """Returns the model metadata for a specific miner, or None if not found."""
        with self.lock:
            return self.miner_hotkey_to_model_metadata.get(hotkey)

    def get_eval_results_for_miner_hotkey(self, hotkey: str, competition_id: int) -> List[EvalResult]:
        """Returns the evaluation results for a specific miner and competition."""
        with self.lock:
            return self.miner_hotkey_to_eval_results.get(hotkey, {}).get(competition_id, [])

    def get_block_last_evaluated(self, hotkey: str) -> Optional[int]:
        """Returns the block number of the last evaluation for a specific miner."""
        with self.lock:
            if hotkey in self.miner_hotkey_to_eval_results and self.miner_hotkey_to_eval_results[hotkey]:
                # Assuming the last result is the most recent.
                # Find the competition with the most recent eval
                last_eval_block = 0
                for results in self.miner_hotkey_to_eval_results[hotkey].values():
                    if results:
                        last_eval_block = max(last_eval_block, results[-1].block)
                return last_eval_block
            return None

    def get_miner_hotkey_to_model_metadata_dict(self) -> Dict[str, ModelMetadata]:
        """Returns a copy of the mapping from hotkey to model metadata."""
        with self.lock:
            return self.miner_hotkey_to_model_metadata.copy()
   
    def save_state(self, filepath: str):
        """Saves the current state of the model tracker to a file."""
        with self.lock:
            with open(filepath, "wb") as f:
                pickle.dump(self.miner_hotkey_to_model_metadata, f)
                pickle.dump(self.miner_hotkey_to_eval_results, f)

    def load_state(self, filepath: str):
        """Loads the state of the model tracker from a file."""
        with open(filepath, "rb") as f:
            self.miner_hotkey_to_model_metadata = pickle.load(f)
            self.miner_hotkey_to_eval_results = pickle.load(f)

    # get fingerprint, append fingerprint
