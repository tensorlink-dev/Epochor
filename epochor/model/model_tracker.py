from typing import Dict, Optional
import pickle
import threading

from epochor.model.data import ModelMetadata

class ModelTracker:
    """Tracks the current model metadata for each miner."""

    def __init__(self):
        self.miner_hotkey_to_model_metadata: Dict[str, ModelMetadata] = {}
        self.lock = threading.RLock()

    def on_hotkeys_updated(self, hotkeys: set[str]):
        """Called when the hotkeys in the metagraph change."""
        with self.lock:
            # Remove any hotkeys that are no longer in the metagraph.
            for hotkey in list(self.miner_hotkey_to_model_metadata.keys()):
                if hotkey not in hotkeys:
                    del self.miner_hotkey_to_model_metadata[hotkey]

    def on_model_updated(self, hotkey: str, metadata: ModelMetadata):
        """Called when a new model is downloaded for a specific miner."""
        with self.lock:
            self.miner_hotkey_to_model_metadata[hotkey] = metadata

    def get_model_metadata_for_miner_hotkey(self, hotkey: str) -> Optional[ModelMetadata]:
        """Returns the model metadata for a specific miner, or None if not found."""
        with self.lock:
            return self.miner_hotkey_to_model_metadata.get(hotkey)

    def get_miner_hotkey_to_model_metadata_dict(self) -> Dict[str, ModelMetadata]:
        """Returns a copy of the mapping from hotkey to model metadata."""
        with self.lock:
            return self.miner_hotkey_to_model_metadata.copy()

    def save_state(self, filepath: str):
        """Saves the current state of the model tracker to a file."""
        with self.lock:
            with open(filepath, "wb") as f:
                pickle.dump(self.miner_hotkey_to_model_metadata, f)

    def load_state(self, filepath: str):
        """Loads the state of the model tracker from a file."""
        with open(filepath, "rb") as f:
            self.miner_hotkey_to_model_metadata = pickle.load(f)
