import os
import shutil
import traceback
from pathlib import Path
from typing import Dict, Optional

from temporal.utils.hf_accessors import save_hf, load_hf
from epochor.utils import logging
from epochor.model.model_data import Model, ModelId
from epochor.model.model_constraints import ModelConstraints
from epochor.model.storage.disk import utils
from epochor.model.base_disk_model_store import LocalModelStore
from epochor.utils.hashing import hash_directory

class DiskModelStore(LocalModelStore):
    """Local storage–based implementation for storing and retrieving a model on disk."""

    def __init__(self, base_dir: str, safe_format: str = "safetensors"):
        super().__init__()
        os.makedirs(utils.get_local_miners_dir(base_dir), exist_ok=True)
        self.base_dir = base_dir
        self.safe_format = safe_format

    def get_path(self, hotkey: str) -> str:
        """Returns the root path for this hotkey’s models."""
        return utils.get_local_miner_dir(self.base_dir, hotkey)

    def store_model(self, hotkey: str, model: Model) -> ModelId:
        """Stores a trained model locally via `save_hf`."""
        # Note: We use the hash of the model as the commit, since we don't have a true "commit" in the local case.

        save_directory = utils.get_local_model_snapshot_dir(self.base_dir, hotkey, model.id)
        os.makedirs(save_directory, exist_ok=True)

        save_hf(
            model=model.model,
            config=model.model.config,
            save_directory=save_directory,
            safe=self.safe_format == "safetensors",
        )
        
        return model_id_with_hash

    def retrieve_model(
        self,
        hotkey: str,
        model_id: ModelId,
        model_constraints: Optional[ModelConstraints] = None
    ) -> Model:
        """Retrieves a trained model locally via `load_hf`."""
        model_dir = utils.get_local_model_snapshot_dir(self.base_dir, hotkey, model_id)

        # Verify the hash of the directory before loading.

        pt_model = load_hf(
            model_name_or_path=model_dir,
            model_cls=model_constraints.model_cls,
            config_cls=model_constraints.config_cls,
            safe=self.safe_format == "safetensors",
            map_location = 'cpu'
        )

        return Model(
            id=model_id,
            model=pt_model,
        )

    def delete_unreferenced_models(
        self, valid_models_by_hotkey: Dict[str, ModelId], grace_period_seconds: int
    ):
        """Check across all of local storage and delete unreferenced models out of grace period."""
        valid_paths = {
            utils.get_local_model_snapshot_dir(self.base_dir, hk, mid)
            for hk, mids in valid_models_by_hotkey.items() for mid in mids
        }

        miners_dir = Path(utils.get_local_miners_dir(self.base_dir))
        for hotkey_dir in miners_dir.iterdir():
            if not hotkey_dir.is_dir():
                continue
            hotkey = hotkey_dir.name

            try:
                if hotkey not in valid_models_by_hotkey:
                    if utils.remove_dir_out_of_grace(str(hotkey_dir), grace_period_seconds):
                        logging.trace(f"Removed unreferenced hotkey directory: {hotkey}")
                    continue

                # clean up stale commits under this hotkey
                for model_ns_name in hotkey_dir.iterdir():
                    if not model_ns_name.is_dir():
                        continue
                    for snapshot in model_ns_name.iterdir():
                        if not snapshot.is_dir():
                            continue
                        for commit_dir in snapshot.iterdir():
                            path = str(commit_dir)
                            if path not in valid_paths:
                                if utils.remove_dir_out_of_grace(path, grace_period_seconds):
                                    logging.trace(f"Removed unreferenced model at: {path}")
            except Exception:
                logging.warning(traceback.format_exc())
