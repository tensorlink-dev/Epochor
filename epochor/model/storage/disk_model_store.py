import os
import shutil
import traceback
from pathlib import Path
from typing import Dict, Optional

from temporal.utils.hf_accessors import save_hf, load_hf
import epochor.utils.logging as logging
from epochor.model.data import Model, ModelId
from epochor.model.competition.data import ModelConstraints
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
        save_directory = utils.get_local_model_snapshot_dir(self.base_dir, hotkey, model.id)
        os.makedirs(save_directory, exist_ok=True)

        save_hf(
            model=model.model,
            config=model.model.config,
            save_directory=save_directory,
            safe=self.safe_format == "safetensors",
        )
        
        # We compute the hash of the directory to store in the model id.
        model_hash = hash_directory(save_directory)

        # For local storage, the commit is the hash.
        commit = model_hash

        # Create a symlink to the "latest" version of this model.
        latest_path = utils.get_local_model_dir(self.base_dir, hotkey, model.id)
        os.makedirs(os.path.dirname(latest_path), exist_ok=True)
        # Create a snapshot directory based on the hash.
        snapshot_dir = os.path.join(latest_path, commit)
        # If the snapshot dir exists, remove it.
        if os.path.exists(snapshot_dir):
            shutil.rmtree(snapshot_dir)

        shutil.copytree(save_directory, snapshot_dir)

        # Create a symlink from "latest" to the snapshot directory.
        latest_symlink = os.path.join(latest_path, "latest")
        if os.path.exists(latest_symlink) or os.path.islink(latest_symlink):
            os.remove(latest_symlink)
        os.symlink(snapshot_dir, latest_symlink)


        return ModelId(
            namespace=model.id.namespace,
            name=model.id.name,
            commit=commit,
            hash=model_hash
        )

    def retrieve_model(
        self,
        hotkey: str,
        model_id: ModelId,
        model_constraints: Optional[ModelConstraints] = None
    ) -> Model:
        """Retrieves a trained model locally via `load_hf`."""
        model_dir = utils.get_local_model_snapshot_dir(self.base_dir, hotkey, model_id)

        # Verify the hash of the directory before loading.
        model_hash = hash_directory(model_dir)
        if model_hash != model_id.hash:
            raise ValueError(f"Hash mismatch for {model_id}. Expected {model_id.hash}, but on-disk content has hash {model_hash}.")

        pt_model = load_hf(
            model_name_or_path=model_dir,
            model_cls=model_constraints.model_cls,
            config_cls=model_constraints.config_cls,
            safe=self.safe_format == "safetensors",
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
            for hk, mid in valid_models_by_hotkey.items()
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
