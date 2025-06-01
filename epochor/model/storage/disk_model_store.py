import os
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from temporal.utils.hf_accessors import save_hf, load_hf
import epochor.utilities.logging as logging
from epochor.model.data import Model, ModelId
from epochor.model.competition.data import ModelConstraints
from epochor.model.storage.disk import utils
from epochor.model.storage.local_model_store import LocalModelStore


class DiskModelStore(LocalModelStore):
    """Local storage–based implementation for storing and retrieving a model on disk."""

    def __init__(self, base_dir: str, safe_format: str = "safetensors"):
        super().__init__(local_dir=base_dir, safe_format=safe_format)
        os.makedirs(utils.get_local_miners_dir(base_dir), exist_ok=True)
        self.base_dir = base_dir

    def get_path(self, hotkey: str) -> str:
        """Returns the root path for this hotkey’s models."""
        return utils.get_local_miner_dir(self.base_dir, hotkey)

    def store_model(self, hotkey: str, model: Model) -> ModelId:
        """Stores a trained model locally via `save_hf`."""
        repo_id = utils.get_local_model_snapshot_dir(self.base_dir, hotkey, model.id)
        os.makedirs(repo_id, exist_ok=True)

        commit_sha: str = save_hf(
            model=model.pt_model,
            config=model.id.__dict__,
            repo_id=repo_id,
            token=None,
            safe_format=self.safe_format,
            commit_message=f"store {hotkey}/{model.id.name}",
        )
        # We treat the returned SHA as the “commit” on disk
        return ModelId(
            namespace=model.id.namespace,
            name=model.id.name,
            commit=commit_sha,
            hash=None
        )

    def retrieve_model(
        self,
        hotkey: str,
        model_id: ModelId,
        model_constraints: Optional[ModelConstraints] = None
    ) -> Model:
        """Retrieves a trained model locally via `load_hf`."""
        repo_id = utils.get_local_model_snapshot_dir(self.base_dir, hotkey, model_id)

        pt_model, _config_dict = load_hf(
            repo_id=repo_id,
            token=None,
            safe_format=self.safe_format,
        )

        # realize symlinks & compute hash as before
        model_dir = repo_id
        utils.realize_symlinks_in_directory(model_dir)
        model_hash = utils.get_hash_of_directory(model_dir)

        return Model(
            id=ModelId(
                namespace=model_id.namespace,
                name=model_id.name,
                commit=model_id.commit,
                hash=model_hash
            ),
            pt_model=pt_model,
            tokenizer=None,
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
