# epochor/storage/hf_model_store.py

import os
import tempfile
import logging
from dataclasses import replace
from typing import Optional

import bittensor as bt
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError

from epochor.model.base_hf_model_store import RemoteModelStore
from epochor.model.model_data import Model, ModelId
from epochor.model.model_constraints import ModelConstraints
from epochor.model.model_updater import MinerMisconfiguredError
from epochor.model.model_utils import save_hf, load_hf
from epochor.utils.hashing import hash_directory


logger = logging.getLogger(__name__)


class 


HuggingFaceModelStore(RemoteModelStore):
    """Epochor’s HF‐Hub implementation for storing & retrieving TS models."""

    @classmethod
    def _ensure_token(cls) -> str:
        token = os.getenv("HF_ACCESS_TOKEN") or os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("No Hugging Face token in HF_ACCESS_TOKEN or HF_TOKEN")
        return token

    async def upload_model(
        self,
        model: Model,
        model_constraints: ModelConstraints,
    ) -> ModelId:
        """
        Saves a model to a local directory, computes its hash, and pushes it to the Hugging Face Hub.

        This method first saves the model and its configuration to a temporary local directory.
        It then computes a secure hash of the directory's contents to ensure integrity.
        Finally, it uploads the directory to the specified Hugging Face repository and retrieves the
        commit hash from the Hub.

        Args:
            model (Model): The model to be uploaded, containing the PyTorch model and configuration.
            model_constraints (ModelConstraints): The constraints to apply to the model.

        Returns:
            ModelId: A new ModelId containing the namespace, name, commit hash from Hugging Face,
                    and the locally computed secure hash.
        """
        token = self._ensure_token()
        repo_id = f"{model.id.namespace}/{model.id.name}"
        api = HfApi(token=token)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save the model and config to a temporary directory.
            # This prepares the content for both hashing and uploading.
            save_hf(
                model=model.model,
                config=model.model.config,
                save_directory=tmpdir,
                safe=True,  # Always use safetensors for safety and consistency.
            )

            # Compute the secure hash from the contents of the local directory before uploading.
            secure_hash = hash_directory(tmpdir)

            # Now, upload the contents of the temporary directory to the Hub.
            try:
                 api.create_repo(repo_id, private=True, exist_ok=True)
                 commit_info = api.upload_folder(
                     repo_id=repo_id,
                     folder_path=tmpdir,
                     commit_message=f"Epochor upload for {model.id.name}",
                 )
                 commit_hash = commit_info.oid
            except Exception as e:
                raise IOError(f"Failed to upload model to Hugging Face: {e}") from e


        # Return a new ModelId with the retrieved commit and the computed hash.
        return ModelId(
            namespace=model.id.namespace,
            name=model.id.name,
            commit=commit_hash,
            hash=secure_hash,
            competition_id=model.id.competition_id
        )

    async def download_model(
        self,
        model_id: ModelId,
        local_path: str,
        model_constraints: ModelConstraints,
    ) -> Model:
        """
        Downloads a model from the Hugging Face Hub, verifies its integrity, and loads it into memory.

        This method first checks the model repository size against constraints. It then downloads the
        model files for a specific commit into a temporary directory. The hash of the downloaded
        content is verified against the hash in the provided ModelId. If the hashes match, the model
        is loaded into memory.

        Args:
            model_id (ModelId): The identifier of the model to be downloaded, including the commit and hash.
            local_path (str): A local path (currently unused, but kept for future cache implementations).
            model_constraints (ModelConstraints): The constraints to apply during model loading.

        Returns:
            Model: The downloaded and verified model, including the PyTorch model and its configuration.
        """
        if not model_id.commit or not model_id.hash:
            raise MinerMisconfiguredError(
                model_id.name, "Missing Hugging Face commit or hash in ModelId."
            )

        repo_id = f"{model_id.namespace}/{model_id.name}"
        token = self._ensure_token()
        api = HfApi(token=token)

        # 1. Check repository size against constraints before downloading.
        try:
            model_info = api.repo_info(
                repo_id=repo_id,
                revision=model_id.commit,
                files_metadata=True,
                token=token,
            )
            total_size = sum(
                f.size for f in model_info.siblings if f.size is not None
            )
            if (
                model_constraints.max_model_size_bytes is not None
                and total_size > model_constraints.max_model_size_bytes
            ):
                raise MinerMisconfiguredError(
                    hotkey=model_id.name,
                    message=f"Repository size {total_size} bytes exceeds max allowed {model_constraints.max_model_size_bytes} bytes.",
                )
        except RepositoryNotFoundError:
            raise MinerMisconfiguredError(
                hotkey=model_id.name,
                message=f"Hugging Face repository '{repo_id}' at commit '{model_id.commit}' not found.",
            )

        # 2. Download files to a temporary directory for verification.
        with tempfile.TemporaryDirectory() as tmp_download_dir:
            try:
                download_path = snapshot_download(
                    repo_id=repo_id,
                    revision=model_id.commit,
                    cache_dir=tmp_download_dir,
                    token=token,
                )
            except Exception as e:
                raise MinerMisconfiguredError(
                    hotkey=model_id.name,
                    message=f"Failed to download '{repo_id}' from Hugging Face: {e}",
                ) from e

            # 3. Verify the hash of the downloaded content.
            computed_hash = hash_directory(download_path)
            if computed_hash != model_id.hash:
                raise ValueError(
                    f"Hash mismatch for {repo_id}. Expected {model_id.hash}, but downloaded content has hash {computed_hash}."
                )

            # 4. If the hash is valid, load the model from the verified local path.
            try:
                pt_model = load_hf(
                    model_name_or_path=download_path,  # Load from the verified temporary path.
                    model_cls=model_constraints.model_cls,
                    config_cls=model_constraints.config_cls,
                    safe=True,
                    map_location="cpu",
                )
            except Exception as e:
                raise MinerMisconfiguredError(
                    hotkey=model_id.name,
                    message=f"Failed to load verified model '{repo_id}': {e}",
                ) from e

        # 5. Return the loaded model with its original, verified ModelId.
        return Model(id=model_id, model=pt_model)
