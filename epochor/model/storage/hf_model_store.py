# epochor/storage/hf_model_store.py

import os
import tempfile
import logging
from dataclasses import replace
from typing import Optional

import bittensor as bt
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

from epochor.storage.remote_model_store import RemoteModelStore
from epochor.model.data import Model, ModelId
from epochor.model.competition.data import ModelConstraints
from epochor.model.model_updater import MinerMisconfiguredError
from epochor.utils.hf_accessors import save_hf, load_hf
from epochor.utils.hashing import hash_directory


logger = logging.getLogger(__name__)


class HuggingFaceModelStore(RemoteModelStore):
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
        model_constraints: ModelConstraints
    ) -> ModelId:
        """
        Save locally + push to HF hub using epochor/utils/hf_accessors.save_hf.
        Returns a new ModelId with the commit hash from HF.
        """
        token = self._ensure_token()
        repo_id = f"{model.id.namespace}/{model.id.name}"

        with tempfile.TemporaryDirectory() as tmp:
            # 1) Save & push to hub
            save_hf(
                model.pt_model,
                config=model.config,
                save_directory=tmp,
                safe=True,
                repo_id=repo_id,
                token=token,
                private=True,
                push_to_hub=True,
                commit_message=f"Epochor upload for {model.id.name}"
            )
            # 2) Load back to compute secure hash
            downloaded, _ = await self._download_and_return_path(
                model.id, tmp, model_constraints, token
            )

        # Return the new ModelId (with updated commit/hash)
        return downloaded.id

    async def download_model(
        self,
        model_id: ModelId,
        local_path: str,
        model_constraints: ModelConstraints
    ) -> Model:
        """
        Downloads from HF hub (or local) via epochor/utils/hf_accessors.load_hf,
        enforces size constraints (using model_constraints.max_bytes),
        and returns a Model with pt_model and updated ModelId (including hash).
        """
        if not model_id.commit:
            raise MinerMisconfiguredError(model_id.name, "Missing HF commit in ModelId")

        repo_id = f"{model_id.namespace}/{model_id.name}"
        token = os.getenv("HF_ACCESS_TOKEN") or os.getenv("HF_TOKEN")

        # 1) Check total repo size at given commit before downloading
        try:
            api = HfApi()
            model_info = api.model_info(
                repo_id=repo_id,
                revision=model_id.commit,
                timeout=10,
                files_metadata=True,
                token=token,
            )
        except RepositoryNotFoundError:
            raise MinerMisconfiguredError(
                hotkey=model_id.name,
                message=f"Hugging Face repository '{repo_id}'@'{model_id.commit}' not found."
            )

        total_size = sum(f.size for f in model_info.siblings)
        if model_constraints.max_bytes is not None and total_size > model_constraints.max_bytes:
            raise MinerMisconfiguredError(
                hotkey=model_id.name,
                message=(
                    f"Repo size {total_size} bytes exceeds max allowed "
                    f"{model_constraints.max_bytes} bytes."
                )
            )

        # 2) Use load_hf to fetch into cache_dir=local_path
        try:
            pt_model = load_hf(
                model_name_or_path=repo_id,
                model_cls=model_constraints.model_cls,
                config_cls=model_constraints.config_cls,
                safe=True,
                map_location="cpu",
                cache_dir=local_path,
                force_download=False,
                token=token,
                **model_constraints.load_kwargs
            )
        except Exception as e:
            raise MinerMisconfiguredError(
                hotkey=model_id.name,
                message=(
                    f"Failed to load '{repo_id}'@'{model_id.commit}' "
                    f"with constraints {model_constraints}. Error: {e}"
                )
            ) from e

        # 3) Compute secure hash of the downloaded directory
        #    The HFHub layout is: local_path/<namespace>/<model_name>/...
        hf_folder = os.path.join(local_path, *repo_id.split("/"))
        if not os.path.isdir(hf_folder):
            raise MinerMisconfiguredError(
                hotkey=model_id.name,
                message=f"Expected directory '{hf_folder}' not found after load."
            )

        secure_hash = hash_directory(hf_folder)
        new_id = replace(model_id, hash=secure_hash)
        return Model(id=new_id, pt_model=pt_model)

    async def _download_and_return_path(
        self,
        model_id: ModelId,
        local_path: str,
        model_constraints: ModelConstraints,
        token: str
    ) -> (Model, str):
        """
        Helper to download and return a Model (with updated commit) and the local path.
        """
        model = await self.download_model(model_id, local_path, model_constraints)
        return model, local_path
