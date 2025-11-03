"""Helper utilities for interacting with the Hugging Face Hub."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from huggingface_hub import HfApi
from safetensors.torch import save_file

HF_WRITE_TOKEN_ENV = "HF_TOKEN"


def save_as_safetensors(model: torch.nn.Module, destination: os.PathLike[str]) -> Path:
    """Persist a ``torch.nn.Module`` to disk using the safetensors format."""

    dest_path = Path(destination)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = {
        key: tensor.detach().cpu()
        for key, tensor in model.state_dict().items()
    }
    save_file(state_dict, str(dest_path))
    return dest_path


def push_artifacts_to_hf(
    repo_id: str,
    *,
    local_dir: os.PathLike[str],
    token_env: str = HF_WRITE_TOKEN_ENV,
    private: bool = True,
    commit_message: Optional[str] = None,
) -> str:
    """Upload artefacts to a Hugging Face repository and return the commit hash."""

    token = os.getenv(token_env)
    if not token:
        raise RuntimeError(
            f"Environment variable '{token_env}' must contain a Hugging Face token"
        )

    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True)

    commit = api.upload_folder(
        repo_id=repo_id,
        folder_path=str(local_dir),
        commit_message=commit_message or "epochor validator upload",
    )
    if not hasattr(commit, "oid") or not commit.oid:
        raise RuntimeError("Hugging Face upload did not return a commit hash")
    return commit.oid


def write_meta(
    run_meta: Mapping[str, Any],
    *,
    path: os.PathLike[str] | str = "model_meta.json",
) -> Path:
    """Write validator-side metadata describing a sandbox execution."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(run_meta, handle, indent=2)
    return destination


__all__ = [
    "HF_WRITE_TOKEN_ENV",
    "push_artifacts_to_hf",
    "save_as_safetensors",
    "write_meta",
]
