# taoverse/model/storage/disk/utils.py

import os
import json
from typing import Type, Any, Tuple
import re
from epochor.model.model_data import ModelId


def save_config_to_disk(config: Any, path: str) -> None:
    """
    Write a config object to a JSON file on disk.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cfg_dict = config.to_dict() if hasattr(config, "to_dict") else config.__dict__
    with open(path, "w") as f:
        json.dump(cfg_dict, f, indent=2)


def load_config_from_disk(config_cls: Type, path: str) -> Any:
    """
    Load a config JSON file and instantiate with config_cls.
    """
    with open(path, "r") as f:
        data = json.load(f)
    return config_cls.from_dict(data) if hasattr(config_cls, "from_dict") else config_cls(**data)


def validate_hf_repo_id(repo_id: str) -> Tuple[str, str]:
    """Ensures a Hugging Face repo ID is valid and returns it as a (namespace, name) tuple."""
    if not isinstance(repo_id, str) or not re.match(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$", repo_id):
        raise ValueError("repo_id must be a string in the format 'namespace/name'")
    return tuple(repo_id.split("/"))


def get_hf_url(model_metadata: "ModelMetadata") -> str:
    """Returns a URL to the HuggingFace repo of the Miner with the given UID."""
    return f"https://huggingface.co/{model_metadata.id.namespace}/{model_metadata.id.name}/tree/{model_metadata.id.commit}"


def get_hf_repo_name(model_metadata: "ModelMetadata") -> str:
    """Returns the HuggingFace repo name of the Miner with the given UID."""
    return f"{model_metadata.id.namespace}/{model_metadata.id.name}"
