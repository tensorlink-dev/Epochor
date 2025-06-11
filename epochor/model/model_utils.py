# taoverse/model/storage/disk/utils.py

import os
import json
from typing import Type, Any

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
