
import os
import json
import pytest
from epochor.model import model_utils

# A simple mock config class for testing
class MockConfig:
    def __init__(self, param1="a", param2=123):
        self.param1 = param1
        self.param2 = param2

    def to_dict(self):
        return {"param1": self.param1, "param2": self.param2}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
    
    def __eq__(self, other):
        return self.param1 == other.param1 and self.param2 == other.param2

def test_save_and_load_config(temp_dir):
    """Tests that a config can be saved to and loaded from disk."""
    config_path = os.path.join(temp_dir, "config.json")
    original_config = MockConfig(param1="test", param2=456)

    # Save the config to disk
    model_utils.save_config_to_disk(original_config, config_path)

    # Check that the file was created and has content
    assert os.path.exists(config_path)
    with open(config_path, 'r') as f:
        content = json.load(f)
        assert content["param1"] == "test"
        assert content["param2"] == 456

    # Load the config from disk
    loaded_config = model_utils.load_config_from_disk(MockConfig, config_path)

    # Verify that the loaded config is identical to the original
    assert original_config == loaded_config

def test_load_nonexistent_config(temp_dir):
    """Tests that loading a non-existent config raises an exception."""
    config_path = os.path.join(temp_dir, "nonexistent_config.json")
    with pytest.raises(FileNotFoundError):
        model_utils.load_config_from_disk(MockConfig, config_path)
