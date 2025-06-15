
import unittest
import tempfile
import shutil
import os
import json
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

class TestModelUtils(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.json")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_save_and_load_config(self):
        # Create a mock config
        original_config = MockConfig(param1="test", param2=456)

        # Save the config to disk
        model_utils.save_config_to_disk(original_config, self.config_path)

        # Check that the file was created and has content
        self.assertTrue(os.path.exists(self.config_path))
        with open(self.config_path, 'r') as f:
            content = json.load(f)
            self.assertEqual(content["param1"], "test")
            self.assertEqual(content["param2"], 456)

        # Load the config from disk
        loaded_config = model_utils.load_config_from_disk(MockConfig, self.config_path)

        # Verify that the loaded config is identical to the original
        self.assertEqual(original_config, loaded_config)

    def test_load_nonexistent_config(self):
        with self.assertRaises(FileNotFoundError):
            model_utils.load_config_from_disk(MockConfig, self.config_path)

if __name__ == '__main__':
    unittest.main()
