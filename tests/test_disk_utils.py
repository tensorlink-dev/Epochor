
import unittest
import tempfile
import shutil
import os
import datetime
from pathlib import Path
from epochor.model.data import ModelId
from epochor.model.storage.disk import utils

class TestDiskUtils(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_get_local_miners_dir(self):
        self.assertEqual(utils.get_local_miners_dir("base"), "base/models")

    def test_get_local_miner_dir(self):
        self.assertEqual(utils.get_local_miner_dir("base", "hotkey1"), "base/models/hotkey1")

    def test_get_local_model_dir(self):
        model_id = ModelId(namespace="ns", name="name", commit="commit", hash="hash")
        expected_path = os.path.join("base", "models", "hotkey1", "models--ns--name")
        self.assertEqual(utils.get_local_model_dir("base", "hotkey1", model_id), expected_path)

    def test_get_local_model_snapshot_dir(self):
        model_id = ModelId(namespace="ns", name="name", commit="commit", hash="hash")
        expected_path = os.path.join("base", "models", "hotkey1", "models--ns--name", "snapshots", "commit")
        self.assertEqual(utils.get_local_model_snapshot_dir("base", "hotkey1", model_id), expected_path)

    def test_get_newest_datetime_under_path(self):
        # Create a directory with a few files
        test_path = os.path.join(self.temp_dir, "test_dir")
        os.makedirs(test_path)
        
        path1 = Path(os.path.join(test_path, "file1.txt"))
        path1.touch()
        
        # Make sure the timestamps are different
        os.utime(path1, (datetime.datetime.now() - datetime.timedelta(seconds=10)).timestamp())
        
        path2 = Path(os.path.join(test_path, "file2.txt"))
        path2.touch()

        newest_time = utils.get_newest_datetime_under_path(test_path)
        self.assertAlmostEqual(newest_time.timestamp(), path2.stat().st_mtime, delta=1)

    def test_get_newest_datetime_empty_dir(self):
        test_path = os.path.join(self.temp_dir, "empty_dir")
        os.makedirs(test_path)
        self.assertEqual(utils.get_newest_datetime_under_path(test_path), datetime.datetime.min)
        
    def test_remove_dir_out_of_grace_by_datetime(self):
        test_path = os.path.join(self.temp_dir, "test_dir_grace")
        os.makedirs(test_path)

        # Test removal
        last_modified_old = datetime.datetime.now() - datetime.timedelta(seconds=100)
        self.assertTrue(utils.remove_dir_out_of_grace_by_datetime(test_path, 50, last_modified_old))
        self.assertFalse(os.path.exists(test_path))

        # Recreate for next test
        os.makedirs(test_path)

        # Test no removal
        last_modified_new = datetime.datetime.now() - datetime.timedelta(seconds=20)
        self.assertFalse(utils.remove_dir_out_of_grace_by_datetime(test_path, 50, last_modified_new))
        self.assertTrue(os.path.exists(test_path))

    def test_remove_dir_out_of_grace(self):
        # Create a directory with a file.
        test_path = os.path.join(self.temp_dir, "test_dir_grace_2")
        os.makedirs(test_path)
        p = Path(os.path.join(test_path, "file1.txt"))
        p.touch()

        # Set modification time to be old.
        old_time = (datetime.datetime.now() - datetime.timedelta(seconds=100)).timestamp()
        os.utime(p, (old_time, old_time))
        
        # Test removal
        self.assertTrue(utils.remove_dir_out_of_grace(test_path, 50))
        self.assertFalse(os.path.exists(test_path))

        # Recreate for next test
        os.makedirs(test_path)
        p = Path(os.path.join(test_path, "file1.txt"))
        p.touch()

        # Test no removal
        self.assertFalse(utils.remove_dir_out_of_grace(test_path, 50))
        self.assertTrue(os.path.exists(test_path))

    def test_realize_symlinks_in_directory(self):
        # Setup directory structure
        source_dir = os.path.join(self.temp_dir, "source")
        os.makedirs(source_dir)
        target_dir = os.path.join(self.temp_dir, "target")
        os.makedirs(target_dir)

        # Create a file and a symlink to it
        source_file = os.path.join(source_dir, "file.txt")
        with open(source_file, "w") as f:
            f.write("test")
        
        symlink_path = os.path.join(target_dir, "link.txt")
        os.symlink(source_file, symlink_path)
        
        self.assertTrue(os.path.islink(symlink_path))

        # Realize symlinks
        count = utils.realize_symlinks_in_directory(target_dir)

        self.assertEqual(count, 1)
        self.assertFalse(os.path.islink(symlink_path))
        self.assertTrue(os.path.exists(symlink_path))
        with open(symlink_path, "r") as f:
            self.assertEqual(f.read(), "test")
            
        # The original file that was linked to should have been moved.
        self.assertFalse(os.path.exists(source_file))

if __name__ == '__main__':
    unittest.main()
