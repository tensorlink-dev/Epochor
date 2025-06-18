import os
import hashlib
import base64
import torch

def get_hash_of_file(path: str) -> str:
    blocksize = 64 * 1024
    file_hash = hashlib.sha256()
    with open(path, "rb") as fp:
        while True:
            data = fp.read(blocksize)
            if not data:
                break
            file_hash.update(data)
    return base64.urlsafe_b64encode(file_hash.digest()).decode("utf-8")


def hash_directory(path: str) -> str:
    """Computes the hash of a directory's contents."""
    if isinstance(path, dict):
        # Handle the case where the input is a state_dict
        # Note: This is not a proper hash of the directory, but a hash of the state_dict.
        # This is a temporary solution to the issue of hashing a directory before it's written to disk.
        # A more robust solution would be to save the state_dict to a temporary file and hash that.
        temp_file = "temp_state_dict.pth"
        torch.save(path, temp_file)
        with open(temp_file, 'rb') as f:
            data = f.read()
        os.remove(temp_file)
        return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode("utf-8")

    dir_hash = hashlib.sha256()

    # Recursively walk everything under the directory for files.
    for cur_path, dirnames, filenames in os.walk(path):
        # Ensure we walk future directories in a consistent order.
        dirnames.sort()
        # Ensure we walk files in a consistent order.
        for filename in sorted(filenames):
            path = os.path.join(cur_path, filename)
            file_hash = get_hash_of_file(path)
            dir_hash.update(file_hash.encode())

    return base64.urlsafe_b64encode(dir_hash.digest()).decode("utf-8")

def get_hash_of_two_strings(str1: str, str2: str) -> str:
    """Returns the sha256 hash of two strings."""
    return hashlib.sha256((str1 + str2).encode()).hexdigest()
