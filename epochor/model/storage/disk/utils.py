import datetime
import os
import shutil

from epochor.model.model_data import ModelId


def get_local_miners_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "models")


def get_local_miner_dir(base_dir: str, hotkey: str) -> str:
    return os.path.join(get_local_miners_dir(base_dir), hotkey)


# Hugging face stores models under models--namespace--name/snapshots/commit when downloading.
def get_local_model_dir(base_dir: str, hotkey: str, model_id: ModelId) -> str:
    return os.path.join(
        get_local_miner_dir(base_dir, hotkey),
        "models" + "--" + model_id.namespace + "--" + model_id.name,
    )


def get_local_model_snapshot_dir(base_dir: str, hotkey: str, model_id: ModelId) -> str:
    # Note: hotkey is not used, but kept for consistency with other methods.
    return os.path.join(
        get_local_model_dir(base_dir, hotkey, model_id),
        "snapshots",
        model_id.commit,
    )


def get_newest_datetime_under_path(path: str) -> datetime.datetime:
    newest_filetime = 0

    # Check to see if any file at any level was modified more recently than the current one.
    for cur_path, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(cur_path, filename)
            try:
                mod_time = os.stat(path).st_mtime
                if mod_time > newest_filetime:
                    newest_filetime = mod_time
            except:
                pass

    if newest_filetime == 0:
        return datetime.datetime.min

    return datetime.datetime.fromtimestamp(newest_filetime)


def remove_dir_out_of_grace_by_datetime(
    path: str, grace_period_seconds: int, last_modified: datetime.datetime
) -> bool:
    """Removes a dir if the last modified time is out of grace period secs. Returns if it was deleted."""
    grace = datetime.timedelta(seconds=grace_period_seconds)

    if last_modified < datetime.datetime.now() - grace:
        shutil.rmtree(path=path, ignore_errors=True)
        return True

    return False


def remove_dir_out_of_grace(path: str, grace_period_seconds: int) -> bool:
    """Removes a dir if the last modified time is out of grace period secs. Returns if it was deleted."""
    last_modified = get_newest_datetime_under_path(path)
    return remove_dir_out_of_grace_by_datetime(
        path, grace_period_seconds, last_modified
    )


def realize_symlinks_in_directory(path: str) -> int:
    """Realizes all symlinks in the given directory, moving the linked file to the location. Returns count removed."""
    realized_symlinks = 0

    for cur_path, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.abspath(os.path.join(cur_path, filename))
            # Get path resolving symlinks if encountered
            real_path = os.path.realpath(path)
            # If different then move
            if path != real_path:
                realized_symlinks += 1
                shutil.move(real_path, path)

    return realized_symlinks
