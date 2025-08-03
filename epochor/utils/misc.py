import concurrent
import dataclasses
import functools
import hashlib
import multiprocessing
import os
import random
import sys
from datetime import datetime, timedelta
from typing import Any, Optional, Sequence
import logging as std_logging # Use the standard logging library
import queue # stdlib queue.Empty

import bittensor as bt
import asyncio

def _wrapped_func(func: functools.partial, queue: multiprocessing.Queue):
    """
    Wrapper function to run in a subprocess. It sets up a dedicated logger,
    puts the result or exception on the queue, and then exits.
    """
    # Use process ID to create a unique log file name.
    log_file_path = f"/tmp/subprocess_{os.getpid()}.log"
    handler = std_logging.FileHandler(log_file_path, mode='w')
    formatter = std_logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    root_logger = std_logging.getLogger()
    # Remove any handlers inherited from the parent
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)
    root_logger.addHandler(handler)
    root_logger.setLevel(std_logging.INFO)

    # Redirect stdout and stderr to the log file to capture all output.
    sys.stdout = open(log_file_path, 'a', buffering=1)
    sys.stderr = open(log_file_path, 'a', buffering=1)

    try:
        root_logger.info(f"Subprocess started for {func.func.__name__}.")
        result = func()
        root_logger.info(f"Subprocess for {func.func.__name__} finished successfully.")
        queue.put(result)
    except Exception as e:
        root_logger.error(f"Exception caught in subprocess for {func.func.__name__}: {e}", exc_info=True)
        queue.put(e)
    except BaseException as e:
        root_logger.critical(f"BaseException caught in subprocess for {func.func.__name__}: {e}", exc_info=True)
        queue.put(e)
    finally:
        root_logger.info(f"Subprocess for {func.func.__name__} exiting.")
        # return from _wrapped_func, let the Process terminate normally


def run_in_subprocess(func: functools.partial, ttl: int, mode="fork") -> Any:
    ctx   = multiprocessing.get_context(mode)
    queue = ctx.Queue()
    proc  = ctx.Process(target=_wrapped_func, args=[func, queue])
    proc.start()
    pid = proc.pid
    log_file = f"/tmp/subprocess_{pid}.log"

    try:
        proc.join(timeout=ttl)
        if proc.is_alive():
            proc.terminate(); proc.join()
            raise TimeoutError(f"Timeout ({ttl}s) running {func.func.__name__}")

        # Now child has exited (exitcode may be zero)
        try:
            result = queue.get(timeout=5)
        except queue.Empty:
            # nothing came back
            if os.path.exists(log_file):
                with open(log_file) as f:
                    logs = f.read()
            else:
                logs = "<no log file found>"
            raise RuntimeError(
                f"No result from subprocess {func.func.__name__} (exitcode={proc.exitcode}).\n"
                f"Subprocess logs:\n{logs}"
            )

        # If the child explicitly sent an Exception, re-raise it here
        if isinstance(result, Exception):
            raise result

        return result

    finally:
        if os.path.exists(log_file):
            bt.logging.info(f"See subprocess log: {log_file}")


async def run_in_thread(func: functools.partial, ttl: int, name=None) -> Any:
    """
    Runs the provided function on a thread with 'ttl' seconds to complete.

    Args:
        func (functools.partial): Function to be run.
        ttl (int): How long to try for in seconds.

    Returns:
        Any: The value returned by 'func'
    """
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(
            None, func
        )
    except concurrent.futures.TimeoutError as e:
        bt.logging.error(f"Failed to complete '{name}' within {ttl} seconds.")
        raise TimeoutError(f"Failed to complete '{name}' within {ttl} seconds.") from e


def get_version(filepath: str) -> Optional[int]:
    """Loads a version from the provided filepath or None if the file does not exist.

    Args:
        filepath (str): Path to the version file."""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            line = f.readline()
            if line:
                return int(line)
            return None
    return None


def save_version(filepath: str, version: int):
    """Saves a version to the provided filepath."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(str(version))


def random_date(start: datetime, end: datetime, seed: int = None) -> datetime:
    """
    Return a random datetime between two datetimes.

    Args:
        start (datetime): Start of the range, inclusive.
        end (datetime): End of the range, inclusive.
        seed (int): Optional Seed for the random number generator.
    """

    if start.tzinfo != end.tzinfo:
        raise ValueError("Start and end must have the same timezone.")

    if start >= end:
        raise ValueError("Start must be before end.")

    if seed:
        random.seed(seed)

    # Choose a random point between the 2 datetimes.
    random_seconds = random.randint(0, int((end - start).total_seconds()))

    # Add the random seconds to the start time
    return start + timedelta(seconds=random_seconds)


def fingerprint(any: "Sequence[DataclassInstance] | DataclassInstance") -> int:
    """Returns a fingerprint for a Dataclass or sequence of Dataclasses."""

    # Convert the dataclass to a string representation of the values
    if isinstance(any, Sequence):
        data_string = str([dataclasses.asdict(x) for x in any]).encode("utf-8")
    else:
        data_string = str(dataclasses.asdict(any)).encode("utf-8")
    return hashlib.sha256(data_string).hexdigest()
