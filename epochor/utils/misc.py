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

import bittensor as bt
import asyncio

def _wrapped_func(func: functools.partial, queue: multiprocessing.Queue):
    """
    Wrapper function to run in a subprocess. It sets up a dedicated logger,
    puts the result or exception on the queue, and then forcefully exits.
    """
    # Set up a dedicated logger for this subprocess to a file.
    # This is crucial because the main process's logger may not be fork-safe.
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
        # Forcefully exit the process to avoid cleanup issues.
        os._exit(0)


def run_in_subprocess(func: functools.partial, ttl: int, mode="fork") -> Any:
    """
    Runs the provided function on a subprocess with 'ttl' seconds to complete.

    Args:
        func (functools.partial): Function to be run.
        ttl (int): How long to try for in seconds.
        mode (str): Mode by which the multiprocessing context is obtained. Default to fork for pickling.

    Returns:
        Any: The value returned by 'func'
    """
    ctx = multiprocessing.get_context(mode)
    queue = ctx.Queue()
    process = ctx.Process(target=_wrapped_func, args=[func, queue])

    process.start()
    pid = process.pid # Get PID for logging purposes.
    process.join(timeout=ttl)

    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError(f"Failed to {func.func.__name__} after {ttl} seconds")

    if process.exitcode != 0 and queue.empty():
        log_file_path = f"/tmp/subprocess_{pid}.log"
        error_message = (
            f"Subprocess for {func.func.__name__} exited unexpectedly with code {process.exitcode}. "
            f"Check the subprocess log file for details: {log_file_path}"
        )
        bt.logging.error(error_message)
        # To make it easier for the user, let's try to read the log and show the tail.
        try:
            with open(log_file_path, "r") as f:
                log_tail = f.readlines()[-20:]
            bt.logging.error("Last 20 lines of subprocess log:")
            for line in log_tail:
                bt.logging.error(line.strip())
        except Exception as e:
            bt.logging.error(f"Could not read subprocess log file: {e}")
            
        raise RuntimeError(error_message)

    try:
        result = queue.get(timeout=10)
    except multiprocessing.queues.Empty:
        bt.logging.error(f"Subprocess for {func.func.__name__} terminated without putting result on queue.")
        raise RuntimeError(f"Subprocess for {func.func.__name__} failed to return a result.")
    except Exception as e:
        bt.logging.error(f"Error getting result from subprocess queue for {func.func.__name__}: {e}", exc_info=True)
        raise

    if isinstance(result, Exception):
        bt.logging.error(f"Exception caught from subprocess for {func.func.__name__}: {result}", exc_info=True)
        raise result
    if isinstance(result, BaseException):
        bt.logging.critical(f"BaseException caught from subprocess for {func.func.__name__}: {result}", exc_info=True)
        raise Exception(f"BaseException raised in subprocess: {str(result)}")

    return result


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
