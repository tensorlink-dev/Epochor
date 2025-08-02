import concurrent
import dataclasses
import functools
import hashlib
import multiprocessing
import os
import random
from datetime import datetime, timedelta
from typing import Any, Optional, Sequence

import bittensor as bt
import asyncio
import epochor.utils.logging as logging


def _wrapped_func(func: functools.partial, queue: multiprocessing.Queue):
    # Drastically simplify logging in the subprocess.
    # Remove all existing handlers from the root logger.
    import logging as std_logging
    root_logger = std_logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass # Ignore errors if handler is already closed or has issues

    # Optionally, re-add a very basic StreamHandler for subprocess-specific errors
    # This logger will only print to stderr and won't interfere with parent's queues.
    subprocess_logger = std_logging.getLogger("subprocess_logger")
    if not subprocess_logger.handlers:
        handler = std_logging.StreamHandler()
        formatter = std_logging.Formatter('%(asctime)s | %(levelname)s | Subprocess | %(message)s')
        handler.setFormatter(formatter)
        subprocess_logger.addHandler(handler)
    subprocess_logger.setLevel(std_logging.ERROR) # Only log errors in subprocess
    subprocess_logger.propagate = False # Ensure it doesn't try to send to parent

    try:
        result = func()
        queue.put(result)
    except Exception as e:
        # Log the exception in the subprocess itself using its isolated logger
        subprocess_logger.error(f"Exception in subprocess: {e}", exc_info=True)
        # Put the exception on the queue so the parent can re-raise it
        queue.put(e)
    except BaseException as e: # Catch BaseException for critical errors like KeyboardInterrupt
        subprocess_logger.critical(f"BaseException in subprocess: {e}", exc_info=True)
        queue.put(e)


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

    process.join(timeout=ttl)

    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError(f"Failed to {func.func.__name__} after {ttl} seconds")

    try:
        # Use a timeout for queue.get as well, to prevent hanging if the child put nothing
        result = queue.get(timeout=10) # Small timeout to avoid infinite block
    except multiprocessing.queues.Empty:
        # This should ideally not happen with the robust _wrapped_func.
        bt.logging.error(f"Subprocess for {func.func.__name__} terminated without putting result on queue.")
        raise RuntimeError(f"Subprocess for {func.func.__name__} failed to return a result.")
    except Exception as e:
        bt.logging.error(f"Error getting result from subprocess queue for {func.func.__name__}: {e}", exc_info=True)
        raise

    # If we put an exception on the queue then re-raise instead of returning.
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
        logging.error(f"Failed to complete '{name}' within {ttl} seconds.")
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


def configure_logging(config: bt.config) -> None:
    """Configures the Taoverse logger from a bittensor config."""

    logging_config = getattr(config, "logging", None)
    if logging_config and logging_config.trace:
        logging.set_verbosity_trace()
    elif logging_config and logging_config.debug:
        logging.set_verbosity_debug()
    else:
        logging.set_verbosity_info()
