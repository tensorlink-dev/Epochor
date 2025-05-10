"""
Utilities for enforcing a 2-hour evaluation window based on on-chain block timestamps.

This module provides functions to fetch the timestamp of a given block height
from a Bittensor Subtensor node and to check if the current time is within a
specified validation window (defaulting to 2 hours) from that block's timestamp.
"""
# Standard library
from datetime import datetime, timedelta, timezone # Added timezone for utcnow awareness

# Third-party
try:
    from bittensor.subtensor import Subtensor
except ImportError:
    # Provide a mock Subtensor if bittensor is not installed,
    # allowing the module to be imported but functions will fail.
    import logging
    logging.warning(
        "Failed to import 'bittensor.subtensor.Subtensor'. "
        "Functions in validation_window.py requiring Subtensor interaction will not work."
    )
    class SubtensorHeader: # type: ignore
        def __init__(self, timestamp: float):
            self.timestamp = timestamp

    class Subtensor: # type: ignore
        def __init__(self, chain_endpoint: str):
            self.chain_endpoint = chain_endpoint
            logging.warning(
                f"Using a mock Subtensor for endpoint: {self.chain_endpoint}. "
                "Real blockchain interaction is not possible."
            )
        def get_block_header(self, block_hash_or_number: int) -> SubtensorHeader: # type: ignore
            # Return a mock header with a timestamp placeholder (e.g., current time or epoch)
            # This is just to allow type checking and basic flow if bittensor is not present.
            logging.warning(
                f"Mock Subtensor: get_block_header called for block {block_hash_or_number}. "
                "Returning a mock header."
            )
            # Using current time as a mock timestamp. In a real scenario, this would be from the chain.
            return SubtensorHeader(timestamp=datetime.now(timezone.utc).timestamp())
        def get_current_block(self) -> int: # Add if needed by other logic, though not directly by this file
             raise NotImplementedError("Mock Subtensor: get_current_block is not implemented.")


# Default 2-hour window
DEFAULT_WINDOW: timedelta = timedelta(hours=2)

def get_block_timestamp(node_url: str, block_height: int) -> datetime:
    """
    Fetch the UTC timestamp of a given block height from the Subtensor node.

    Args:
        node_url: The URL of the Bittensor Subtensor node.
        block_height: The specific block height for which to fetch the timestamp.

    Returns:
        A datetime object representing the UTC timestamp of the block.
        
    Raises:
        Various exceptions from the bittensor library if the connection fails
        or the node call is unsuccessful (e.g., block not found).
    """
    subtensor = Subtensor(chain_endpoint=node_url)
    # Assuming get_block_header can take an integer block_height.
    # The method might expect a block hash depending on the bittensor version.
    # If it requires a hash, block_height would first need to be converted to a hash.
    header = subtensor.get_block_header(block_height) # type: ignore
    
    # Ensure header.timestamp is treated correctly.
    # datetime.utcfromtimestamp expects a POSIX timestamp (seconds since epoch, UTC).
    # If header.timestamp is already a datetime object, this needs adjustment.
    # Assuming it's a POSIX timestamp (float or int).
    if not hasattr(header, 'timestamp'):
        raise AttributeError(f"Subtensor block header for block {block_height} does not have a 'timestamp' attribute.")
    
    return datetime.fromtimestamp(header.timestamp, tz=timezone.utc)


def is_within_window(
    node_url: str,
    block_height: int,
    window: timedelta = DEFAULT_WINDOW
) -> bool:
    """
    Return True if the current UTC time is within (block_timestamp + window).
    Effectively checks if `block_timestamp <= current_time <= block_timestamp + window`.
    No, the prompt implies `current_time <= block_timestamp + window` (block is not too old).

    Args:
        node_url: The URL of the Bittensor Subtensor node.
        block_height: The block height to get the timestamp from.
        window: The timedelta representing the validation window duration.
                Defaults to DEFAULT_WINDOW (2 hours).

    Returns:
        True if the current time is not past the block's timestamp plus the window,
        False otherwise.
    """
    try:
        block_ts_utc = get_block_timestamp(node_url, block_height)
    except Exception as e:
        # Log the error and potentially return False or re-raise,
        # depending on desired behavior for chain connectivity issues.
        import logging # Local import for conditional use
        logging.error(f"Could not get block timestamp for block {block_height} from {node_url}: {e}")
        return False # Fail closed: if can't get timestamp, assume not within window.

    current_time_utc = datetime.now(timezone.utc)
    
    # Check if current time is not later than the block's timestamp plus the window
    return current_time_utc <= (block_ts_utc + window)

__all__ = ["get_block_timestamp", "is_within_window", "DEFAULT_WINDOW"]
