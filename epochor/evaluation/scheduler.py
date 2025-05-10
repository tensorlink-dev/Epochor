"""
Scheduler for evaluation rounds based on Bittensor Subtensor block height.

This module provides utilities to determine the current evaluation round identifier,
which is directly derived from the current block height of a specified
Bittensor Subtensor node. This approach ensures that scheduling is synchronized
with the blockchain's progression rather than wall-clock time.
"""
# Standard library imports (none directly needed for new logic, but good practice if expanded)
# import datetime # No longer needed for core logic

# Third-party imports
try:
    from bittensor.subtensor import Subtensor # For interacting with Bittensor network
except ImportError:
    # Provide a mock Subtensor if bittensor is not installed,
    # allowing the module to be imported but functions will fail.
    import logging
    logging.warning(
        "Failed to import 'bittensor.subtensor.Subtensor'. "
        "Scheduler functions requiring Subtensor interaction will not work."
    )
    class Subtensor: # type: ignore
        def __init__(self, chain_endpoint: str):
            self.chain_endpoint = chain_endpoint
            logging.warning(
                f"Using a mock Subtensor for endpoint: {chain_endpoint}. "
                "Real blockchain interaction is not possible."
            )
        def get_current_block(self) -> int:
            raise NotImplementedError(
                "Mock Subtensor: get_current_block is not implemented. "
                "Install 'bittensor' library for actual functionality."
            )

# Local application/library specific imports (none in this file)

def get_block_height(node_url: str) -> int:
    """
    Retrieve the current block height from the Subtensor node at `node_url`.

    Args:
        node_url: The URL of the Bittensor Subtensor node.
                  (e.g., "ws://127.0.0.1:9944" or an external endpoint)

    Returns:
        The current block height as an integer.
        
    Raises:
        Various exceptions from the bittensor library if the connection fails
        or the node call is unsuccessful.
    """
    subtensor = Subtensor(chain_endpoint=node_url)
    return subtensor.get_current_block()

def get_round_seed(node_url: str) -> int:
    """
    Use the current on-chain block height from the Subtensor node as the seed/round ID.

    This function serves as a direct way to get a unique, chain-derived identifier
    for an evaluation round, ensuring that the 'seed' is tied to the
    blockchain's state.

    Args:
        node_url: The URL of the Bittensor Subtensor node.

    Returns:
        The current block height, used as a round seed.
    """
    return get_block_height(node_url)

__all__ = [
    "get_block_height",
    "get_round_seed",
]
