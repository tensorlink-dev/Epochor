# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
import asyncio
import os
import time

import bittensor as bt
from dotenv import load_dotenv

from epochor.utils import logging
from epochor.utils import misc as epochor_utils
from epochor.competitions import competitions

load_dotenv()  # take environment variables from .env.

os.environ["TOKENIZERS_PARALLELISM"] = "true"


# === Config ===
def get_config():
    """
    Set up and parse the command-line arguments to configure the system.
    """

    # Initialize an argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--offline",
        action="store_true",
        help="Does not check if registered.",
    )
    parser.add_argument(
        "--netuid",
        type=int,
        default=31,
        help="The subnet UID.",
    )
    parser.add_argument(
        "--list_competitions", action="store_true", help="Print out all competitions"
    )

    # Include wallet and logging arguments from bittensor
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.axon.add_args(parser)

    # Parse the arguments and create a configuration namespace
    config = bt.config(parser)

    return config


async def main(config: bt.config):
    # Create bittensor objects.
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(
        f"Running miner for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint} with config: {config}"
    )

    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(config.netuid)
    axon = bt.axon(wallet=wallet, config=config)

    # This miner does not train, but instead serves a model that is uploaded externally.
    # It just needs to stay alive on the network.
    bt.logging.info(
        "This miner does not train. Use scripts/upload_model_miner.py to upload a model."
    )

    # Attach a dummy forward function to the axon.
    # This is necessary to keep the axon alive.
    def dummy_forward(synapse: bt.Synapse) -> bt.Synapse:
        # This function does not need to do anything.
        # Validators will fetch the model directly from Hugging Face.
        pass

    axon.attach(forward_fn=dummy_forward)

    # Serve the axon to the network.
    try:
        axon.serve(netuid=config.netuid, subtensor=subtensor)
        bt.logging.info(
            f"Serving axon on network: {config.subtensor.chain_endpoint} with netuid: {config.netuid}"
        )
    except Exception as e:
        bt.logging.error(f"Failed to serve axon: {e}")
        pass

    # Start the axon in the background.
    try:
        axon.start()
        bt.logging.info(f"Axon started on port: {config.axon.port}")
    except Exception as e:
        bt.logging.error(f"Failed to start axon: {e}")
        pass

    if not config.offline:
        epochor_utils.assert_registered(wallet, metagraph)

    # Keep the miner alive indefinitely.
    bt.logging.info("Miner running...")
    try:
        while True:
            time.sleep(60)
            if not config.offline:
                # Periodically check if the miner is registered.
                try:
                    my_uid = epochor_utils.assert_registered(wallet, metagraph)
                    bt.logging.trace(f"Miner is registered with UID {my_uid}.")
                except Exception as e:
                    bt.logging.warning(f"Could not check registration status: {e}")

                # Sync metagraph
                try:
                    metagraph.sync(subtensor=subtensor)
                    bt.logging.trace("Metagraph synced.")
                except Exception as e:
                    bt.logging.warning(f"Could not sync metagraph: {e}")

    except KeyboardInterrupt:
        axon.stop()
        bt.logging.success("Miner stopped by user.")
    except Exception as e:
        bt.logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    # Parse and print configuration
    config = get_config()

    if config.list_competitions:
        print(competitions.COMPETITION_SCHEDULE_BY_BLOCK)
    else:
        print(config)
        asyncio.run(main(config))
