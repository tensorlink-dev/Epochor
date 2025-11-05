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
from typing import Optional

import bittensor as bt
from dotenv import load_dotenv
import httpx

from epochor.utils import logging
from epochor.utils import metagraph_utils
from competitions import competitions

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
    parser.add_argument(
        "--platform_api_url",
        type=str,
        default=os.environ.get("EPOCHOR_API_URL", ""),
        help="Optional platform API endpoint for submissions and heartbeats.",
    )
    parser.add_argument(
        "--platform_api_token",
        type=str,
        default=os.environ.get("EPOCHOR_API_TOKEN", ""),
        help="Bearer token for the platform API.",
    )
    parser.add_argument(
        "--model_code_url",
        type=str,
        default=os.environ.get("EPOCHOR_MODEL_CODE_URL", ""),
        help="Model submission code URL for automatic registration.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=os.environ.get("EPOCHOR_MODEL_ID", ""),
        help="Stable identifier for the miner submission when auto-submitting.",
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
    logging.reinitialize_logging()

    logging.info(
        f"Running miner for subnet: {config.netuid} on network: {config.subtensor.chain_endpoint} with config: {config}"
    )

    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(config.netuid)
    axon = bt.axon(wallet=wallet, config=config)

    # This miner does not train locally. Validators will execute the submitted
    # miner_submission module under their own training loop. The axon only needs
    # to stay responsive so the hotkey remains registered on-chain.
    logging.info(
        "This miner does not train locally. Provide a miner_submission module for validators to execute."
    )

    # Attach a dummy forward function to the axon.
    # This is necessary to keep the axon alive.
    def dummy_forward(synapse: bt.Synapse) -> bt.Synapse:
        # This function does not need to do anything.
        # Validators do not query the axon for parameters; they only use the
        # miner_submission artefacts they download from the remote store.
        return synapse

    axon.attach(forward_fn=dummy_forward)

    # Serve the axon to the network.
    try:
        axon.serve(netuid=config.netuid, subtensor=subtensor)
        logging.info(
            f"Serving axon on network: {config.subtensor.chain_endpoint} with netuid: {config.netuid}"
        )
    except Exception as e:
        logging.error(f"Failed to serve axon: {e}")
        pass

    # Start the axon in the background.
    try:
        axon.start()
        logging.info(f"Axon started on port: {config.axon.port}")
    except Exception as e:
        logging.error(f"Failed to start axon: {e}")
        pass

    if not config.offline:
        metagraph_utils.assert_registered(wallet, metagraph)

    _maybe_submit_to_platform(
        api_url=config.platform_api_url,
        api_token=config.platform_api_token,
        hotkey=wallet.hotkey.ss58_address,
        model_code_url=config.model_code_url,
        model_id=config.model_id or None,
    )

    # Keep the miner alive indefinitely.
    logging.info("Miner running...")
    try:
        while True:
            time.sleep(60)
            if not config.offline:
                # Periodically check if the miner is registered.
                try:
                    my_uid = metagraph_utils.assert_registered(wallet, metagraph)
                    logging.trace(f"Miner is registered with UID {my_uid}.")
                except Exception as e:
                    logging.warning(f"Could not check registration status: {e}")

                # Sync metagraph
                try:
                    metagraph.sync(subtensor=subtensor)
                    logging.trace("Metagraph synced.")
                except Exception as e:
                    logging.warning(f"Could not sync metagraph: {e}")

    except KeyboardInterrupt:
        axon.stop()
        logging.success("Miner stopped by user.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")


def _maybe_submit_to_platform(
    api_url: str,
    api_token: str,
    hotkey: str,
    model_code_url: Optional[str],
    model_id: Optional[str],
) -> None:
    api_url = (api_url or "").strip().rstrip("/")
    if not api_url or not model_code_url:
        return
    payload = {"hotkey": hotkey, "model_code_url": model_code_url}
    if model_id:
        payload["model_id"] = model_id
    headers = {}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    try:
        response = httpx.post(f"{api_url}/miner/submit", json=payload, timeout=10.0, headers=headers)
        response.raise_for_status()
        logging.info(
            "Submitted miner to platform API",
            extra={"submission": response.json()},
        )
    except Exception as exc:
        logging.warning(f"Failed to submit miner payload to API: {exc}")


if __name__ == "__main__":
    # Parse and print configuration
    config = get_config()

    if config.list_competitions:
        print(competitions.COMPETITION_SCHEDULE_BY_BLOCK)
    else:
        print(config)
        asyncio.run(main(config))
