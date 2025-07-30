
"""A script that registers a model from Hugging Face to the subnet for evaluation.

Usage:
    python scripts/register_hf_model.py --hf_repo_id <your-hf-repo-id> --competition_id <competition_id> --wallet.name <coldkey> --wallet.hotkey <hotkey>

Prerequisites:
   1. HF_ACCESS_TOKEN is set in the environment or .env file.
   2. The model repository specified by --hf_repo_id exists and is accessible.
   3. Your miner is registered on the subnet.
"""

import asyncio
import os
import argparse
import bittensor as bt
from dotenv import load_dotenv
from huggingface_hub import HfApi
import epochor.mining as mining
import constants
from epochor.utils import metagraph_utils

from epochor.utils import logging
from constants import CompetitionId


from epochor.model.storage.hf_model_store import HuggingFaceModelStore
from epochor.model.storage.metadata_model_store import ChainModelMetadataStore


from enum import IntEnum

# Load environment variables from .env file.
load_dotenv()

# Set TOKENIZERS_PARALLELISM to true to avoid warnings.
os.environ["TOKENIZERS_PARALLELISM"] = "true"



def get_config() -> bt.config:
    """
    Initializes an argument parser and returns the parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hf_repo_id",
        type=str,
        help="The Hugging Face repo id, which should include the org/user and repo name. E.g. my-username/my-model",
    )
    parser.add_argument(
        "--netuid",
        type=int,
        default=constants.SUBNET_UID,
        help="The subnet UID.",
    )
    parser.add_argument(
        "--competition_id",
        type=int,
        choices=[c.value for c in CompetitionId],
        required=True,
        help="The competition to upload the model for.",
    )
    parser.add_argument(
        "--list_competitions",
        action="store_true",
        help="Print out all available competitions and their IDs.",
    )

    # Add Bittensor wallet, subtensor, and logging arguments
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)

    config = bt.config(parser)
    return config


async def main(config: bt.config):
    """
    Main function to handle model upload.
    """
    # Initialize Bittensor objects.
    bt.logging(config=config)
    logging.reinitialize_logging()
    logging.configure_logging(config)

    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(config.netuid)

    # Convert the integer competition_id to the CompetitionId enum.
    config.competition_id = CompetitionId(config.competition_id)
    
    chain_metadata_store = ChainModelMetadataStore(
        subtensor=subtensor,
        subnet_uid=config.netuid,
        wallet=wallet,
    )

    # Ensure the miner is registered and has a Hugging Face token.
    metagraph_utils.assert_registered(wallet, metagraph)
    
    # Get the HF Repo Information
    hf_api = HfApi()
    repo_info = hf_api.repo_info(repo_id=config.hf_repo_id, token=os.environ.get("HF_ACCESS_TOKEN"))
    commit_hash = repo_info.sha

    # Push the model to the subnet.
    await mining.register(
        wallet,
        repo_id=config.hf_repo_id,
        commit=commit_hash,
        competition_id=config.competition_id,
        metadata_store=chain_metadata_store,
        netuid=config.netuid,
        subtensor=subtensor,
    )


if __name__ == "__main__":
    config = get_config()
    if config.list_competitions:
        print(constants.COMPETITION_SCHEDULE_BY_BLOCK)
    else:
        print(config)
        asyncio.run(main(config))
