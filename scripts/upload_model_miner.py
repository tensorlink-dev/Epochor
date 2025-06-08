"""A script that pushes a model from disk to the subnet for evaluation.

Usage:
    python scripts/upload_model_miner.py --load_model_dir <path to model> --hf_repo_id my-username/my-project --competition_id <competition_id> --wallet.name <coldkey> --wallet.hotkey <hotkey>

Prerequisites:
   1. HF_ACCESS_TOKEN is set in the environment or .env file.
   2. load_model_dir points to a directory containing a previously trained model, with relevant Hugging Face files (e.g. config.json).
   3. Your miner is registered on the subnet.
"""

import asyncio
import os
import argparse
import bittensor as bt
from dotenv import load_dotenv

import epochor.mining as mining
import constants
from epochor.utils import metagraph_utils
from epochor.model.storage.chain.chain_model_metadata_store import (
    ChainModelMetadataStore,
)
from epochor.model.storage.hugging_face.hugging_face_model_store import (
    HuggingFaceModelStore,
)
from epochor.utils import utils as epochor_utils
from epochor.competition.data import CompetitionId, IntEnumAction


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
        "--load_model_dir",
        type=str,
        default=None,
        help="If provided, loads a previously trained HF model from the specified directory.",
    )
    parser.add_argument(
        "--netuid",
        type=int,
        default=constants.SUBNET_UID,
        help="The subnet UID.",
    )
    parser.add_argument(
        "--competition_id",
        type=CompetitionId,
        action=IntEnumAction,
        required=True,
        help="The competition to upload the model for.",
    )
    parser.add_argument(
        "--list_competitions",
        action="store_true",
        help="Print out all available competitions and their IDs.",
    )
    parser.add_argument(
        "--update_repo_visibility",
        action="store_true",
        help="If true, the repo will be made public after uploading.",
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
    epochor_utils.reinitialize_logging()
    epochor_utils.configure_logging(config)

    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(config.netuid)
    
    chain_metadata_store = ChainModelMetadataStore(
        subtensor=subtensor,
        subnet_uid=config.netuid,
        wallet=wallet,
    )

    # Ensure the miner is registered and has a Hugging Face token.
    metagraph_utils.assert_registered(wallet, metagraph)
    HuggingFaceModelStore.assert_access_token_exists()

    # Load the model from the specified directory.
    model = mining.load_local_model(config.load_model_dir, config.competition_id)

    # Push the model to the subnet.
    await mining.push(
        model,
        config.hf_repo_id,
        wallet,
        competition_id=config.competition_id,
        metadata_store=chain_metadata_store,
        update_repo_visibility=config.update_repo_visibility,
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
