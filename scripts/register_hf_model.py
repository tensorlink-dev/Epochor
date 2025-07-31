#!/usr/bin/env python3
"""
A script that registers a model from Hugging Face to the subnet for evaluation—
and also computes the same secure hash that our HF-store would generate.

Usage:
    python scripts/register_hf_model.py \
      --hf_repo_id <your-hf-repo-id> \
      --competition_id <competition_id> \
      --wallet.name <coldkey> \
      --wallet.hotkey <hotkey> \
      [--revision <branch-or-tag-or-commit>]

Prerequisites:
   1. HF_ACCESS_TOKEN is set in the environment or .env file.
   2. The model repository specified by --hf_repo_id exists and is accessible.
   3. Your miner is registered on the subnet.
"""

import asyncio
import os
import argparse
import tempfile
import logging

import bittensor as bt
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download

from epochor.utils.hashing import hash_directory
from epochor.utils import metagraph_utils, logging as epo_logging
from epochor.model.storage.metadata_model_store import ChainModelMetadataStore
import epochor.mining as mining
import constants
from competitions.competitions import CompetitionId # Updated import

# Load .env and silence tokenizer warnings
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def get_config() -> bt.config:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hf_repo_id",
        type=str,
        required=True,
        help="The HF repo id (e.g. my-user/my-model).",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Branch, tag or SHA.  Defaults to latest on default branch.",
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
        help="Competition to register under.",
    )
    parser.add_argument(
        "--list_competitions",
        action="store_true",
        help="Print available competitions & exit.",
    )

    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)

    return bt.config(parser)


async def main(config: bt.config):
    # — initialize logging & objects
    bt.logging(config=config)
    epo_logging.reinitialize_logging()
    epo_logging.configure_logging(config)

    wallet    = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(config.netuid)

    metagraph_utils.assert_registered(wallet, metagraph)

    # — resolve commit SHA on HF —
    hf_api = HfApi(token=os.environ.get("HF_ACCESS_TOKEN"))
    repo_info = hf_api.repo_info(
        repo_id=config.hf_repo_id,
        token=os.environ.get("HF_ACCESS_TOKEN"),
        revision=config.revision,
    )
    commit_hash = repo_info.sha
    logging.info(f"Using commit {commit_hash} on {config.hf_repo_id}")

    # — compute secure hash exactly as upload_model would —
    with tempfile.TemporaryDirectory() as tmpdir:
        local_tree = snapshot_download(
            repo_id=config.hf_repo_id,
            revision=commit_hash,
            cache_dir=tmpdir,
            token=os.environ.get("HF_ACCESS_TOKEN"),
        )
        secure_hash = hash_directory(local_tree)
    logging.info(f"Computed secure hash: {secure_hash}")

    # — prepare on-chain metadata store & register —
    chain_store = ChainModelMetadataStore(
        subtensor=subtensor,
        subnet_uid=config.netuid,
        wallet=wallet,
    )

    await mining.register(
        wallet=wallet,
        repo_id=config.hf_repo_id,
        commit_hash=commit_hash,
        # pass the secure hash into the metadata layer:
        metadata_store=chain_store,
        competition_id=CompetitionId(config.competition_id),
        netuid=config.netuid,
        subtensor=subtensor,
        secure_hash=secure_hash,       # ← ensure your register fn accepts this!
    )


if __name__ == "__main__":
    cfg = get_config()
    if cfg.list_competitions:
        # No longer using constants.COMPETITION_SCHEDULE_BY_BLOCK here directly
        # Instead, you might want to fetch it from competitions.competitions if needed
        print("List of competitions is not directly available via constants.COMPETITION_SCHEDULE_BY_BLOCK anymore.")
        print("Please refer to competitions/competitions.py for competition definitions.")
    else:
        print(cfg)
        asyncio.run(main(cfg))
