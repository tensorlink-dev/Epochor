# Refactored ChainModelMetadataStore for Epochor

import os
import asyncio
from typing import Optional

import bittensor as bt
import logging

from epochor.model.data import ModelId, ModelMetadata
from epochor.storage.model_metadata_store import ModelMetadataStore
from epochor.utils.run_utils import run_in_thread  # helper to offload sync calls

logger = logging.getLogger(__name__)


class ChainModelMetadataStore(ModelMetadataStore):
    """Epochor chain-backed metadata store for model registration."""

    def __init__(
        self,
        subtensor: bt.subtensor,
        subnet_uid: int,
        wallet: Optional[bt.wallet] = None,
    ):
        self.subtensor = subtensor
        self.subnet_uid = subnet_uid
        self.wallet = wallet

    async def store_model_metadata(
        self,
        hotkey: str,
        model_id: ModelId,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        ttl: int = 60,
    ):
        """Commit compressed ModelId to chain for this miner hotkey."""
        if self.wallet is None:
            raise ValueError("Wallet required to write metadata on-chain")

        data = model_id.to_compressed_str()
        commit_fn = lambda: self.subtensor.commit(
            self.wallet, self.subnet_uid, data
        )
        # Offload blocking call with timeout
        await run_in_thread(commit_fn, timeout=ttl)

    async def retrieve_model_metadata(
        self,
        uid: int,
        hotkey: str,
        ttl: int = 60
    ) -> Optional[ModelMetadata]:
        """Fetch the latest ModelId and block for a given miner hotkey."""
        # 1) get on-chain metadata (block info)
        get_meta_fn = lambda: bt.core.extrinsics.serving.get_metadata(
            self.subtensor, self.subnet_uid, hotkey
        )
        meta = await run_in_thread(get_meta_fn, timeout=ttl)
        if not meta:
            return None

        # 2) get the raw commitment string
        get_commit_fn = lambda: self.subtensor.get_commitment(
            self.subnet_uid, uid
        )
        commit_str = await run_in_thread(get_commit_fn, timeout=ttl)

        # 3) parse ModelId
        try:
            model_id = ModelId.from_compressed_str(commit_str)
        except Exception:
            logger.debug(f"Failed to parse ModelId from chain for {hotkey}")
            return None

        return ModelMetadata(id=model_id, block=meta["block"])


# Helper tests (run as needed)
async def _test_roundtrip():
    # Example usage with env-configured wallet and subtensor
    subtensor = bt.subtensor()
    hotkey = os.getenv("EPOCHOR_HOTKEY")
    wallet_name = os.getenv("EPOCHOR_WALLET")
    coldkey = os.getenv("EPOCHOR_COLDKEY")
    subnet_uid = int(os.getenv("EPOCHOR_SUBNET_UID"))
    uid = int(os.getenv("EPOCHOR_MODEL_UID"))

    wallet = bt.wallet(name=wallet_name, hotkey=hotkey)
    store = ChainModelMetadataStore(subtensor, subnet_uid, wallet)

    model_id = ModelId(namespace="user", name="mymodel", competition_id=1, hash="abc", commit="rev1")
    await store.store_model_metadata(hotkey, model_id)
    fetched = await store.retrieve_model_metadata(uid, hotkey)
    print("Roundtrip OK:", fetched and fetched.id == model_id)

if __name__ == "__main__":
    asyncio.run(_test_roundtrip())


