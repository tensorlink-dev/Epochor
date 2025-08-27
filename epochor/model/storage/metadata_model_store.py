# Refactored ChainModelMetadataStore for Epochor

import os
import asyncio
from typing import Optional

import bittensor as bt
import logging

from epochor.model.model_data import ModelId, ModelMetadata, MAX_METADATA_BYTES  # ← add MAX_METADATA_BYTES
from epochor.model.base_metadata_model_store import ModelMetadataStore
from epochor.utils.misc import run_in_thread  # helper to offload sync calls

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
        b = data.encode("utf-8")

        # Hard guard: on-chain commitment must be <= MAX_METADATA_BYTES
        if len(b) > MAX_METADATA_BYTES:
            raise ValueError(
                f"Compressed model_id is {len(b)}B > {MAX_METADATA_BYTES}B. "
                "Check ModelId.to_compressed_str() trimming."
            )

        logger.info(
            "Committing model_id (bytes=%d) for hotkey=%s",
            len(b),
            hotkey,
        )

        # Optionally warn if the provided hotkey doesn't match the wallet hotkey
        try:
            if self.wallet and getattr(self.wallet, "hotkey", None):
                w_hotkey = getattr(self.wallet.hotkey, "ss58_address", None)
                if w_hotkey and w_hotkey != hotkey:
                    logger.warning("Hotkey mismatch: arg=%s wallet=%s", hotkey, w_hotkey)
        except Exception:
            pass

        # Thread off the blocking commit; honor wait flags
        def _commit():
            return self.subtensor.commit(
                wallet=self.wallet,
                netuid=self.subnet_uid,
                commitment=data,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

        await run_in_thread(_commit, ttl=ttl)

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
        meta = await run_in_thread(get_meta_fn, ttl=ttl)
        if not meta:
            return None

        # 2) get the raw commitment string
        get_commit_fn = lambda: self.subtensor.get_commitment(
            self.subnet_uid, uid
        )
        commit_str = await run_in_thread(get_commit_fn, ttl=ttl)
        if not commit_str:
            return None

        # 3) parse ModelId
        try:
            model_id = ModelId.from_compressed_str(commit_str)
        except Exception:
            logger.debug("Failed to parse ModelId from chain for %s", hotkey)
            return None

        return ModelMetadata(id=model_id, block=meta["block"])
