import os
from pathlib import Path
from typing import List, Optional, Tuple

import logging
from epochor.utils import competition_utils
from epochor.model.model_constraints import Competition
from epochor.model.model_data import Model, ModelMetadata, MinerSubmissionSnapshot
from epochor.model.model_tracker import ModelTracker
from epochor.model.base_disk_model_store import LocalModelStore
from epochor.model.base_hf_model_store import RemoteModelStore
from epochor.model.base_metadata_model_store import ModelMetadataStore


class MinerMisconfiguredError(Exception):
    """Error raised when a miner is misconfigured for Epochor."""

    def __init__(self, hotkey: str, message: str):
        self.hotkey = hotkey
        super().__init__(f"[{hotkey}] {message}")


class ModelUpdater:
    """Checks and syncs each miner’s TS model against on-chain metadata."""

    def __init__(
        self,
        metadata_store: ModelMetadataStore,
        remote_store: RemoteModelStore,
        local_store: LocalModelStore,
        model_tracker: ModelTracker,
    ):
        self.metadata_store = metadata_store
        self.remote_store = remote_store
        self.local_store = local_store
        self.model_tracker = model_tracker

    @staticmethod
    def verify_submission_snapshot(snapshot_path: str) -> bool:
        if not snapshot_path:
            logging.debug("Missing snapshot path for miner submission")
            return False

        expected_file = Path(snapshot_path) / "miner_submission.py"
        if not expected_file.is_file():
            logging.debug("miner_submission.py not found in submission bundle")
            return False

        return True

    async def _get_metadata(self, uid: int, hotkey: str) -> Optional[ModelMetadata]:
        # Prefer the locally cached submission snapshot; fall back to the metadata store.
        submission = self.model_tracker.get_submission_for_miner_hotkey(hotkey)
        if submission is not None:
            return ModelMetadata(id=submission.model_id, block=submission.block)
        return await self.metadata_store.retrieve_model_metadata(uid, hotkey)

    async def sync_model(
        self,
        uid: int,
        hotkey: str,
        curr_block: int,
        schedule: List[Tuple[int, List[Competition]]],
        force: bool = False,
    ) -> bool:
        """
        Download and validate a miner’s model if on-chain metadata changed.

        Returns True if a new model was fetched and passes all checks.
        """
        # 1) Fetch on-chain metadata
        try:
            metadata = await self._get_metadata(uid, hotkey)
            if metadata is None:
                raise MinerMisconfiguredError(hotkey, "No metadata on-chain")
        except Exception as e:
            raise MinerMisconfiguredError(hotkey, f"Failed to get metadata: {e}") from e

        # 2) Find the competition at upload and at current block
        comp_at_upload = competition_utils.get_competition_for_block(
            metadata.id.competition_id, metadata.block, schedule
        )
        comp_now = competition_utils.get_competition_for_block(
            metadata.id.competition_id, curr_block, schedule
        )
        if comp_at_upload is None or comp_now is None:
            raise MinerMisconfiguredError(
                hotkey,
                f"Competition {metadata.id.competition_id} not active at block {metadata.block if comp_at_upload is None else curr_block}"
            )

        # 3) Respect evaluation delay - not sure if I need this? QUERY
        #delay = comp_now.constraints.eval_block_delay
        #if curr_block - metadata.block < delay:
        #    logging.info(f"{hotkey} waiting for eval delay ({delay} blocks)")
        #    return False

        # 4) Skip if metadata unchanged and not forced
        tracked_snapshot = self.model_tracker.get_submission_for_miner_hotkey(hotkey)
        if not force and tracked_snapshot is not None and tracked_snapshot.model_id == metadata.id and tracked_snapshot.block == metadata.block:
            return False

        # 5) Resolve snapshot path and download model + submission artefacts.
        if hasattr(self.local_store, "base_dir"):
            from epochor.model.storage.disk import utils as disk_utils  # local import to avoid cycle

            snapshot_path = disk_utils.get_local_model_snapshot_dir(self.local_store.base_dir, hotkey, metadata.id)
        else:
            base_local_path = self.local_store.get_path(hotkey)
            snapshot_path = os.path.join(base_local_path, metadata.id.commit or "latest")

        try:
            model: Model = await self.remote_store.download_model(metadata.id, snapshot_path, comp_now.constraints)
        except ValueError as e:
            raise MinerMisconfiguredError(hotkey, f"Failed to download model: {e}") from e

        if model.source_path is None:
            model.source_path = snapshot_path

        if not ModelUpdater.verify_submission_snapshot(model.source_path):
            raise MinerMisconfiguredError(
                hotkey,
                "Downloaded submission bundle is missing miner_submission.py",
            )

        submission_snapshot = MinerSubmissionSnapshot(
            model_id=model.id,
            competition_id=metadata.id.competition_id,
            block=metadata.block,
            snapshot_path=model.source_path,
        )

        self.model_tracker.on_submission_updated(hotkey, submission_snapshot)

        return True

    @staticmethod
    def _validate_layer_norms(
        base_model,
        eps_soft: float,
        soft_pct: float,
        eps_hard: float,
    ) -> bool:
        """
        Ensures no projection weight norm exceeds eps_hard, and
        that fewer than `soft_pct` proportion exceed eps_soft.
        """
        exceed = 0
        total = 0

        for layer in getattr(base_model, "layers", []):
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]:
                w = getattr(layer, proj, None)
                if w is None:
                    continue
                norm = w.weight.norm().item()
                total += 1
                if norm > eps_hard:
                    return False
                if norm > eps_soft:
                    exceed += 1

        if total == 0:
            return True
        return (exceed / total) <= soft_pct
