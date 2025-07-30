# epochor/validator/model_updater.py

import os
from typing import List, Optional, Tuple

import logging
from epochor.utils import competition_utils
from epochor.model.model_constraints import Competition, ModelConstraints, MODEL_CONSTRAINTS_BY_COMPETITION_ID
from epochor.model.model_data import Model, ModelId, ModelMetadata
from epochor.model.model_tracker import ModelTracker
from epochor.model.base_disk_model_store import LocalModelStore
from epochor.model.base_hf_model_store import RemoteModelStore
from epochor.model.base_metadata_model_store import ModelMetadataStore
from epochor.utils.hashing import get_hash_of_two_strings


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
    def verify_model_satisfies_constraints(
        model: Model, constraints: ModelConstraints
    ) -> bool:
        if not constraints:
            logging.debug(f"No competition constraints for {model.id.competition_id}")
            return False

        # 1) Parameter count
        total_params = sum(p.numel() for p in model.pt_model.parameters())
        if not (total_params <= constraints.max_params):
            logging.debug(f"{model.id.name} parameter count {total_params} outside of [{constraints.max_params}]")
            return False

        # 2) Allowed architectures
        if type(model.model) not in constraints.model_cls:
            logging.debug(f"{type(model.model)} not in allowed model classes")
            return False

        # 3) Optional norm checks
        #  norm_cfg = constraints.norm_validation
        # if norm_cfg is not None:
        #    return ModelUpdater._validate_layer_norms(
        #        model.pt_model,
        #         eps_soft=norm_cfg.eps_soft,
        #         soft_pct=norm_cfg.soft_pct,
        #        eps_hard=norm_cfg.eps_hard
        #     )

        return True

    async def _get_metadata(self, uid: int, hotkey: str) -> Optional[ModelMetadata]:
        # Tries to get the metadata from the tracker first, then from the store.
        tracked_metadata = self.model_tracker.get_model_metadata_for_miner_hotkey(hotkey)
        if tracked_metadata:
            return tracked_metadata
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
        tracked = self.model_tracker.get_model_metadata_for_miner_hotkey(hotkey)
        if not force and tracked == metadata:
            return False

        # 5) Download model
        local_path = self.local_store.get_path(hotkey)
        try:
            model: Model = await self.remote_store.download_model(metadata.id, local_path, comp_now.constraints)
        except ValueError as e:
            raise MinerMisconfiguredError(hotkey, f"Failed to download model: {e}") from e

        # 6) Record in tracker (even if validation fails)
        self.model_tracker.on_model_updated(hotkey, metadata)

        # 7) Optional hash check
        if metadata.id.hash:
            combined = get_hash_of_two_strings(metadata.id.hash, hotkey)
            if combined != metadata.id.secure_hash:
                raise MinerMisconfiguredError(hotkey, "Hash mismatch")

        # 8) Validate constraints
        if not ModelUpdater.verify_model_satisfies_constraints(model, comp_now.constraints):
            raise MinerMisconfiguredError(
                hotkey,
                f"Model fails parameter/architecture constraints for competition {comp_now.id}"
            )

        # 9) Store the model locally.
        try:
            self.local_store.store_model(hotkey, model)
        except ValueError as e:
            raise MinerMisconfiguredError(hotkey, f"Failed to store model: {e}") from e


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
