import statistics
from typing import List, Optional, Tuple

import taoverse.utilities.logging as logging
from taoverse.model.competition import utils as competition_utils
from taoverse.model.competition.data import Competition, ModelConstraints
from taoverse.model.data import Model, ModelMetadata
from taoverse.model.model_tracker import ModelTracker
from taoverse.model.storage.local_model_store import LocalModelStore
from taoverse.model.storage.model_metadata_store import ModelMetadataStore
from taoverse.model.storage.remote_model_store import RemoteModelStore
from taoverse.model.utils import get_hash_of_two_strings


class MinerMisconfiguredError(Exception):
    """Error raised when a miner is misconfigured."""

    def __init__(self, hotkey: str, message: str):
        self.hotkey = hotkey
        super().__init__(f"[{hotkey}] {message}")


class ModelUpdater:
    """Checks if tracked models match the chain and fetches/validates new ones as needed."""

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
    def verify_model_satisfies_parameters(
        model: Model, model_constraints: ModelConstraints
    ) -> bool:
        if not model_constraints:
            logging.trace(f"No competition found for ID {model.id.competition_id}")
            return False

        # 1) Parameter‐count check
        parameter_count = sum(p.numel() for p in model.pt_model.parameters())
        if (
            parameter_count < model_constraints.min_model_parameter_size
            or parameter_count > model_constraints.max_model_parameter_size
        ):
            logging.debug(f"{model.id.name}: parameter count {parameter_count} outside "
                          f"[{model_constraints.min_model_parameter_size}, "
                          f"{model_constraints.max_model_parameter_size}]")
            return False

        # 2) Allowed architectures
        if type(model.pt_model) not in model_constraints.allowed_architectures:
            logging.debug(f"{model.id.name}: architecture {type(model.pt_model)} not allowed")
            return False

        # 3) Norm‐based sanity checks
        norm_cons = model_constraints.norm_validation_constraints
        if norm_cons is not None:
            return ModelUpdater._validate_parameters(
                model.pt_model,
                norm_cons.norm_eps_soft,
                norm_cons.norm_eps_soft_percent_threshold,
                norm_cons.norm_eps_hard,
            )

        return True

    async def _get_metadata(self, uid: int, hotkey: str) -> Optional[ModelMetadata]:
        return await self.metadata_store.retrieve_model_metadata(uid, hotkey)

    async def sync_model(
        self,
        uid: int,
        hotkey: str,
        curr_block: int,
        schedule_by_block: List[Tuple[int, List[Competition]]],
        force: bool = False,
    ) -> bool:
        """
        Checks chain metadata vs. local tracker, downloads new model if needed,
        validates against competition constraints, and updates the tracker.
        Returns True if a new model was fetched (even if invalid), False otherwise.
        """
        metadata = await self._get_metadata(uid, hotkey)
        if not metadata:
            raise MinerMisconfiguredError(hotkey, "No metadata found on chain")

        # Ensure competition existed at upload time
        comp_at_upload = competition_utils.get_competition_for_block(
            comp_id=metadata.id.competition_id,
            block=metadata.block,
            schedule_by_block=schedule_by_block,
        )
        if not comp_at_upload:
            raise MinerMisconfiguredError(
                hotkey,
                f"No competition {metadata.id.competition_id} at upload block {metadata.block}",
            )

        # Ensure competition is active now
        comp_now = competition_utils.get_competition_for_block(
            comp_id=metadata.id.competition_id,
            block=curr_block,
            schedule_by_block=schedule_by_block,
        )
        if not comp_now:
            raise MinerMisconfiguredError(
                hotkey,
                f"No competition {metadata.id.competition_id} at current block {curr_block}",
            )

        # Respect evaluation delay
        if curr_block - metadata.block < comp_now.constraints.eval_block_delay:
            logging.debug(
                f"Sync delayed for {hotkey}: need {comp_now.constraints.eval_block_delay} "
                f"blocks after {metadata.block}, now at {curr_block}"
            )
            return False

        # Short‐circuit if metadata unchanged and not forced
        prev_meta = self.model_tracker.get_metadata(hotkey)
        if not force and metadata == prev_meta:
            return False

        # Fetch new model
        download_path = self.local_store.get_path(hotkey)
        model = await self.remote_store.download_model(
            metadata.id, download_path, comp_now.constraints
        )

        # Update tracker immediately to avoid redownload loops
        self.model_tracker.on_model_updated(hotkey, metadata)

        # Legacy hash check (if used)
        if model.id.hash or metadata.id.hash:
            derived = get_hash_of_two_strings(model.id.hash or "", hotkey)
            if derived != metadata.id.secure_hash:
                raise MinerMisconfiguredError(
                    hotkey,
                    "Downloaded model hash does not match chain secure_hash",
                )

        # Parameter & architecture validation
        if not ModelUpdater.verify_model_satisfies_parameters(model, comp_now.constraints):
            raise MinerMisconfiguredError(
                hotkey,
                f"Model fails parameter constraints for competition {comp_now.id}",
            )

        return True

    @staticmethod
    def _validate_parameters(
        base_model: Any,
        eps_soft: float,
        eps_soft_percent_threshold: float,
        eps_hard: float,
        print_vals: bool = False
    ) -> bool:
        """
        Norm‐based sanity check on projection weights.
        Returns False immediately if any norm > eps_hard;
        else ensures the % of weights > eps_soft is below threshold.
        """
        exceed = defaultdict(int)
        total = defaultdict(int)

        # Iterate layers
        for layer in base_model.model.layers:
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"):
                norm_val = getattr(
                    layer.self_attn if proj.endswith("_proj") else layer.mlp,
                    proj
                ).weight.norm().item()

                total[proj] += 1
                if norm_val > eps_hard:
                    return False
                if norm_val > eps_soft:
                    exceed[proj] += 1

        # Compute percentages
        percents = [exceed[p] / total[p] for p in total]
        if print_vals:
            print({p: exceed[p] / total[p] for p in total})
        return statistics.fmean(percents) <= eps_soft_percent_threshold
