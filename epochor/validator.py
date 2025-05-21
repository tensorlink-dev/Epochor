# epochor/validator.py

"""
Main validator logic for Epochor subnet with full error handling.
"""

import time
import numpy as np
import bittensor as bt
from epochor.config import EPOCHOR_CONFIG
from epochor.generators import CombinedGenerator
from epochor.evaluation import CRPSEvaluator
# from epochor.validation import validate # Using EMA scores directly
from epochor.rewards import allocate_rewards
from epochor.api import load_hf
from epochor.logging import reinitialize
from epochor.ema_tracker import EMATracker
from epochor.metrics_logger import MetricsLogger

logger = reinitialize()


class EpochorValidator(bt.Validator):
    def __init__(self, config: bt.Config):
        super().__init__(config=config)
        self.generator = CombinedGenerator(
            samplers=EPOCHOR_CONFIG.samplers,
            registries=EPOCHOR_CONFIG.registries,
            length=EPOCHOR_CONFIG.series_length,
            weights=EPOCHOR_CONFIG.category_weights
        )
        self.evaluator = CRPSEvaluator()
        # self.history = {} # Replaced by EMATracker for scores used in reward allocation
        self.previous_block = -1

        # Initialize EMATracker
        if EPOCHOR_CONFIG.ema_span <= 0:
            logger.warning("EMA span is non-positive, EMA tracking might behave unexpectedly. Setting alpha to 1.0 (no smoothing).")
            alpha = 1.0
        else:
            alpha = 2 / (EPOCHOR_CONFIG.ema_span + 1)
        self.ema_tracker = EMATracker(alpha=alpha)

        # Initialize WandB Metrics Logger
        self.metrics_logger = MetricsLogger(
            project_name=EPOCHOR_CONFIG.wandb_project,
            entity=EPOCHOR_CONFIG.wandb_entity,
            disabled=not EPOCHOR_CONFIG.use_wandb
        )

    def forward(self):
        start_time = time.time()

        try:
            current_block = bt.utils.get_current_block()
        except Exception as e:
            logger.error(f"Failed to get current block: {e}")
            self.metrics_logger.log({"error_get_current_block": 1})
            return

        if current_block <= self.previous_block:
            # Log if running frequently, otherwise it's normal if just restarted
            if self.previous_block != -1: # Avoid warning on first run
                 logger.warning(f"Block {current_block} did not advance from {self.previous_block}")
            # self.previous_block = current_block # Update to current block even if not advanced, to handle restarts correctly
            return # Do not proceed if block has not advanced
        
        # Only update previous_block if the current block is greater
        # and we are proceeding with the round.
        # self.previous_block = current_block # Moved this to after successful round processing or at least after UID fetch

        try:
            series, tag = self.generator.generate(seed=current_block)
        except Exception as e:
            logger.error(f"Failed to generate time-series batch: {e}")
            self.metrics_logger.log({"error_generate_series": 1}, step=current_block)
            return

        try:
            all_uids = self.subtensor.get_current_uids(self.netuid)
            # Filter out uids that are not serving. TODO: Implement a more robust check if possible.
            metagraph = self.subtensor.metagraph(self.netuid)
            active_uids = [uid.item() for uid in metagraph.uids if metagraph.axons[uid.item()].is_serving]

        except Exception as e:
            logger.error(f"Failed to fetch or filter miner UIDs: {e}")
            self.metrics_logger.log({"error_fetch_uids": 1}, step=current_block)
            return

        if not active_uids:
            logger.warning("No active UIDs found.")
            self.metrics_logger.log({"active_uids_count": 0}, step=current_block)
            # self.previous_block = current_block # Update block even if no UIDs, to keep sync with chain
            return
        
        # Now that we have UIDs and are proceeding, update previous_block
        # self.previous_block = current_block # Moved further down

        raw_scores_this_round = {}
        evaluated_uids_count = 0
        for uid in active_uids:
            try:
                # TODO: Query the specific axon for this UID instead of relying on load_hf to find it.
                # axon = metagraph.axons[uid]
                # if not axon.is_serving: 
                #     logger.info(f"UID {uid} is not serving. Skipping.")
                #     raw_scores_this_round[uid] = np.nan # Or some other placeholder
                #     continue

                model = load_hf(uid) # This might need adjustment if it relies on an old metagraph or UID list
                if model is None: # load_hf might return None if model not found or error
                    raise ValueError("Model could not be loaded from Hugging Face")
                
                preds = model(series) # series needs to be in the format expected by the model
                loss = self.evaluator.evaluate(series, preds)

                if not np.isfinite(loss):
                    raise ValueError("Non-finite loss calculated")
                
                # Store raw score for this round for logging
                raw_scores_this_round[uid] = loss 
                # Update EMA score
                self.ema_tracker.update(uid, loss) 
                logger.info(f"Scored UID {uid} | Raw Loss: {loss:.4f} | EMA Score: {self.ema_tracker.get(uid):.4f}")
                evaluated_uids_count += 1

            except Exception as e:
                logger.warning(f"Evaluation failed for UID {uid}: {e}")
                raw_scores_this_round[uid] = np.nan # Log NaN for this round
                # Do not update EMA for failed evaluations, or update with a penalty if desired.
                # For now, EMA for this UID will remain unchanged until a successful evaluation.
        
        current_ema_scores = self.ema_tracker.get_all_scores()
        
        # Filter EMA scores to only include UIDs active in *this* round, or handle appropriately
        # For allocation, we should only use scores of UIDs we attempted to query or were present.
        scores_for_allocation = {uid: current_ema_scores.get(uid, 0.0) for uid in active_uids 
                                 if uid in current_ema_scores and np.isfinite(current_ema_scores.get(uid, np.nan))}
        
        # If a UID was active but has no EMA score (e.g. first time seen and failed eval), it won't be in scores_for_allocation
        # Ensure all active UIDs get some score, e.g. a default low score or are handled by allocate_rewards if not present
        for uid in active_uids:
            if uid not in scores_for_allocation:
                 # Assign a default (e.g., neutral or penalty) if no valid EMA score exists. 
                 # allocate_rewards should handle missing UIDs or UIDs with 0 scores appropriately.
                 scores_for_allocation[uid] = 0.0 # Or np.nan, if allocate_rewards is robust to it
                 logger.info(f"UID {uid} has no valid EMA score for allocation, defaulting to 0.0")

        if not scores_for_allocation:
            logger.warning("No valid scores available for reward allocation after EMA processing.")
            self.metrics_logger.log({
                "block": current_block,
                "round_duration": time.time() - start_time,
                "active_uids_count": len(active_uids),
                "evaluated_uids_count": evaluated_uids_count,
                "uids_for_allocation_count": 0,
                "error_no_scores_for_allocation": 1
            }, step=current_block)
            self.previous_block = current_block # Update block to keep sync
            return

        try:
            # `validate` function is removed; EMA scores are used directly.
            # `scores = validate(self.history)` replaced by `current_ema_scores`
            # Ensure scores_for_allocation contains values that allocate_rewards expects (e.g., higher is better)
            # Our current losses are 'lower is better'. EMA tracker stores these directly.
            # allocate_rewards expects 'higher is better'. So we need to invert scores (e.g., 1/loss or max_loss - loss)
            # For simplicity, let's assume allocate_rewards can handle lower-is-better scores by a config, 
            # or we invert them here. Let's invert: score = 1 / (1 + loss) to make it bounded and higher is better.
            inverted_ema_scores = {}
            for uid, ema_loss in scores_for_allocation.items():
                if np.isnan(ema_loss):
                    inverted_ema_scores[uid] = 0.0 # Penalize NaNs from EMA
                else:
                    inverted_ema_scores[uid] = 1.0 / (1.0 + ema_loss) # Ensure positive, higher is better

        except Exception as e:
            logger.error(f"Failed during score inversion or preparation for allocation: {e}")
            self.metrics_logger.log({"error_score_inversion": 1}, step=current_block)
            self.previous_block = current_block
            return
            
        try:
            weights = allocate_rewards(inverted_ema_scores) # allocate_rewards uses the strategy from EPOCHOR_CONFIG
        except Exception as e:
            logger.error(f"Reward allocation failed: {e}")
            self.metrics_logger.log({"error_allocate_rewards": 1}, step=current_block)
            self.previous_block = current_block # Update block to keep sync
            return

        # Filter weights for UIDs that are still considered active or are appropriate to set weights for
        # This step might be redundant if active_uids were already used for scores_for_allocation correctly
        final_weights_uids = [uid for uid in active_uids if uid in weights and np.isfinite(weights[uid])]
        final_weights_values = [weights[uid] for uid in final_weights_uids]

        if not final_weights_uids:
            logger.warning("No valid weights to submit after allocation.")
            self.metrics_logger.log({
                "block": current_block,
                "round_duration": time.time() - start_time,
                "active_uids_count": len(active_uids),
                "evaluated_uids_count": evaluated_uids_count,
                "uids_for_allocation_count": len(scores_for_allocation),
                "inverted_scores_for_allocation_count": len(inverted_ema_scores),
                "weights_allocated_count": 0,
                "warning_no_weights_to_submit": 1
            }, step=current_block)
            self.previous_block = current_block
            return
        
        try:
            logger.info(f"Submitting weights for {len(final_weights_uids)} miners. Sum: {sum(final_weights_values):.2f}")
            self.subtensor.set_weights(
                netuid=self.netuid, # Ensure netuid is passed if required by the version of bittensor
                uids=np.array(final_weights_uids, dtype=np.int64),
                weights=np.array(final_weights_values, dtype=np.float32),
                # block=current_block # Block might not be needed for new bittensor versions
                wait_for_inclusion=False, # Faster, but less certain
                wait_for_finalization=False,
                version_key = bt.__version_as_int__ # For compatibility
            )
            logger.info(f"Successfully set weights for UIDs: {final_weights_uids}")
            self.previous_block = current_block # Update block only after successful operation including set_weights

        except Exception as e:
            logger.error(f"set_weights failed: {e}")
            self.metrics_logger.log({"error_set_weights": 1, "num_uids_set_weights_attempt": len(final_weights_uids)}, step=current_block)
            # self.previous_block remains unchanged here, so this block might be re-processed or attempted again
            # depending on the main loop and how often forward() is called.
            return

        # --- WandB Logging --- 
        duration = time.time() - start_time
        log_data = {
            "block": current_block,
            "round_duration_seconds": duration,
            "active_uids_count": len(active_uids),
            "evaluated_uids_count": evaluated_uids_count,
            "uids_for_allocation_count": len(scores_for_allocation),
            "weights_submitted_count": len(final_weights_uids),
            "total_weight_sum_submitted": sum(final_weights_values),
        }

        # Per-UID logs
        uid_metrics = {}
        for uid_ in active_uids: # Log for all active UIDs, even if they failed or had no weight
            uid_metrics[f"raw_score_uid_{uid_}"] = raw_scores_this_round.get(uid_, np.nan)
            uid_metrics[f"ema_score_uid_{uid_}"] = current_ema_scores.get(uid_, np.nan)
            # inverted_ema_scores might not have all uids if they had no ema score initially
            uid_metrics[f"inverted_ema_score_uid_{uid_}"] = inverted_ema_scores.get(uid_, np.nan) 
            uid_metrics[f"final_weight_uid_{uid_}"] = weights.get(uid_, np.nan)
        log_data.update(uid_metrics)

        # Histograms (requires wandb.Histogram)
        if not self.metrics_logger.disabled and wandb.Histogram:
            if raw_scores_this_round:
                log_data["raw_score_histogram"] = wandb.Histogram(list(filter(np.isfinite, raw_scores_this_round.values())))
            if current_ema_scores:
                log_data["ema_score_histogram"] = wandb.Histogram(list(filter(np.isfinite, current_ema_scores.values())))
            if inverted_ema_scores:
                 log_data["inverted_ema_score_histogram"] = wandb.Histogram(list(filter(np.isfinite, inverted_ema_scores.values())))
            if final_weights_values:
                log_data["final_weights_histogram"] = wandb.Histogram(final_weights_values) # Already filtered for finite by final_weights_uids

        # Top 5 UIDs by EMA score (inverted, so higher is better)
        if inverted_ema_scores:
            sorted_inverted_scores = sorted(inverted_ema_scores.items(), key=lambda item: item[1], reverse=True)
            top_5_uids_data = {f"top_5_uid_{i+1}": uid for i, (uid, score) in enumerate(sorted_inverted_scores[:5])}
            top_5_scores_data = {f"top_5_score_uid_{uid}": score for uid,score in sorted_inverted_scores[:5]}
            log_data.update(top_5_uids_data)
            log_data.update(top_5_scores_data)

        self.metrics_logger.log(log_data, step=current_block)
        logger.info(f"Block {current_block} round complete in {duration:.2f}s. Logged to WandB.")

    # It's good practice to clean up resources if any were explicitly opened.
    def __del__(self):
        if hasattr(self, 'metrics_logger') and self.metrics_logger:
            self.metrics_logger.finish()
