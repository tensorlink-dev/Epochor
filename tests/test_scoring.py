# tests/test_scoring.py

"""
Unit tests for the Epochor scoring pipeline and reward allocation.
Includes a simulation to validate ranking over multiple rounds.
"""

import unittest
import numpy as np
import random
from typing import Dict

# Assuming these can be imported or mocked for testing
from epochor.config import EPOCHOR_CONFIG
from epochor.rewards import allocate_rewards # Using the actual reward allocation
from epochor.ema_tracker import EMATracker

# Mocking the score generation part for the simulation
# In a real scenario, this would be more sophisticated, simulating miner behavior

def generate_mock_scores(uids: list, round_number: int) -> Dict[int, float]:
    """
    Generates mock raw scores (losses, lower is better) for a list of UIDs.
    Scores can vary per round to simulate changing performance.
    """
    scores = {}
    for uid in uids:
        # Simulate some base performance + noise + round-based trend
        base_performance = uid * 0.01 # Different UIDs have different inherent skill
        noise = random.uniform(-0.05, 0.05)
        trend = round_number * 0.001 # Simulate slight improvement over time for some
        
        # Simulate occasional very good or bad scores
        if random.random() < 0.05: # 5% chance of a much lower (better) score
            score = random.uniform(0.01, 0.1)
        elif random.random() < 0.05: # 5% chance of a much higher (worse) score or failure
            score = random.uniform(1.0, 2.0)
            if random.random() < 0.1: # 0.5% chance of NaN
                 score = np.nan
        else:
            score = max(0.01, 0.1 + base_performance + noise - trend + uid % 0.1) # Ensure positive
        scores[uid] = score
    return scores

class TestScoringSimulation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up parameters for the simulation."""
        cls.num_miners = 20
        cls.num_rounds = 100
        cls.uids = list(range(cls.num_miners))

        # Configure Epochor for testing - use a specific reward strategy if needed
        # For example, to test softmax directly:
        EPOCHOR_CONFIG.reward_strategy = "softmax" # or "linear", "ranked_decay"
        EPOCHOR_CONFIG.reward_temperature = 0.1 # Sharper distribution for testing ranking
        EPOCHOR_CONFIG.min_weight_threshold = 0.0001
        EPOCHOR_CONFIG.reward_exponent = 1.0 # No exponentiation for simpler analysis
        EPOCHOR_CONFIG.first_place_boost = 1.0 # No boost for simpler analysis
        EPOCHOR_CONFIG.ema_span = 10 # Reasonable EMA span for simulation
        alpha = 2 / (EPOCHOR_CONFIG.ema_span + 1)
        cls.ema_tracker = EMATracker(alpha=alpha)
        
        cls.simulation_results = {
            "round_raw_scores": [],
            "round_ema_scores": [],
            "round_inverted_ema_scores": [],
            "round_weights": [],
            "final_ema_scores": {},
            "final_rank_by_ema": []
        }

    def test_simulation_and_ranking(self):
        """Simulate multiple rounds, apply EMA, allocate rewards, and validate ranking."""
        
        all_round_weights = []

        for i in range(self.num_rounds):
            # 1. Generate mock raw scores (losses) for the current round
            raw_scores_this_round = generate_mock_scores(self.uids, round_number=i)
            self.simulation_results["round_raw_scores"].append(raw_scores_this_round)

            # 2. Update EMA tracker with raw scores
            for uid, raw_score in raw_scores_this_round.items():
                if np.isfinite(raw_score):
                    self.ema_tracker.update(uid, raw_score)
            
            current_ema_losses = self.ema_tracker.get_all_scores()
            self.simulation_results["round_ema_scores"].append(current_ema_losses.copy())

            # 3. Prepare scores for allocation (invert: higher is better, handle NaNs)
            inverted_ema_scores = {}
            for uid in self.uids: # Ensure all UIDs are considered for weights
                ema_loss = current_ema_losses.get(uid, np.nan) # Get EMA, default to NaN if not seen
                if np.isnan(ema_loss) or not np.isfinite(ema_loss):
                    inverted_ema_scores[uid] = 0.0 # Penalize NaNs or non-finite EMA scores
                else:
                    inverted_ema_scores[uid] = 1.0 / (1.0 + ema_loss) # Lower loss = higher score
            self.simulation_results["round_inverted_ema_scores"].append(inverted_ema_scores.copy())

            # 4. Allocate rewards based on inverted EMA scores
            weights = allocate_rewards(inverted_ema_scores.copy()) # Use a copy
            self.simulation_results["round_weights"].append(weights.copy())
            all_round_weights.append(weights.copy())

            # Basic checks per round
            self.assertAlmostEqual(sum(weights.values()), 1.0, places=5, msg=f"Round {i}: Weights do not sum to 1")
            for uid in self.uids:
                self.assertIn(uid, weights, msg=f"Round {i}: UID {uid} missing from weights")
                self.assertGreaterEqual(weights[uid], 0, msg=f"Round {i}: UID {uid} has negative weight")

        # --- Post-simulation analysis ---
        final_ema_losses = self.ema_tracker.get_all_scores()
        self.simulation_results["final_ema_scores"] = final_ema_losses

        # Determine final ranking based on final EMA losses (lower loss is better rank)
        # Filter out UIDs that might not have scores if they consistently failed
        valid_final_ema_losses = {uid: loss for uid, loss in final_ema_losses.items() if np.isfinite(loss)}
        if not valid_final_ema_losses:
            raise AssertionError("No valid final EMA scores to rank.")
            
        sorted_uids_by_final_ema = sorted(valid_final_ema_losses.keys(), key=lambda uid: valid_final_ema_losses[uid])
        self.simulation_results["final_rank_by_ema"] = sorted_uids_by_final_ema

        print("
--- TestScoringSimulation Results ---")
        print(f"Simulated {self.num_rounds} rounds for {self.num_miners} miners.")
        print(f"Reward Strategy: {EPOCHOR_CONFIG.reward_strategy}, Temp: {EPOCHOR_CONFIG.reward_temperature}")
        print("Final Ranking by EMA loss (lower is better, top 5):")
        for rank, uid in enumerate(sorted_uids_by_final_ema[:5]):
            print(f"  Rank {rank+1}: UID {uid} (EMA Loss: {final_ema_losses.get(uid, np.nan):.4f})")
        
        # Validate ranking consistency (example: does lower final EMA loss generally lead to higher average weight?)
        # This is a heuristic check, perfect correlation isn't always expected due to EMA lag, non-linearities, etc.
        avg_weights = {uid: np.mean([round_weights.get(uid, 0) for round_weights in all_round_weights]) for uid in self.uids}
        
        # Check if top-ranked UIDs (by final EMA) have higher average weights than bottom-ranked UIDs
        top_n = max(1, self.num_miners // 4) # Top 25%
        bottom_n = max(1, self.num_miners // 4) # Bottom 25%

        avg_weight_top_ranked = np.mean([avg_weights[uid] for uid in sorted_uids_by_final_ema[:top_n]])
        avg_weight_bottom_ranked = np.mean([avg_weights[uid] for uid in sorted_uids_by_final_ema[-bottom_n:]])

        print(f"Avg weight for top {top_n} UIDs (by final EMA): {avg_weight_top_ranked:.6f}")
        print(f"Avg weight for bottom {bottom_n} UIDs (by final EMA): {avg_weight_bottom_ranked:.6f}")

        self.assertGreater(avg_weight_top_ranked, avg_weight_bottom_ranked,
                           msg="Average weight of top-ranked UIDs (by EMA) should generally be higher than bottom-ranked UIDs.")

        # Further checks can be added:
        # - Distribution of weights (e.g., Gini coefficient or entropy if relevant).
        # - Sensitivity to EMATracker alpha or reward temperature.
        # - Impact of first-place boost or min_weight_threshold if enabled.

if __name__ == '__main__':
    unittest.main()
