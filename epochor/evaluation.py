# epochor/evaluation.py

"""
Evaluator base and CRPS-like evaluator for Epochor subnet.
"""

import numpy as np
from typing import Any
from epochor.config import EPOCHOR_CONFIG


class BaseEvaluator:
    """
    Abstract base class for evaluation metrics.
    """

    def evaluate(self, target: np.ndarray, prediction: np.ndarray) -> float:
        """
        Compute score (e.g., loss, CRPS, accuracy) between prediction and ground truth.

        Args:
            target: array of shape [T]
            prediction: array of shape [T]

        Returns:
            scalar float score
        """
        raise NotImplementedError

    def score_to_weight(self, scores: np.ndarray) -> np.ndarray:
        """
        Convert raw scores into normalized weights using power law + optional boost.

        Args:
            scores: array of raw performance scores

        Returns:
            array of normalized weights summing to 1
        """
        scores = np.nan_to_num(scores, nan=0.0)
        scores = np.clip(scores, 0.0, None)

        if EPOCHOR_CONFIG.reward_exponent != 1.0:
            scores = np.power(scores, EPOCHOR_CONFIG.reward_exponent)

        total = scores.sum()
        if total <= 0:
            return np.ones_like(scores) / len(scores)

        return scores / total


class CRPSEvaluator(BaseEvaluator):
    """
    Default evaluator using squared error (placeholder for CRPS).
    """

    def evaluate(self, target: np.ndarray, prediction: np.ndarray) -> float:
        """
        Compute squared error (mean over time steps).
        Replace this with actual CRPS if using distributions.

        Args:
            target: [T]
            prediction: [T]

        Returns:
            Mean squared error
        """
        target = np.asarray(target)
        prediction = np.asarray(prediction)

        if target.shape != prediction.shape:
            raise ValueError("Prediction and target shapes must match.")

        return float(np.mean((target - prediction) ** 2))


__all__ = ["BaseEvaluator", "CRPSEvaluator"]
