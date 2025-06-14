"""
Evaluator for the Epochor subnet, using the Continuous Ranked Probability Score.

This module provides the CRPSEvaluator, which uses the properscoring library
to compute the CRPS for ensemble forecasts.
"""

import numpy as np
from properscoring import crps_ensemble

from epochor.config import EPOCHOR_CONFIG


class BaseEvaluator:
    """
    Abstract base class for evaluation metrics.
    """

    def evaluate(self, target: np.ndarray, prediction: np.ndarray) -> float:
        """
        Compute score (e.g., loss, CRPS, accuracy) between prediction and ground truth.

        Args:
            target: array of ground truth values.
            prediction: array of predicted values or ensembles.

        Returns:
            A scalar float score for the prediction.
        """
        raise NotImplementedError

    def score_to_weight(self, scores: np.ndarray) -> np.ndarray:
        """
        Convert raw scores into normalized weights. Lower scores are considered better.

        The conversion uses a power law and optional boosting based on configuration.
        A lower score results in a higher weight.

        Args:
            scores: array of raw performance scores (lower is better).

        Returns:
            array of normalized weights summing to 1.
        """
        # Invert scores so that lower scores get higher values.
        # Add a small epsilon to avoid division by zero.
        inverted_scores = 1 / (scores + 1e-9)
        
        # Replace any NaNs or infinities that may have occurred
        inverted_scores = np.nan_to_num(inverted_scores, nan=0.0, posinf=np.nanmax(inverted_scores[np.isfinite(inverted_scores)]))
        
        # Clip scores to be non-negative
        processed_scores = np.clip(inverted_scores, 0.0, None)

        if EPOCHOR_CONFIG.reward_exponent != 1.0:
            processed_scores = np.power(processed_scores, EPOCHOR_CONFIG.reward_exponent)

        total = processed_scores.sum()
        if total <= 0:
            # If all scores are zero, return uniform weights
            return np.ones_like(processed_scores) / len(processed_scores) if len(processed_scores) > 0 else np.array([])

        return processed_scores / total


class CRPSEvaluator(BaseEvaluator):
    """
    Evaluator using the ensemble Continuous Ranked Probability Score (CRPS).

    This evaluator is designed for probabilistic forecasting, where the prediction
    is an ensemble of possible future outcomes.
    """
    def eval_task(self) -> str:
        """Returns the name of the evaluation task."""
        return 'CRPS'

    def evaluate(self, target: np.ndarray, prediction: np.ndarray) -> float:
        """
        Computes the mean ensemble CRPS.

        Args:
            target (np.ndarray): Ground truth values, expected shape [T].
            prediction (np.ndarray): Ensemble predictions, expected shape [T, N_ensemble_members].
                                     If a 1D array is passed, it will be treated as an
                                     ensemble with a single member.

        Returns:
            float: The mean CRPS score over the time series. A lower score is better.
        """
        target = np.asarray(target)
        prediction = np.asarray(prediction)

        # --- Shape Validation ---
        if target.ndim != 1:
            raise ValueError(f"Target must be a 1D array, but got shape {target.shape}")
        
        if prediction.ndim != 2:
            # Handle the case of a deterministic forecast by creating a dummy ensemble dimension
            if prediction.ndim == 1:
                prediction = prediction[:, np.newaxis]
            else:
                raise ValueError(f"Prediction must be a 2D ensemble array, but got shape {prediction.shape}")
        
        if target.shape[0] != prediction.shape[0]:
            raise ValueError(f"Time dimension mismatch: target shape {target.shape[0]} vs prediction shape {prediction.shape[0]}")

        # --- CRPS Calculation ---
        # crps_ensemble returns a score for each of the T observations.
        # We take the mean to get a single scalar score for the entire series.
        crps_scores = crps_ensemble(observations=target, forecasts=prediction)
        
        return float(np.mean(crps_scores))


__all__ = ["BaseEvaluator", "CRPSEvaluator"]
