"""
Evaluator for the Epochor subnet, using the Continuous Ranked Probability Score.

This module provides the CRPSEvaluator, which uses the properscoring library
to compute the CRPS for ensemble forecasts.
"""

import numpy as np
from properscoring import crps_ensemble
from epochor.evaluation.method import EvalMethodId


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
        score to weight placeholder
        """
        raise NotImplementedError

class CRPSEvaluator(BaseEvaluator):
    """
    Extended CRPS evaluator that handles batched time series.
    """

    def evaluate(self, target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        """
        Args:
            target: shape (T,) or (B, T)
            prediction: shape (T, N) or (B, T, N)
        Returns:
            If single series: array of shape (T,)
            If batched:      array of shape (B, T)
        """
        target = np.asarray(target)
        prediction = np.asarray(prediction)

        # --- Batched case ---
        if target.ndim == 2:
            B, T = target.shape
            # Normalize prediction to (B, T, N)
            if prediction.ndim == 3:
                pass
            elif prediction.ndim == 2:
                prediction = prediction[..., np.newaxis]
            else:
                raise ValueError(f"Prediction must be 2D or 3D, got {prediction.shape}")

            if prediction.shape[0] != B or prediction.shape[1] != T:
                raise ValueError(f"Shape mismatch: target {target.shape}, pred {prediction.shape}")

            # Compute CRPS per series
            scores = np.zeros((B, T), dtype=float)
            for i in range(B):
                scores[i] = crps_ensemble(observations=target[i], forecasts=prediction[i])
            return scores

        # --- Single-series case (unchanged) ---
        if target.ndim != 1:
            raise ValueError(f"Target must be 1D or 2D, but got {target.shape}")
        if prediction.ndim == 1:
            prediction = prediction[:, np.newaxis]
        elif prediction.ndim != 2:
            raise ValueError(f"Prediction must be 1D or 2D, but got {prediction.shape}")
        if target.shape[0] != prediction.shape[0]:
            raise ValueError(f"Length mismatch: target {target.shape[0]} vs pred {prediction.shape[0]}")

        return crps_ensemble(observations=target, forecasts=prediction)


EVALUATION_BY_COMPETITION = {
    EvalMethodId.CRPS_LOSS.value: CRPSEvaluator,
}

__all__ = ["BaseEvaluator", "CRPSEvaluator", "EVALUATION_BY_COMPETITION"]
