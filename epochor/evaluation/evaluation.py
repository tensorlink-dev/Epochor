import numpy as np
from properscoring import crps_ensemble
from epochor.evaluation.method import EvalMethodId


class BaseEvaluator:
    """
    Abstract base class for evaluation metrics.
    """

    def evaluate(self, target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        """
        Compute score (e.g., loss, CRPS, accuracy) between prediction and ground truth.

        Args:
            target: array of ground truth values. Shape (T,) or (B, T)
            prediction: array of predicted values or ensembles. Shape (T, N) or (B, T, N) or (B, T, 1, N)

        Returns:
            An array of scores, one value per time series (shape (B,)).
            If single series: a scalar array (shape ()).
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
            prediction: shape (T, N) or (B, T, N) or (B, T, 1, N)
        Returns:
            If single series: a scalar array (shape ()).
            If batched:      array of shape (B,). Each element is the mean CRPS for that series.
        """
        target = np.asarray(target)
        prediction = np.asarray(prediction)

        if target.ndim not in [1, 2]:
            raise ValueError(f"Target must be 1D or 2D, but got shape {target.shape}")

        if prediction.ndim not in [2, 3, 4]:
            raise ValueError(f"Prediction must be 2D, 3D or 4D, but got shape {prediction.shape}")

        # Handle 4D prediction: squeeze out the 1-sized feature dimension
        if prediction.ndim == 4:
            if prediction.shape[2] == 1:
                prediction = prediction.squeeze(axis=2) # Result (B, T, N)
            else:
                raise ValueError(f"Prediction 4D shape must have a 1-sized 3rd dimension for squeezing, got {prediction.shape}")

        # Ensure prediction has an explicit ensemble dimension if it's currently (B, T)
        if prediction.ndim == 2: # This covers (T, N) for single series and (B, T) for batched (N=1)
            prediction = prediction[..., np.newaxis] # Result (T, N, 1) or (B, T, 1)

        # Now, prediction should be (T, N) or (B, T, N)

        if target.ndim == 2: # Batched case
            B, T_target = target.shape
            B_pred, T_pred, N_ensemble = prediction.shape

            if B_pred != B or T_pred != T_target: # This check is crucial
                raise ValueError(f"Shape mismatch in batched evaluation: target {target.shape}, prediction {prediction.shape}")

            # Compute CRPS for each time series in the batch
            batch_scores = []
            for i in range(B):
                # crps_ensemble expects observations=(T,) and forecasts=(T, N)
                series_crps = crps_ensemble(observations=target[i], forecasts=prediction[i])
                batch_scores.append(np.mean(series_crps)) # Mean CRPS over time steps for this series
            return np.array(batch_scores) # Return array of shape (B,)

        else: # Single-series case (target.ndim == 1)
            T_target = target.shape[0]
            T_pred, N_ensemble = prediction.shape

            if T_pred != T_target:
                raise ValueError(f"Length mismatch in single series evaluation: target {target.shape[0]} vs pred {prediction.shape[0]}")

            # For single series, crps_ensemble returns (T,). Average it to a scalar array.
            return np.mean(crps_ensemble(observations=target, forecasts=prediction)) # Returns scalar array ()


EVALUATION_BY_COMPETITION = {
    EvalMethodId.CRPS_LOSS.value: CRPSEvaluator,
}

__all__ = ["BaseEvaluator", "CRPSEvaluator", "EVALUATION_BY_COMPETITION"]
