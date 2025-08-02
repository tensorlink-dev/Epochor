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

        # --- Batched case ---
        if target.ndim == 2:
            B, T = target.shape
            
            # Normalize prediction to (B, T, N)
            if prediction.ndim == 4:
                if prediction.shape[2] == 1:
                    prediction = prediction.squeeze(axis=2) # Remove the 1-sized feature dimension
                else:
                    raise ValueError(f"Prediction must be 2D or 3D or 4D with a 1-sized 3rd dimension, got {prediction.shape}")
            if prediction.ndim == 3:
                pass
            elif prediction.ndim == 2:
                # If prediction is (B, T), assume N=1 ensemble
                prediction = prediction[..., np.newaxis] # Result: (B, T, 1)
            else:
                raise ValueError(f"Prediction must be 2D or 3D, got {prediction.shape}")

            if prediction.shape[0] != B or prediction.shape[1] != T:
                raise ValueError(f"Shape mismatch: target {target.shape}, pred {prediction.shape}")

            # Compute CRPS for each time series in the batch
            batch_scores = []
            for i in range(B):
                # Each call to crps_ensemble expects (T,) observation and (T, N) forecasts
                # So, target[i] is (T,) and prediction[i] is (T, N)
                series_crps = crps_ensemble(observations=target[i], forecasts=prediction[i])
                batch_scores.append(np.mean(series_crps)) # Mean CRPS over time steps for this series
            return np.array(batch_scores) # Return array of shape (B,)

        # --- Single-series case (T,) target ---
        if target.ndim == 1:
            if prediction.ndim == 1:
                prediction = prediction[:, np.newaxis] # Add ensemble dimension (N=1)
            elif prediction.ndim != 2:
                raise ValueError(f"Prediction must be 1D or 2D for single series, but got {prediction.shape}")
            if target.shape[0] != prediction.shape[0]:
                raise ValueError(f"Length mismatch: target {target.shape[0]} vs pred {prediction.shape[0]}")

            # For single series, crps_ensemble returns (T,). Average it to a scalar array.
            return np.mean(crps_ensemble(observations=target, forecasts=prediction)) # Returns scalar array ()
        else:
            raise ValueError(f"Target must be 1D or 2D, but got {target.shape}")


EVALUATION_BY_COMPETITION = {
    EvalMethodId.CRPS_LOSS.value: CRPSEvaluator,
}

__all__ = ["BaseEvaluator", "CRPSEvaluator", "EVALUATION_BY_COMPETITION"]
