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

        # Handle 4D prediction (B, T, 1, N) -> (B, T, N)
        if prediction.ndim == 4:
            if prediction.shape[2] == 1:
                prediction = prediction.squeeze(axis=2) # Result: (B, T, N)
            else:
                raise ValueError(f"Prediction 4D shape must have a 1-sized 3rd dimension for squeezing, got {prediction.shape}")
        
        # Handle 2D prediction (T, N) or (B, T) -> add ensemble dim if needed
        if prediction.ndim == 2:
            # If target is 2D (B, T) and prediction is 2D (B, T), assume N=1 ensemble
            if target.ndim == 2 and target.shape == prediction.shape:
                 prediction = prediction[..., np.newaxis] # Result: (B, T, 1)
            # If target is 1D (T) and prediction is 1D (T), assume N=1 ensemble
            elif target.ndim == 1 and target.shape == prediction.shape:
                prediction = prediction[..., np.newaxis] # Result: (T, 1)
            # If target is 1D (T) and prediction is 2D (T,N) with N>1, it's already correct. No change.
            # Otherwise, invalid 2D prediction for the given target dimensionality.
            elif target.ndim == 1 and prediction.shape[0] == target.shape[0]:
                pass # Already (T, N), nothing to do
            else:
                raise ValueError(f"Unsupported 2D prediction shape {prediction.shape} for target shape {target.shape}")

        # After the above, prediction should be 3D (B, T, N) or 2D (T, N).
        # Now proceed based on target dimensionality.

        if target.ndim == 2: # Batched case: target (B, T)
            B, T_target = target.shape
            if prediction.ndim != 3:
                raise ValueError(f"For batched target {target.shape}, prediction must be 3D (B, T, N) after normalization, but got {prediction.shape}")
            B_pred, T_pred, N_ensemble = prediction.shape
            if B_pred != B or T_pred != T_target:
                raise ValueError(f"Shape mismatch in batched evaluation: target {target.shape}, prediction {prediction.shape}")

            # Compute CRPS for each time series in the batch
            batch_scores = []
            for i in range(B):
                # crps_ensemble expects observations=(T,) and forecasts=(T, N)
                series_crps = crps_ensemble(observations=target[i], forecasts=prediction[i])
                batch_scores.append(np.mean(series_crps)) # Mean CRPS over time steps for this series
            return np.array(batch_scores) # Return array of shape (B,)

        else: # Single-series case: target (T,)
            T_target = target.shape[0]
            if prediction.ndim != 2:
                raise ValueError(f"For single series target {target.shape}, prediction must be 2D (T, N) after normalization, but got {prediction.shape}")
            T_pred, N_ensemble = prediction.shape
            if T_pred != T_target:
                raise ValueError(f"Length mismatch in single series evaluation: target {target.shape[0]} vs pred {prediction.shape[0]}")

            # For single series, crps_ensemble returns (T,). Average it to a scalar array.
            return np.mean(crps_ensemble(observations=target, forecasts=prediction)) # Returns scalar array ()


EVALUATION_BY_COMPETITION = {
    EvalMethodId.CRPS_LOSS.value: CRPSEvaluator,
}

__all__ = ["BaseEvaluator", "CRPSEvaluator", "EVALUATION_BY_COMPETITION"]
