import numpy as np
from properscoring import crps_ensemble
from epochor.evaluation.method import EvalMethodId
from epochor.utils import logging


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
            target: observations, expected shape (T,) or (B, T)
            prediction: forecasts, expected shape (T, N) or (B, T, N) or (B, T, 1, N)

        Returns:
            If single series: a scalar array (shape ())
            If batched:      array of shape (B,). Each element is the mean CRPS for that series.
        """
        target = np.asarray(target)
        prediction = np.asarray(prediction)

        # --- Sanitize inputs to replace NaNs/Infs ---
        if not np.all(np.isfinite(target)):
            logging.warning("Non-finite values found in target, replacing with 0.")
            target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

        if not np.all(np.isfinite(prediction)):
            logging.warning(f"Non-finite values found in prediction (shape: {prediction.shape}), replacing with 1e9.")
            prediction = np.nan_to_num(prediction, nan=1e9, posinf=1e9, neginf=-1e9)

        # --- Normalize prediction to have 3 dimensions (B, T, N) or 2 dimensions (T, N) ---
        # Handle 4D prediction (B, T, 1, N) -> squeeze out the 1-sized feature dimension
        if prediction.ndim == 4:
            if prediction.shape[2] == 1: # If the 3rd dim is the problematic feature dim of size 1
                prediction = prediction.squeeze(axis=2) # Result: (B, T, N)
            else:
                raise ValueError(f"Prediction 4D shape {prediction.shape} must have a 1-sized 3rd dimension if it represents a single feature.")

        # Handle 2D prediction: (T, N) or (B, T). If (B, T) or (T,), add ensemble dim N=1.
        if prediction.ndim == 2:
            # If target is 2D (B, T) and prediction is 2D (B, T), it implies N=1 ensemble per series
            if target.ndim == 2 and target.shape == prediction.shape:
                prediction = prediction[..., np.newaxis] # Result: (B, T, 1)
            # If target is 1D (T) and prediction is 1D (T), it implies N=1 ensemble for a single series
            elif target.ndim == 1 and target.shape == prediction.shape:
                prediction = prediction[..., np.newaxis] # Result: (T, 1)
            # If target is 1D (T) and prediction is 2D (T, N) with N > 1, it's already correct.
            # No change needed.
            elif target.ndim == 1 and prediction.shape[0] == target.shape[0] and prediction.shape[1] > 1:
                pass
            else:
                raise ValueError(f"Unsupported 2D prediction shape {prediction.shape} for target shape {target.shape}. Expected (T,N) or (B,T) as prediction for single series or (B,T) for batched target.")
        elif prediction.ndim == 1: # If prediction is 1D (T,), assume N=1 ensemble (for single series case)
             if target.ndim == 1 and target.shape == prediction.shape:
                 prediction = prediction[..., np.newaxis] # Result: (T, 1)
             else:
                 raise ValueError(f"Unsupported 1D prediction shape {prediction.shape} for target shape {target.shape}. Expected (T,).")

        # After normalization, prediction should be 3D (B, T, N) or 2D (T, N).

        # --- Validate the overall dimensionality and shape consistency before CRPS calculation ---
        if target.ndim == 2: # Batched case: target (B, T)
            B_target, T_target = target.shape
            if prediction.ndim != 3: # Must be (B, T, N) for batched target
                raise ValueError(f"For batched target {target.shape}, prediction must be 3D (B, T, N) after normalization, but got {prediction.shape}")
            B_pred, T_pred, N_ensemble = prediction.shape

            if B_pred != B_target or T_pred != T_target:
                raise ValueError(f"Shape mismatch in batched evaluation: target {target.shape}, prediction {prediction.shape}. Batch and Time dimensions must match.")

            # Compute CRPS for each time series in the batch
            batch_scores = []
            for i in range(B_target):
                # crps_ensemble expects observations=(T,) and forecasts=(T, N)
                series_crps = crps_ensemble(observations=target[i], forecasts=prediction[i])
                batch_scores.append(np.mean(series_crps)) # Mean CRPS over time steps for this series
            return np.array(batch_scores) # Return array of shape (B,)

        elif target.ndim == 1: # Single-series case: target (T,)
            T_target = target.shape[0]
            if prediction.ndim != 2: # Must be (T, N) for single-series target
                raise ValueError(f"For single series target {target.shape}, prediction must be 2D (T, N) after normalization, but got {prediction.shape}")
            T_pred, N_ensemble = prediction.shape

            if T_pred != T_target:
                raise ValueError(f"Length mismatch in single series evaluation: target {target.shape[0]} vs pred {prediction.shape[0]}. Time dimension must match.")

            # For single series, crps_ensemble returns (T,). Average it to a scalar array.
            return np.mean(crps_ensemble(observations=target, forecasts=prediction)) # Returns scalar array ()
        else:
            # This case should ideally be caught by the initial check, but as a fallback
            raise ValueError(f"Unhandled target dimension: {target.ndim}. Target must be 1D or 2D.")


EVALUATION_BY_COMPETITION = {
    EvalMethodId.CRPS_LOSS.value: CRPSEvaluator,
}

__all__ = ["BaseEvaluator", "CRPSEvaluator", "EVALUATION_BY_COMPETITION"]
