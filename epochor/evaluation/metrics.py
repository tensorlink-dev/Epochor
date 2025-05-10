"""
Evaluation metrics for time series forecasting and generation models.

This module provides implementations for common metrics such as
Mean Squared Error (MSE), and placeholders or wrappers for others like
Continuous Ranked Probability Score (CRPS). It also includes a
utility for computing a composite score from multiple weighted metrics.
"""
from typing import Callable, Dict # Standard library imports

import numpy as np # Third-party imports

def mse(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Calculate the Mean Squared Error (MSE) between predictions and true values.

    Args:
        pred: A numpy array of predicted values.
        true: A numpy array of true values. Must have the same shape as pred.

    Returns:
        The mean squared error as a float.
    """
    if pred.shape != true.shape:
        raise ValueError("Predictions and true values must have the same shape.")
    return float(np.mean((pred - true)**2))

def crps(
    pred_cdf: Callable[[float], float], # Placeholder for a CDF or ensemble predictions
    true_value: float
) -> float:
    """
    Calculate the Continuous Ranked Probability Score (CRPS).
    
    This is a placeholder function. A proper implementation would typically
    use a library like `properscoring` and operate on probabilistic forecasts
    (e.g., an ensemble of predictions or parameters of a predictive distribution).

    Args:
        pred_cdf: A function representing the predicted cumulative distribution
                  function (CDF), or an array of ensemble predictions.
        true_value: The single observed true value.

    Returns:
        The CRPS score as a float.
        
    Raises:
        NotImplementedError: As this is a placeholder.
    """
    # Example usage with properscoring (if pred_cdf were an ensemble array):
    # import properscoring as ps
    # return ps.crps_ensemble(observations=true_value, forecasts=pred_cdf).mean()
    raise NotImplementedError("CRPS calculation is not yet implemented. Use e.g. properscoring.crps_ensemble.")

def composite(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Calculate a weighted composite score from a dictionary of individual metric scores.

    If a metric in `scores` does not have a corresponding weight in `weights`,
    a default weight of 1.0 is used for that metric.

    Args:
        scores: A dictionary where keys are metric names (str) and values are
                their scores (float).
        weights: A dictionary where keys are metric names (str) and values are
                 their corresponding weights (float).

    Returns:
        The composite score as a float.
    """
    total_score = 0.0
    for metric_name, score_value in scores.items():
        total_score += score_value * weights.get(metric_name, 1.0)
    return total_score

__all__ = ["mse", "crps", "composite"]
