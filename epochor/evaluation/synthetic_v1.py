"""
Benchmarker implementation for V1 synthetic time series data.

This module defines a concrete Benchmarker that uses the V1 synthetic
data generators (specifically `random_blended_config` and `TimeSeriesGenerator`
from the `epochor.generators` package) to create evaluation datasets.
It primarily computes Mean Squared Error (MSE) as the evaluation metric.
"""
from typing import Dict, Any # Standard library imports

import numpy as np # Third-party imports

# Local application/library specific imports
from .base import Benchmarker
from .metrics import mse
# Assuming the generator structures are as previously refactored.
# The prompt refers to TimeSeriesGenerator.from_config(cfg)
# This implies TimeSeriesGenerator might need a classmethod `from_config`.
# Or, it means `random_blended_config` returns a config that is directly
# passable to a BlendedSeriesGeneratorV1 (or similar) constructor.
# For now, I will assume `epochor.generators.base.BlendedSeriesGeneratorV1`
# is the intended generator, and `random_blended_config` is from `epochor.generators.sampler`.

# Attempting to import the specific generator and config sampler
try:
    from epochor.generators.base import BlendedSeriesGeneratorV1 # More specific than TimeSeriesGenerator
    from epochor.generators.sampler import random_blended_config
except ImportError as e:
    # Provide placeholders if imports fail, to allow module definition
    # This helps in incremental development or if generator paths change.
    import logging
    logging.error(
        "Failed to import BlendedSeriesGeneratorV1 or random_blended_config "
        f"from epochor.generators: {e}. SyntheticBenchmarkerV1 will not be fully functional."
    )
    # Define placeholder for BlendedSeriesGeneratorV1 if not found
    class BlendedSeriesGeneratorV1: # type: ignore
        def __init__(self, config: Dict[str, Any]):
            self.config = config
            self.length = config.get("length", 0)
            logging.warning("Using placeholder BlendedSeriesGeneratorV1.")
        def generate(self, seed: int) -> np.ndarray:
            return np.zeros((self.length,1))

    def random_blended_config(length: int, seed: int) -> Dict[str, Any]: # type: ignore
        logging.warning("Using placeholder random_blended_config.")
        return {"length": length, "blend_kernels": []}


class SyntheticBenchmarkerV1(Benchmarker):
    """
    Uses BlendedSeriesGeneratorV1 to produce test series and score them using MSE.

    This benchmarker generates multiple time series based on `random_blended_config`
    from the V1 synthetic generators. The model's task is to predict these series.
    """

    def __init__(self, length: int, n_series: int = 10):
        """
        Args:
            length: The length of each synthetic time series to generate.
            n_series: The number of synthetic time series to generate for evaluation.
                      Defaults to 10.
        """
        if length <= 0:
            raise ValueError("Time series length must be positive.")
        if n_series <= 0:
            raise ValueError("Number of series must be positive.")
            
        self.length = length
        self.n_series = n_series

    def prepare_data(self, seed: int) -> Dict[str, np.ndarray]:
        """
        Generates a batch of synthetic time series for evaluation.

        For each series, a configuration is sampled using `random_blended_config`,
        and a `BlendedSeriesGeneratorV1` instance creates the series.
        The "inputs" are simple range arrays (0 to length-1) for each series,
        and "targets" are the generated time series.

        Args:
            seed: An integer seed to ensure reproducibility of data generation.

        Returns:
            A dictionary with "inputs" and "targets" as stacked numpy arrays.
            "inputs" shape: (n_series, length)
            "targets" shape: (n_series, length)
        """
        all_inputs = []
        all_targets = []
        
        for i in range(self.n_series):
            current_seed = seed + i # Vary seed for each series generation
            cfg = random_blended_config(length=self.length, seed=current_seed)
            
            # Assuming BlendedSeriesGeneratorV1 takes the config directly.
            # The prompt mentioned `TimeSeriesGenerator.from_config(cfg)` which might
            # be a factory method on an abstract TimeSeriesGenerator, or it might mean
            # the config directly initializes a specific generator type like BlendedSeriesGeneratorV1.
            gen = BlendedSeriesGeneratorV1(config=cfg)
            
            # Generate should return shape (length, 1), flatten to (length,)
            ts = gen.generate(seed=current_seed).flatten() 
            
            # For many models, inputs might be different (e.g., lagged values).
            # Here, a simple arange is used as a placeholder input if the model expects
            # sequence indices or if it's an auto-regressive model that handles its own input creation.
            # This part might need adjustment based on typical model input formats.
            all_inputs.append(np.arange(self.length, dtype=float)) # Inputs are indices
            all_targets.append(ts)
            
        return {
            "inputs": np.stack(all_inputs), 
            "targets": np.stack(all_targets)
        }

    def evaluate_model(
        self,
        model: Any, # The model should have a .predict(inputs) method
        inputs: np.ndarray, # Expected shape: (n_series, length) or (n_series, length, n_features)
        targets: np.ndarray # Expected shape: (n_series, length)
    ) -> Dict[str, float]:
        """
        Evaluates the given model's predictions against the target series using MSE.

        Args:
            model: The model to evaluate. It must have a `predict` method
                   that accepts the `inputs` array.
            inputs: A numpy array of input data, typically shape (n_series, length).
            targets: A numpy array of target time series, shape (n_series, length).

        Returns:
            A dictionary containing the Mean Squared Error: {"mse": float_value}.
            
        Raises:
            AttributeError: If the model does not have a `predict` method.
            ValueError: If predictions and targets shapes are incompatible for MSE.
        """
        if not hasattr(model, "predict"):
            raise AttributeError("Model does not have a 'predict' method.")
            
        # Assuming model.predict(inputs) returns predictions of the same shape as targets.
        # inputs might need reshaping or feature engineering depending on the model.
        # For example, if model expects (n_samples, n_timesteps, n_features)
        # and inputs is (n_series, length), it might need:
        # inputs_reshaped = inputs.reshape(self.n_series, self.length, 1) # if 1 feature
        # For now, assume model.predict handles the input shape directly.
        predictions = model.predict(inputs)

        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}. "
                "Model's predict method should return outputs compatible with targets."
            )
            
        return {"mse": mse(predictions, targets)}

__all__ = ["SyntheticBenchmarkerV1"]
