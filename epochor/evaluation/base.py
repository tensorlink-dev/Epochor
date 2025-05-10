"""
Defines the abstract base class for benchmarkers.

A benchmarker provides a standardized interface for preparing evaluation data
and evaluating a model against that data to produce a set of scores.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np # Third-party import

class Benchmarker(ABC):
    """
    Defines interface for a data → model → score pipeline.
    """

    @abstractmethod
    def prepare_data(self, seed: int) -> Dict[str, np.ndarray]:
        """
        Given a seed or round identifier, return a dictionary containing
        the input data and target data for model evaluation.

        The dictionary should have keys "inputs" and "targets", where
        the values are numpy arrays.

        Args:
            seed: An integer seed to ensure reproducibility of data generation.

        Returns:
            A dictionary with "inputs" and "targets" numpy arrays.
        """
        ...

    @abstractmethod
    def evaluate_model(
        self,
        model: Any,
        inputs: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """
        Run model inference on `inputs`, compare to `targets`,
        and return a dictionary of metric scores.

        Args:
            model: The model instance to be evaluated.
            inputs: A numpy array of input data for the model.
            targets: A numpy array of target data for comparison.

        Returns:
            A dictionary where keys are metric names (str) and
            values are their corresponding scores (float).
        """
        ...

    def run(self, model: Any, seed: int) -> Dict[str, float]:
        """
        One‐line API that calls prepare_data → evaluate_model.

        This method orchestrates the data preparation and model evaluation
        steps.

        Args:
            model: The model instance to be evaluated.
            seed: An integer seed for data preparation.

        Returns:
            A dictionary of metric scores from the evaluation.
        """
        data = self.prepare_data(seed)
        # Consider adding validation for data["inputs"] and data["targets"] here
        return self.evaluate_model(model, data["inputs"], data["targets"])

__all__ = ["Benchmarker"]
