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



__all__ = ["Benchmarker"]
