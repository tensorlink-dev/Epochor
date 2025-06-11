"""
Benchmarker implementation for V1 synthetic time series data.

This module defines a concrete Benchmarker that uses V1 synthetic data generators
to create evaluation datasets. Each generated series is split into an input
portion and a target portion. The benchmarker prepares batches of these
input/target pairs with appropriate padding and attention masks for evaluation
with transformer-style models. It primarily computes Mean Squared Error (MSE).
"""
import random
from typing import Dict, Any, List # Added List

import numpy as np
import torch # Added torch for tensor operations

# Local application/library specific imports
from .base import Benchmarker
try:
    from epochor.generators.base import Benchmarker
    from epochor.generators.sampler import random_blended_config
    from epochor.generators.v1 import BlendedSeriesGeneratorV1
except ImportError as e:
    import logging
    logging.error(
        "Failed to import BlendedSeriesGeneratorV1 or random_blended_config "
        f"from epochor.generators: {e}. SyntheticBenchmarkerV1 will not be fully functional."
    )


class SyntheticBenchmarkerV1(Benchmarker):
    """
    Generates synthetic time series, splits them into input/target pairs,
    prepares padded batches suitable for transformer models, and scores predictions
    using MSE.
    """

    def __init__(self, 
                 length: int, 
                 n_series: int = 10, 
                 min_input_frac: float = 0.5, 
                 max_input_frac: float = 0.95,
                 padding_value: float = 0.0): # Added padding_value
        """
        Args:
            length: Total length of each full synthetic time series.
            n_series: Number of series to generate for evaluation.
            min_input_frac: Min fraction of series to use as input (0 to 1).
            max_input_frac: Max fraction of series to use as input (0 to 1).
            padding_value: Value to use for padding input and target sequences.
        """
        if length <= 1: # Adjusted: length must be > 1 to allow for a split
            raise ValueError("Time series length must be greater than 1 to allow for input/target split.")
        if n_series <= 0:
            raise ValueError("Number of series must be positive.")
        if not (0 < min_input_frac < 1 and 0 < max_input_frac < 1): # Fractions must be strictly between 0 and 1
            raise ValueError("Input fractions must be strictly between 0 and 1.")
        if min_input_frac >= max_input_frac:
            raise ValueError("min_input_frac must be less than max_input_frac.")
        
        self.length = length
        self.n_series = n_series
        self.min_input_frac = min_input_frac
        self.max_input_frac = max_input_frac
        self.padding_value = padding_value


    def prepare_data(self, seed: int) -> Dict[str, Any]:
        """
        Generates a batch of synthetic time series, splits each into input and target,
        and prepares padded batches with attention masks.

        Args:
            seed: An integer seed for reproducibility.

        Returns:
            A dictionary containing:
            - "inputs_padded": torch.Tensor of shape (n_series, max_input_len).
            - "attention_mask": torch.Tensor of shape (n_series, max_input_len).
            - "targets_padded": torch.Tensor of shape (n_series, max_target_len).
            - "actual_target_lengths": List[int] of actual lengths for each target series.
        """
        raw_inputs: List[np.ndarray] = []
        raw_targets: List[np.ndarray] = []
        
        rng = random.Random(seed) # For reproducible split point selection

        for i in range(self.n_series):
            current_series_seed = seed + i
            cfg = random_blended_config(length=self.length, seed=current_series_seed)
            gen = BlendedSeriesGeneratorV1(config=cfg)
            full_ts = gen.generate(seed=current_series_seed).flatten().astype(np.float32)

            input_frac = rng.uniform(self.min_input_frac, self.max_input_frac)
            split_idx = int(self.length * input_frac)
            
            # Ensure at least 1 point for input and 1 for target
            split_idx = max(1, min(split_idx, self.length - 1))

            ts_input = full_ts[:split_idx]
            ts_target = full_ts[split_idx:]
            
            raw_inputs.append(ts_input)
            raw_targets.append(ts_target)
            
        # Determine max lengths for padding
        max_input_len = 0
        if raw_inputs: # Check if raw_inputs is not empty
             max_input_len = max(len(s) for s in raw_inputs) if raw_inputs else 0
        
        max_target_len = 0
        if raw_targets: # Check if raw_targets is not empty
            max_target_len = max(len(s) for s in raw_targets) if raw_targets else 0


        # Initialize padded tensors and attention mask
        inputs_padded = torch.full((self.n_series, max_input_len), self.padding_value, dtype=torch.float32)
        attention_mask = torch.zeros((self.n_series, max_input_len), dtype=torch.long) 
        targets_padded = torch.full((self.n_series, max_target_len), self.padding_value, dtype=torch.float32)
        actual_target_lengths: List[int] = []

        for i in range(self.n_series):
            inp = raw_inputs[i]
            tar = raw_targets[i]
            
            input_len = len(inp)
            inputs_padded[i, :input_len] = torch.from_numpy(inp)
            attention_mask[i, :input_len] = 1 
            
            target_len = len(tar)
            targets_padded[i, :target_len] = torch.from_numpy(tar)
            actual_target_lengths.append(target_len)
            
        return {
            "inputs_padded": inputs_padded,
            "attention_mask": attention_mask,
            "targets_padded": targets_padded,
            "actual_target_lengths": actual_target_lengths,
        }

   
