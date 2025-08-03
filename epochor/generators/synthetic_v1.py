
import random
from typing import Dict, Any, List

import numpy as np
import torch

from epochor.generators.base import Benchmarker
from epochor.generators.kernel_synth import KernelSynth


class SyntheticBenchmarkerV1(Benchmarker):
    """
    Generates synthetic time series using KernelSynth, splits them into input/target pairs,
    prepares padded batches suitable for transformer models.
    """

    def __init__(self,
                 length: int,
                 n_series: int = 10,
                 min_input_frac: float = 0.5,
                 max_input_frac: float = 0.95,
                 padding_value: float = 0.0):
        """
        Args:
            length: Total length of each full synthetic time series.
            n_series: Number of series to generate for evaluation.
            min_input_frac: Min fraction of series to use as input (0 to 1).
            max_input_frac: Max fraction of series to use as input (0 to 1).
            padding_value: Value to use for padding input and target sequences.
        """
        if length <= 1:
            raise ValueError("Time series length must be greater than 1 to allow for input/target split.")
        if n_series <= 0:
            raise ValueError("Number of series must be positive.")
        if not (0 < min_input_frac < 1 and 0 < max_input_frac < 1):
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
            - "kernels": List[str] of kernel descriptions for each series.
        """
        raw_inputs: List[np.ndarray] = []
        raw_targets: List[np.ndarray] = []
        kernels: List[str] = []

        rng = random.Random(seed)

        ks = KernelSynth(min_length=self.length, max_length=self.length, random_seed=seed)
        synthetics = ks.generate_dataset(num_series=self.n_series, max_kernels=5, samples_per_kernel=50)
        
        for synthetic_series in synthetics:
            full_ts = synthetic_series['target'].flatten().astype(np.float32)
            kernel = synthetic_series['kernel']

            input_frac = rng.uniform(self.min_input_frac, self.max_input_frac)
            split_idx = int(self.length * input_frac)

            # Ensure at least 1 point for input and 1 for target
            split_idx = max(1, min(split_idx, self.length - 1))

            ts_input = full_ts[:split_idx]
            ts_target = full_ts[split_idx:]

            raw_inputs.append(ts_input)
            raw_targets.append(ts_target)
            kernels.append(kernel)

        # Determine max lengths for padding
        max_input_len = max(len(s) for s in raw_inputs) if raw_inputs else 0
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
            "kernels": kernels
        }
