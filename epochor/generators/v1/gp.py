"""
Gaussian Process (GP) Time Series Generators.

This module provides tools for generating synthetic time series by sampling
from the prior distribution of a Gaussian Process with a randomly composed kernel.
It leverages scikit-learn's GaussianProcessRegressor for the underlying
computations but adds a layer for stochastic kernel composition and optimized
sampling.
"""
import functools
from typing import Callable, Dict, Optional, Sequence

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, ConstantKernel, DotProduct, ExpSineSquared, Kernel, Matern,
    RationalQuadratic, WhiteKernel
)

# 1) Kernel Bank Construction
def build_kernel_bank(max_length: int) -> Dict[str, list]:
    """
    Creates a dictionary of pre-initialized scikit-learn kernels.
    """
    # Define periodicities relative to the max_length for scale-invariance
    periodicities = [1/60, 1, 24, 168]
    periodic = [ExpSineSquared(periodicity=p / max_length) for p in periodicities]
    
    smooth = [RBF(length_scale=l) for l in (0.1, 1, 10)]
    varying = [RationalQuadratic(alpha=a) for a in (0.1, 1, 10)]
    rough = [Matern(length_scale=l, nu=nu) for nu in (0.5, 1.5, 2.5) for l in (0.1, 1, 10)]
    noise = [WhiteKernel(noise_level=n) for n in (0.1, 1)]
    bias = [ConstantKernel(c) for c in (0.1, 1, 10)]
    trend = [DotProduct(sigma_0=s) for s in (0.0, 1, 10)] # Use sigma_0 instead of s

    return {
        "periodic": periodic, "smooth": smooth, "varying": varying,
        "rough": rough, "noise": noise, "bias": bias, "trend": trend
    }

# 2) Kernel Synthesizer Class
class KernelSynth:
    """
    Synthesizes and samples from complex Gaussian Process kernels.
    """
    DEFAULT_MIN_LENGTH = 256
    DEFAULT_MAX_LENGTH = 1024

    def __init__(
        self,
        min_length: int = DEFAULT_MIN_LENGTH,
        max_length: int = DEFAULT_MAX_LENGTH,
        kernel_bank: Optional[Dict[str, list]] = None,
        random_seed: Optional[int] = None,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.kernel_bank = kernel_bank or build_kernel_bank(self.max_length)
        self.random_seed = random_seed

    def random_binary_map(self, a: Kernel, b: Kernel, rng: np.random.Generator) -> Kernel:
        """Randomly combines two kernels with either addition or multiplication."""
        ops = [lambda x, y: x + y, lambda x, y: x * y]
        return rng.choice(ops)(a, b)

    def sample_kernel_composition(self, rng: np.random.Generator, max_kernels: int) -> Kernel:
        """
        Creates a new kernel by randomly selecting and combining base kernels.
        """
        base_kernels = (
            self.kernel_bank["periodic"] + self.kernel_bank["smooth"] +
            self.kernel_bank["varying"] + self.kernel_bank["noise"]
        )
        all_kernels = base_kernels + self.kernel_bank["bias"] + self.kernel_bank["trend"]
        
        # Start with one base kernel
        selected_kernels = [rng.choice(base_kernels)]
        
        # Add more kernels
        num_additional = rng.integers(1, max_kernels)
        selected_kernels += list(rng.choice(all_kernels, size=num_additional, replace=True))
        
        rng.shuffle(selected_kernels)
        
        # Compose them using random binary operations
        return functools.reduce(lambda a, b: self.random_binary_map(a, b, rng), selected_kernels)

    @staticmethod
    def sample_from_cholesky(L: np.ndarray, rng: np.random.Generator, n_samples: int = 1) -> np.ndarray:
        """
        Optimized sampling using a pre-computed Cholesky factor.
        This is much faster as it avoids re-computing the decomposition.
        """
        z = rng.standard_normal((L.shape[1], n_samples))
        samples = L @ z
        return samples.squeeze()

    def generate_series(
        self, max_kernels: int = 5, seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generates a single time series by sampling a random kernel composition.
        This method is slow if called in a loop because it re-computes the Cholesky on each call.
        """
        rng = np.random.default_rng(seed or self.random_seed)
        length = int(rng.integers(self.min_length, self.max_length + 1))
        X = np.linspace(0, 1, length).reshape(-1, 1)
        
        kernel = self.sample_kernel_composition(rng, max_kernels)
        
        try:
            # This is the slow part: instantiating GPR computes the Cholesky
            gpr = GaussianProcessRegressor(kernel=kernel, random_state=int(rng.integers(0, 2**31 - 1)))
            ts = gpr.sample_y(X, n_samples=1).squeeze()
        except np.linalg.LinAlgError:
            # If kernel is not positive definite, retry with a new random kernel
            return self.generate_series(max_kernels, seed=rng.integers(0, 1e9))
            
        return {"start": np.datetime64("2000-01-01"), "target": ts}

    def generate_many_from_same_kernel(
        self,
        n_series: int,
        max_kernels: int = 5,
        seed: Optional[int] = None,
    ) -> Sequence[Dict[str, np.ndarray]]:
        """
        Efficiently generates multiple time series from the *same* random kernel.
        It computes the Cholesky decomposition once and reuses it.
        """
        rng = np.random.default_rng(seed or self.random_seed)
        length = int(rng.integers(self.min_length, self.max_length + 1))
        X = np.linspace(0, 1, length).reshape(-1, 1)
        
        # 1. Sample a kernel
        kernel = self.sample_kernel_composition(rng, max_kernels)
        
        try:
            # 2. Compute the expensive Cholesky decomposition ONCE
            K = kernel(X)
            # Add jitter for numerical stability
            K[np.diag_indices_from(K)] += 1e-8 
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # If kernel is not positive definite, retry with a new one
            return self.generate_many_from_same_kernel(
                n_series, max_kernels, seed=rng.integers(0, 1e9)
            )

        # 3. Generate N series efficiently using the cached Cholesky factor 'L'
        samples = self.sample_from_cholesky(L, rng, n_samples=n_series)

        # Format output
        series_list = []
        start_time = np.datetime64("2000-01-01")
        for i in range(n_series):
            series_list.append({"start": start_time, "target": samples[:, i]})
            
        return series_list

# 3) Top-level Generator Function
def generate_gaussian_process(length: int, max_kernels: int = 5, seed: Optional[int] = None) -> np.ndarray:
    """
    High-level function to generate a single Gaussian Process time series.
    
    Note: For generating many series, use KernelSynth().generate_many_from_same_kernel()
    for better performance.
    """
    # Force length to be used for min and max length in synth
    ks = KernelSynth(min_length=length, max_length=length, random_seed=seed)
    return ks.generate_series(max_kernels=max_kernels, seed=seed)["target"]

__all__ = [
    "build_kernel_bank",
    "KernelSynth",
    "generate_gaussian_process",
]
