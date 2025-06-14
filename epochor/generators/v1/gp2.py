import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    RationalQuadratic,
    WhiteKernel,
    Kernel,
    Matern,
)

import functools
from typing import Optional, Dict

def build_kernel_bank(length: int) -> dict:
    periodicities = [1/60, 5/60, 1, 5, 24, 48, 96]
    periodic = [ExpSineSquared(periodicity=p/length) for p in periodicities]
    smooth   = [RBF(length_scale=l)               for l in (0.1,1.0,10.0)]
    varying  = [RationalQuadratic(alpha=a)         for a in (0.1,1.0,10.0)]
    rough    = [Matern(length_scale=l, nu=nu)     for nu in (0.5,1.5,2.5) for l in (0.1,1.0,10.0)]
    noise    = [WhiteKernel(noise_level=n)        for n in (0.1,1.0)]
    bias     = [ConstantKernel(constant_value=c)  for c in (0.1,1.0,10.0)]
    trend    = [DotProduct(sigma_0=s)             for s in (0.0,1.0,10.0)]
    return {
        "periodic": periodic,
        "smooth":   smooth,
        "varying":  varying,
        "rough":    rough,
        "noise":    noise,
        "bias":     bias,
        "trend":    trend,
    }

class KernelSynth:
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
        ops = [lambda x, y: x + y, lambda x, y: x * y]
        return rng.choice(ops)(a, b)

    def sample_from_gp_prior(
        self,
        kernel: Kernel,
        X: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        X = X[:, None] if X.ndim == 1 else X
        rs = int(rng.integers(0, 2**31 - 1))
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=rs)
        return gpr.sample_y(X, n_samples=1, random_state=rs).squeeze()

    def sample_kernel_composition(
        self,
        rng: np.random.Generator,
        max_kernels: int = 5
    ) -> Kernel:
        base_pool = (
            self.kernel_bank["periodic"] +
            self.kernel_bank["smooth"] +
            self.kernel_bank["varying"] +
            self.kernel_bank["noise"]
        )
        all_kernels = base_pool + self.kernel_bank["bias"] + self.kernel_bank["trend"]

        selected = [rng.choice(base_pool)]
        n_add = rng.integers(1, max_kernels + 1)
        selected += list(rng.choice(all_kernels, size=n_add, replace=True))
        rng.shuffle(selected)
        return functools.reduce(lambda a, b: self.random_binary_map(a, b, rng), selected)

    def generate_time_series(
        self,
        max_kernels: int = 5,
        seed: Optional[int] = None
    ) -> dict:
        rng = np.random.default_rng(seed)
        length = int(rng.integers(self.min_length, self.max_length + 1))
        X = np.linspace(0, 1, length)

        kernel = self.sample_kernel_composition(rng, max_kernels)
        try:
            ts = self.sample_from_gp_prior(kernel, X, rng)
        except np.linalg.LinAlgError:
            return self.generate_time_series(max_kernels, seed=rng.integers(0,1e9))

        return {"start": np.datetime64("2000-01-01"), "target": ts}


def generate_gaussian_process(
    length: int,
    max_kernels: int = 5,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Draw one GP sample of exactly `length` points by constraining
    KernelSynth to [length,length], and return just the array.
    """
    ks = KernelSynth(
        min_length=length,
        max_length=length,
        random_seed=seed
    )
    out = ks.generate_time_series(max_kernels=max_kernels, seed=seed)
    return out["target"]

__all__ = [
    "build_kernel_bank",
    "KernelSynth",
    "generate_gaussian_process",
]