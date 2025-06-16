import numpy as np
import functools
from typing import Dict, Optional, List
from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic, ExpSineSquared,
    WhiteKernel, ConstantKernel, DotProduct, Kernel
)
from joblib import Parallel, delayed
from tqdm.auto import tqdm


# === Custom Kernels ===

class ChangePointKernel(Kernel):
    def __init__(self, k1: Kernel, k2: Kernel, location=0.5, steepness=10.0):
        self.k1, self.k2 = k1, k2
        self.location = location
        self.steepness = steepness

    def __call__(self, X, Y=None, eval_gradient=False):
        sigmoid = lambda x: 1 / (1 + np.exp(-self.steepness * (x - self.location)))
        Xs = sigmoid(X.ravel())
        Ys = Xs if Y is None else sigmoid(Y.ravel())
        gate = np.outer(Xs, Ys)
        return self.k1(X, Y) * gate + self.k2(X, Y) * (1 - gate)

    def diag(self, X): return np.diag(self(X))
    def is_stationary(self): return False
    def __str__(self): return f"ChangePoint({self.k1}, {self.k2}, loc={self.location:.2f})"
    __repr__ = __str__


class LocallyPeriodicKernel(Kernel):
    def __init__(self, length_scale=1.0, periodicity=1.0):
        self.length_scale = length_scale
        self.periodicity = periodicity
        self.k = ExpSineSquared(length_scale=length_scale, periodicity=periodicity) * RBF(length_scale=length_scale)

    def __call__(self, X, Y=None, eval_gradient=False): return self.k(X, Y)
    def diag(self, X): return np.diag(self(X))
    def is_stationary(self): return True
    def get_params(self, deep=True): return {"length_scale": self.length_scale, "periodicity": self.periodicity}
    def __str__(self): return f"LocallyPeriodic(length_scale={self.length_scale:.2f}, periodicity={self.periodicity:.2f})"
    __repr__ = __str__


class PolynomialKernel(Kernel):
    def __init__(self, degree=2): self.degree = degree
    def __call__(self, X, Y=None, eval_gradient=False): return (1 + np.dot(X, (X if Y is None else Y).T)) ** self.degree
    def diag(self, X): return np.diag(self(X))
    def is_stationary(self): return True
    def __str__(self): return f"Polynomial(degree={self.degree})"
    __repr__ = __str__


class ArcCosineKernel(Kernel):
    def __init__(self, order=1): self.order = order
    def __call__(self, X, Y=None, eval_gradient=False):
        X, Y = np.atleast_2d(X), np.atleast_2d(X if Y is None else Y)
        norm_X, norm_Y = np.linalg.norm(X, axis=1, keepdims=True), np.linalg.norm(Y, axis=1, keepdims=True)
        dot = np.dot(X, Y.T)
        theta = np.arccos(np.clip(dot / (norm_X @ norm_Y.T + 1e-6), -1, 1))
        return (1 / np.pi) * (np.sin(theta) + (np.pi - theta) * np.cos(theta))
    def diag(self, X): return np.diag(self(X))
    def is_stationary(self): return True
    def __str__(self): return f"ArcCosine(order={self.order})"
    __repr__ = __str__


# === Kernel Bank

def build_kernel_bank(max_length: int, rng: Optional[np.random.Generator] = None) -> Dict[str, list]:
    rng = rng or np.random.default_rng()

    # Sampling helpers
    sample_log = lambda mu, sigma: rng.lognormal(mean=mu, sigma=sigma)
    sample_uniform = lambda a, b: rng.uniform(a, b)
    sample_choice = lambda lst: rng.choice(lst)
    sample_loguniform = lambda low, high: 10 ** rng.uniform(np.log10(low), np.log10(high))  # Log-uniform

    # === Basic components ===
    periodic = [
        ExpSineSquared(
            length_scale=sample_log(-0.25, 0.75),
            periodicity=sample_loguniform(0.01, 1.0)  # log-scale: ~hour to year in normalized [0,1]
        ) for _ in range(5)
    ]

    smooth = [RBF(length_scale=sample_log(-0.25, 0.75)) for _ in range(4)]
    varying = [RationalQuadratic(alpha=sample_log(0, 1.0)) for _ in range(3)]
    rough = [
        Matern(length_scale=sample_log(-0.25, 0.75), nu=sample_choice([0.5, 1.5, 2.5]))
        for _ in range(4)
    ]

    noise = [WhiteKernel(noise_level=sample_log(-2.5, 0.5)) for _ in range(2)]
    bias = [ConstantKernel(constant_value=sample_log(-0.5, 0.5)) for _ in range(3)]
    trend = [DotProduct(sigma_0=np.clip(sample_log(0, 0.5), 0.01, 5.0)) for _ in range(3)]

    # === Composite / advanced ===
    changepoint = [
        ChangePointKernel(
            k1=RBF(length_scale=sample_log(-0.25, 0.75)),
            k2=ExpSineSquared(
                length_scale=sample_log(-0.25, 0.75),
                periodicity=sample_loguniform(0.01, 1.0)
            ),
            location=sample_uniform(0.2, 0.8),
            steepness=min(sample_log(1.5, 0.5), 50.0),  # cap steepness
        ) for _ in range(3)
    ]

    local_periodic = [
        LocallyPeriodicKernel(
            length_scale=sample_log(-0.25, 0.75),
            periodicity=sample_loguniform(0.01, 1.0)
        ) for _ in range(3)
    ]

    polynomial = [PolynomialKernel(degree=int(sample_choice([1, 2, 3, 4]))) for _ in range(2)]
    arccosine = [ArcCosineKernel(order=int(sample_choice([0, 1, 2]))) for _ in range(2)]

    spectral = [
        ExpSineSquared(periodicity=1 / sample_loguniform(1, 10)) *
        RBF(length_scale=sample_log(-0.25, 0.75))
        for _ in range(3)
    ]

    return {
        "periodic": periodic,
        "smooth": smooth,
        "varying": varying,
        "rough": rough,
        "noise": noise,
        "bias": bias,
        "trend": trend,
        "changepoint": changepoint,
        "local_periodic": local_periodic,
        "polynomial": polynomial,
        "arccosine": arccosine,
        "spectral": spectral,
    }


# === KernelSynth
# based on the original kernel synth: https://github.com/amazon-science/chronos-forecasting
# paper: 
class KernelSynth:
    def __init__(self, min_length: int, max_length: int, random_seed: Optional[int] = None):
        self.min_length = min_length
        self.max_length = max_length
        self.random_seed = random_seed
        self.master_seed_seq = np.random.SeedSequence(random_seed)

    def build_rngs_and_bank(self, seed: int):
        seed_seq = np.random.SeedSequence(seed)
        kernel_seed, length_seed, sample_seed = seed_seq.spawn(3)
        kernel_rng = np.random.default_rng(kernel_seed.generate_state(1)[0])
        length_rng = np.random.default_rng(length_seed.generate_state(1)[0])
        sample_rng = np.random.default_rng(sample_seed.generate_state(1)[0])
        bank = build_kernel_bank(self.max_length, kernel_rng)
        return kernel_rng, length_rng, sample_rng, bank

    def sample_kernel_composition(self, rng: np.random.Generator, bank: Dict[str, list], max_kernels: int) -> Kernel:
        base = bank["periodic"] + bank["smooth"] + bank["varying"] + bank["noise"]
        extra = sum([bank[k] for k in ("bias", "trend", "changepoint", "local_periodic", "polynomial", "arccosine", "spectral")], [])
        sel = [rng.choice(base)]
        sel += list(rng.choice(extra + base, size=rng.integers(1, max_kernels + 1), replace=True))
        rng.shuffle(sel)
        return functools.reduce(lambda a, b: rng.choice([lambda x, y: x + y, lambda x, y: x * y])(a, b), sel)

    def sample_from_gp_prior(
        self, kernel: Kernel, X: np.ndarray, rng: np.random.Generator,
        samples_per_kernel: int = 1, alpha: float = 1e-6
    ) -> np.ndarray:
        X2 = X[:, None]
        K = kernel(X2, X2)
        K += alpha * np.eye(len(K))
        try:
            L = np.linalg.cholesky((K + K.T) / 2)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError("Cholesky failed: kernel matrix not PSD")
        z = rng.standard_normal(size=(len(K), samples_per_kernel))
        return L @ z  # (T, S)

    def _generate_with_seed(self, seed: int, max_kernels: int, samples_per_kernel: int) -> List[Dict[str, np.ndarray]]:
        kernel_rng, length_rng, sample_rng, bank = self.build_rngs_and_bank(seed)
        length = int(length_rng.integers(self.min_length, self.max_length + 1))
        X = np.linspace(0, 1, length)
        kernel = self.sample_kernel_composition(kernel_rng, bank, max_kernels)

        try:
            samples = self.sample_from_gp_prior(kernel, X, sample_rng, samples_per_kernel=samples_per_kernel)
        except np.linalg.LinAlgError:
            return self._generate_with_seed(int(sample_rng.integers(0, 1e9)), max_kernels, samples_per_kernel)

        return [
            {
                "start": np.datetime64("2000-01-01"),
                "target": samples[:, i],
                "kernel": str(kernel)
            }
            for i in range(samples.shape[1])
        ]

    def generate_dataset(
        self, num_series: int, max_kernels: int = 5, samples_per_kernel: int = 1, n_jobs: int = 2
    ) -> List[dict]:
        n_kernels = (num_series + samples_per_kernel - 1) // samples_per_kernel
        seeds = self.master_seed_seq.spawn(n_kernels)

        batches = Parallel(n_jobs=n_jobs)(
            delayed(self._generate_with_seed)(int(seed.generate_state(1)[0]), max_kernels, samples_per_kernel)
            for seed in tqdm(seeds, desc="Generating synthetic series")
        )

        return [item for batch in batches for item in batch][:num_series]
