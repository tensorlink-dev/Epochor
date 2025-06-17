import numpy as np
import random
import inspect
from typing import Callable, Sequence, Optional, Dict, Any, Union, Tuple

# Import Enums and registries
from epochor.generators.registry import (
    blended_registry,
    statistical_registry,
    chaotic_registry,
    mechanistic_registry,
    gaussian_registry,
    BlendedGeneratorName,
    StatisticalGeneratorName,
    ChaoticGeneratorName,
    MechanisticGeneratorName,
    GaussianGeneratorName,
)

# Import parameter spaces which now use Enums as keys
from epochor.generators.param_spaces import global_param_space, synthetic_param_space

# Define GeneratorNameType for type hinting
GeneratorNameType = Union[
    BlendedGeneratorName,
    StatisticalGeneratorName,
    ChaoticGeneratorName,
    MechanisticGeneratorName,
    GaussianGeneratorName,
]

# Update SEASONAL_KERNS to use Enum members
SEASONAL_KERNS = {
    BlendedGeneratorName.SEASONALITY,
    BlendedGeneratorName.SINUSOID,
    BlendedGeneratorName.SUM_OF_SINES,
    BlendedGeneratorName.SAWTOOTH,
    BlendedGeneratorName.TRIANGLE,
    BlendedGeneratorName.SQUARE,
}

# Helper function to call a generator with filtered parameters
def _call_filtered(func: Callable, length: int, params: Dict[str, Any]) -> np.ndarray:
    """Calls a generator function with only the parameters it accepts."""
    sig = inspect.signature(func)
    filtered_params = {k: v for k, v in params.items() if k in sig.parameters}
    return func(length=length, **filtered_params)

# --- Parameter Sampling ---

def sample_params(name: GeneratorNameType, length: int) -> Dict[str, Any]:
    """
    Samples parameters for a given generator by looking up its spec
    in the global_param_space using its Enum member.
    """
    if name not in global_param_space:
        raise KeyError(f"No parameter space defined for '{name}'")
    spec = global_param_space[name]
    cfg = {}

    for k, v in spec.items():
        # 1) Special handling for sum_of_sines
        if name == BlendedGeneratorName.SUM_OF_SINES and k in ("periods", "amps", "phases"):
            n = random.randint(1, 3)
            if k == "periods":
                lo, hi = synthetic_param_space[BlendedGeneratorName.SINUSOID]["period"]
                cfg[k] = [round(random.uniform(lo, hi), 2) for _ in range(n)]
            elif k == "amps":
                lo, hi = synthetic_param_space[BlendedGeneratorName.SINUSOID]["amplitude"]
                cfg[k] = [round(random.uniform(lo, hi), 2) for _ in range(n)]
            else:  # phases
                cfg[k] = [round(random.uniform(0, 2 * np.pi), 2) for _ in range(n)]
            continue

        # 2) "auto" parameter generation
        if v == "auto":
            if k == "event_locs":
                cfg[k] = sorted(random.sample(range(1, length), 2))
            elif name == StatisticalGeneratorName.HMM:
                if k == "mus":
                    cfg[k] = [round(random.uniform(-1, 1), 2) for _ in range(cfg.get("n_states", 3))]
                elif k == "sigmas":
                    cfg[k] = [round(random.uniform(0.1, 2.0), 2) for _ in range(cfg.get("n_states", 3))]
                elif k == "transition":
                    n_states = cfg.get("n_states", 3)
                    mat = np.random.rand(n_states, n_states)
                    cfg[k] = mat / mat.sum(axis=1, keepdims=True)
            elif k in ("coeffs", "coeffs1", "coeffs2"):
                order = max(cfg.get("p", 1), cfg.get("q", 1), 1)
                cfg[k] = [round(random.uniform(-0.9, 0.9), 3) for _ in range(order)]
            else:
                raise ValueError(f"Cannot auto-sample '{k}' for '{name.value}'")
            continue

        # 3) Integer ranges
        if isinstance(v, tuple) and all(isinstance(z, int) for z in v):
            cfg[k] = random.randint(v[0], v[1])
            continue

        # 4) Continuous ranges
        if isinstance(v, tuple):
            cfg[k] = round(random.uniform(v[0], v[1]), 4)
            continue

        # 5) Categorical choices
        cfg[k] = random.choice(v)

    # Post-fix for GARCH stationarity
    if name == StatisticalGeneratorName.GARCH:
        alpha, beta = cfg.get("alpha", 0), cfg.get("beta", 0)
        if alpha + beta >= 0.99:
            cfg["beta"] = round(0.99 - alpha, 4)

    return cfg

# --- Configuration Samplers ---

def random_blended_config(
    length: int = 500,
    seed: Optional[int] = None,
    min_k: int = 1,
    max_k: int = 5,
    peak_k: int = 3,
    spread: float = 1.0,
    force_seasonal: bool = False
) -> Dict[str, Any]:
    """Generates a configuration for a blended time series."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    k = int(round(np.random.normal(peak_k, spread)))
    k = max(min_k, min(max_k, k))

    blend = [{
        "name": BlendedGeneratorName.NOISE,
        "weight": random.uniform(0.2, 1.0),
        "params": sample_params(BlendedGeneratorName.NOISE, length)
    }]

    pool = [nm for nm in blended_registry.keys() if nm != BlendedGeneratorName.NOISE]
    n_to_add = k - 1

    if force_seasonal and n_to_add > 0:
        seasonal_choices = list(SEASONAL_KERNS.intersection(pool))
        if seasonal_choices:
            pick = random.choice(seasonal_choices)
            blend.append({
                "name": pick,
                "weight": random.uniform(0.2, 1.0),
                "params": sample_params(pick, length)
            })
            pool.remove(pick)
            n_to_add -= 1

    if n_to_add > 0:
        for nm in random.sample(pool, min(n_to_add, len(pool))):
            blend.append({
                "name": nm,
                "weight": random.uniform(0.2, 1.0),
                "params": sample_params(nm, length)
            })

    total_w = sum(item["weight"] for item in blend)
    for item in blend:
        item["weight"] /= total_w

    return {"length": length, "blend_kernels": blend}

def random_statistical_config(length: int = 500, seed: Optional[int] = None) -> Dict[str, Any]:
    """Generates a random configuration for a statistical model."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    name = random.choice(list(statistical_registry.keys()))
    return {"name": name, "length": length, "params": sample_params(name, length)}

def random_chaotic_config(length: int = 500, seed: Optional[int] = None) -> Dict[str, Any]:
    """Generates a random configuration for a chaotic model."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    name = random.choice(list(chaotic_registry.keys()))
    return {"name": name, "length": length, "params": sample_params(name, length)}

def random_mechanistic_config(length: int = 500, seed: Optional[int] = None) -> Dict[str, Any]:
    """Generates a random configuration for a mechanistic model."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    name = random.choice(list(mechanistic_registry.keys()))
    return {"name": name, "length": length, "params": sample_params(name, length)}

def random_gaussian_config(length: int = 500, seed: Optional[int] = None) -> Dict[str, Any]:
    """Generates a configuration for a Gaussian Process model."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    name = GaussianGeneratorName.GAUSSIAN_PROCESS
    return {"name": name, "length": length, "params": sample_params(name, length)}

# --- Generator Classes ---

class SyntheticGenerator:
    """Generates a time series by blending multiple weighted kernels."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def generate(self, seed: Optional[int] = None) -> np.ndarray:
        L = self.config["length"]
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        out = np.zeros(L)
        for kd in self.config["blend_kernels"]:
            fn = blended_registry[kd["name"]]
            fragment = _call_filtered(fn, L, kd["params"])
            out += kd["weight"] * fragment.squeeze()
        return out[:, None]


class UnivariateGenerator:
    """Generates a time series from a single, non-blended model."""
    def __init__(self, name: GeneratorNameType, length: int, params: Dict, registry: Dict):
        self.name = name
        self.length = length
        self.params = params
        self.registry = registry

    def generate(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        fn = self.registry[self.name]
        ts = _call_filtered(fn, self.length, self.params)
        return ts if ts.ndim > 1 else ts[:, None]

# --- Master Generator ---

SAMPLER_CATEGORIES = [
    ("blended", random_blended_config, blended_registry),
    ("statistical", random_statistical_config, statistical_registry),
    ("chaotic", random_chaotic_config, chaotic_registry),
    ("mechanistic", random_mechanistic_config, mechanistic_registry),
    ("gaussian_process", random_gaussian_config, gaussian_registry),
]

class CombinedGenerator:
    """
    A master generator that can produce a time series from any of the defined
    categories (blended, statistical, etc.) based on a weighted choice.
    """
    def __init__(self,
                 length: int = 500,
                 categories: Optional[Sequence[Tuple[str, Callable, Dict]]] = None,
                 weights: Optional[Sequence[float]] = None):
        self.length = length
        self.categories = categories if categories is not None else SAMPLER_CATEGORIES

        if weights is None:
            self.weights = np.ones(len(self.categories)) / len(self.categories)
        else:
            if len(weights) != len(self.categories):
                raise ValueError("Length of weights must match the number of generator categories.")
            w = np.array(weights, dtype=float)
            if not np.isclose(w.sum(), 1.0):
                w = w / w.sum()
            self.weights = w

    def generate(self, seed: Optional[int] = None) -> (np.ndarray, str):
        """
        Generates a time series and returns it along with a descriptive tag.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        cat_idx = np.random.choice(len(self.categories), p=self.weights)
        cat_name, sampler, registry = self.categories[cat_idx]

        config = sampler(length=self.length, seed=seed)

        if cat_name == "blended":
            ts = SyntheticGenerator(config).generate(seed=seed)
            tag = "blended"
        else:
            name = config["name"]
            params = config["params"]
            ts = UnivariateGenerator(name, self.length, params, registry).generate(seed=seed)
            tag = f"{cat_name}({name.value})"

        return ts, tag


__all__ = [
    "sample_params",
    "random_blended_config",
    "random_statistical_config",
    "random_chaotic_config",
    "random_mechanistic_config",
    "random_gaussian_config",
    "SyntheticGenerator",
    "UnivariateGenerator",
    "CombinedGenerator",
    "GeneratorNameType",
    "SAMPLER_CATEGORIES",
]
