# epochor/generators/base.py

import inspect
import random
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import numpy as np

from .registry import (
    blended_registry,
    statistical_registry,
    chaotic_registry,
    mechanistic_registry,
    gaussian_registry,
)

def _call_filtered(fn: Callable[..., np.ndarray],
                   length: int,
                   params: Dict[str, Any]) -> np.ndarray:
    """
    Call `fn` with only those items from `params` that match its signature.
    Ensures 'length' is passed if required by `fn`.
    """
    sig = inspect.signature(fn)
    filtered = {k: v for k, v in params.items() if k in sig.parameters}
    if 'length' in sig.parameters and 'length' not in filtered:
        filtered['length'] = length
    return fn(**filtered)


class TimeSeriesGenerator(ABC):
    """
    Abstract interface for all time‐series generators.

    A subclass must implement:
      - sample_config: draw a reproducible configuration dict
      - generate_series: produce a 1D array of length `self.length`
    """

    def __init__(self, length: int):
        """
        Args:
          length: desired length of the output series
        """
        self.length = length

    @abstractmethod
    def sample_config(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Returns a dict of parameters to drive series generation.
        If `seed` is given, use it for reproducible sampling.
        """
        ...

    @abstractmethod
    def generate_series(self,
                        config: Dict[str, Any],
                        seed: Optional[int] = None
                       ) -> np.ndarray:
        """
        Given a config from sample_config, return a 1D numpy array
        of shape (length,) containing the generated series.
        """
        ...

    def generate(self, seed: Optional[int] = None) -> np.ndarray:
        """
        End‐to‐end API: sample config, optionally reseed RNGs,
        generate the series, and return it as shape (length, 1).
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        cfg = self.sample_config(seed=seed)
        ts = self.generate_series(cfg, seed=seed)
        return ts.reshape(-1, 1)


class BlendedSeriesGeneratorV1(TimeSeriesGenerator):
    """
    V1 synthetic generator that blends multiple primitive kernels
    according to a sampled 'blend_kernels' config.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
          config: dict containing 'length' and 'blend_kernels' list
        """
        super().__init__(length=config["length"])
        self.config = config

    def sample_config(self, seed: Optional[int] = None) -> Dict[str, Any]:
        # In V1, config is provided at init; ignore seed here.
        return self.config

    def generate_series(self,
                        config: Dict[str, Any],
                        seed: Optional[int] = None
                       ) -> np.ndarray:
        """
        Iterate over config['blend_kernels'], call appropriate kernel
        functions from blended_registry, and weight‐sum them.
        """
        length = config["length"]
        out = np.zeros(length)
        for kd in config["blend_kernels"]:
            fn = blended_registry[kd["name"]]
            frag = _call_filtered(fn, length, kd["params"])
            out += kd["weight"] * frag
        return out


class SingleKernelSeriesGenerator(TimeSeriesGenerator):
    """
    Generator for a single kernel drawn from a registry (e.g., AR, GARCH, Lorenz).
    """

    def __init__(self,
                 name: str,
                 length: int,
                 params: Dict[str, Any],
                 registry: Dict[str, Callable[..., np.ndarray]]
                ):
        """
        Args:
          name: key in the chosen registry
          length: desired series length
          params: parameters for the kernel function
          registry: mapping name → kernel function
        """
        super().__init__(length=length)
        self.name = name
        self.params = params
        self.registry = registry

    def sample_config(self, seed: Optional[int] = None) -> Dict[str, Any]:
        # Config for a single kernel is just its name + params
        return {"name": self.name, "params": self.params}

    def generate_series(self,
                        config: Dict[str, Any],
                        seed: Optional[int] = None
                       ) -> np.ndarray:
        """
        Call the selected kernel function with its params.
        """
        fn = self.registry[config["name"]]
        return _call_filtered(fn, self.length, config["params"])


class CategoryMixtureGenerator(TimeSeriesGenerator):
    """
    Meta-generator that chooses among multiple generator categories
    (blended, statistical, chaotic, mechanistic, gaussian) by weight.
    """

    def __init__(self,
                 samplers: Sequence[Callable[..., Dict[str, Any]]],
                 registries: Sequence[Dict[str, Callable[..., np.ndarray]]],
                 length: int = 500,
                 weights: Optional[Sequence[float]] = None
                ):
        """
        Args:
          samplers: functions like random_blended_config, random_statistical_config, ...
          registries: matching registries [blended_registry, statistical_registry, ...]
          length: default series length
          weights: optional probabilities for each category (must sum to 1)
        """
        super().__init__(length=length)
        if len(samplers) != len(registries):
            raise ValueError("samplers and registries must align")
        self.categories = list(zip(samplers, registries))
        if weights is None:
            self.weights = np.ones(len(self.categories)) / len(self.categories)
        else:
            w = np.array(weights, float)
            if not np.isclose(w.sum(), 1.0):
                w = w / w.sum()
            self.weights = w

    def sample_config(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Chooses a category by self.weights, invokes its sampler to get a config dict.
        """
        idx = np.random.choice(len(self.categories), p=self.weights)
        sampler, _ = self.categories[idx]
        return sampler(length=self.length, seed=seed)

    def generate_series(self,
                        config: Dict[str, Any],
                        seed: Optional[int] = None
                       ) -> np.ndarray:
        """
        Dispatches to either BlendedSeriesGeneratorV1 or SingleKernelSeriesGenerator
        based on config contents.
        """
        # Determine category by presence of 'blend_kernels'
        if "blend_kernels" in config:
            gen = BlendedSeriesGeneratorV1(config)
        else:
            # single‐kernel config must contain "name" and "params"
            name = config["name"]
            # find registry for this name
            for reg in (blended_registry,
                        statistical_registry,
                        chaotic_registry,
                        mechanistic_registry,
                        gaussian_registry):
                if name in reg:
                    registry = reg
                    break
            else:
                raise KeyError(f"Unknown kernel '{name}' in config")
            gen = SingleKernelSeriesGenerator(name, self.length, config["params"], registry)
        return gen.generate(seed=seed)


__all__ = [
    "TimeSeriesGenerator",
    "BlendedSeriesGeneratorV1",
    "SingleKernelSeriesGenerator",
    "CategoryMixtureGenerator",
    "_call_filtered",
]
