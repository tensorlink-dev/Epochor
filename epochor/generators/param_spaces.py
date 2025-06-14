"""
Parameter spaces for the time series generator functions.

This module defines the valid parameter ranges and choices for each
generator. These specifications are used by the sampler to generate
valid configurations. The keys for the parameter spaces are the Enum
members defined in the registry, ensuring type-safe coupling.
"""

import numpy as np
from .registry import (
    BlendedGeneratorName,
    StatisticalGeneratorName,
    ChaoticGeneratorName,
    MechanisticGeneratorName,
    GaussianGeneratorName
)

# --- 1) Synthetic kernels parameter space ---
synthetic_param_space = {
    BlendedGeneratorName.TREND: {
        "slope": (-0.05, 0.05),
        "intercept": (-5, 5),
        "nonlinear": [True, False]
    },
    BlendedGeneratorName.SEASONALITY: {
        "period": (3, 100),
        "amplitude": (0.1, 2.0),
        "harmonics": (1, 4),
        "phase": (0.0, 2*np.pi)
    },
    BlendedGeneratorName.NOISE: {
        "noise_type": ["gaussian", "uniform", "t", "laplace"],
        "scale": (0.1, 1.0),
        "clip_sigma": (0.1, 2.5)
    },
    BlendedGeneratorName.CHANGEPOINT: {
        "n_breakpoints": (0, 3),
        "max_slope": (0.001, 0.02)
    },
    BlendedGeneratorName.FGN: {"hurst": (0.1, 0.95)},
    BlendedGeneratorName.EVENT_DECAY: {
        "event_locs": "auto",
        "decay_rate": (0.85, 0.99),
        "magnitude": (1.0, 5.0)
    },
    BlendedGeneratorName.TIME_VARYING_FREQ: {
        "base_freq": (0.005, 0.05),
        "freq_slope": (1e-5, 5e-4)
    },
    BlendedGeneratorName.HETEROSKEDASTIC_NOISE: {
        "base_std": (0.1, 1.5),
        "mod_freq": (0.01, 0.1),
        "noise_type": ["gaussian", "t"],
        "clip_sigma": (1.0, 2.0)
    },
    BlendedGeneratorName.AUTOREGRESSIVE: {
        "coeffs": "auto",
        "noise_scale": (0.05, 0.3)
    },
    BlendedGeneratorName.SINUSOID: {
        "period": (5, 200),
        "amplitude": (0.1, 2.0),
        "phase": (0.0, 2*np.pi)
    },
    BlendedGeneratorName.SUM_OF_SINES: {
        "periods": "auto",
        "amps": "auto",
        "phases": "auto"
    },
    BlendedGeneratorName.SAWTOOTH: {"period": (5, 200)},
    BlendedGeneratorName.TRIANGLE: {"period": (5, 200)},
    BlendedGeneratorName.SQUARE: {"period": (5, 200)},
    BlendedGeneratorName.POLYNOMIAL_TREND: {"coeffs": "auto"},
    BlendedGeneratorName.EXPONENTIAL_TREND: {"a": (0.5, 2.0), "b": (0.001, 0.02)},
    BlendedGeneratorName.LOGISTIC_GROWTH: {
        "K": (0.5, 2.0),
        "r": (0.01, 0.3),
        "x0": (0.01, 0.2)
    },
    BlendedGeneratorName.DAMPED_HARMONIC: {
        "omega": (0.01, 0.5),
        "zeta": (0.001, 0.2),
        "x0": (0.5, 2.0),
        "v0": (-1.0, 1.0),
        "dt": (0.1, 1.0)
    },
    BlendedGeneratorName.AMPLITUDE_MODULATED_SINUSOID: {
        "base_amp": (0.1, 2.0),
        "mod_freq": (0.001, 0.1),
        "carrier_period": (10, 200)
    }
}

# --- 2) Statistical processes parameter space ---
statistical_param_space = {
    StatisticalGeneratorName.AR: {"p": (1, 4), "coeffs": "auto", "noise_scale": (0.01, 1.0)},
    StatisticalGeneratorName.MA: {"q": (1, 6), "coeffs": "auto", "noise_scale": (0.01, 1.0)},
    StatisticalGeneratorName.ARMA: {"p": (1, 4), "q": (1, 4), "coeffs": "auto", "noise_scale": (0.01, 1.0)},
    StatisticalGeneratorName.ARIMA: {"p": (1, 4), "d": (0, 1), "q": (1, 4), "coeffs": "auto", "noise_scale": (0.01, 0.5)},
    StatisticalGeneratorName.GARCH: {"p": (1, 2), "q": (1, 2), "omega": (0.001, 0.05), "alpha": (0.2, 0.5), "beta": (0.4, 0.9)},
    StatisticalGeneratorName.OU: {"theta": (0.01, 1.0), "mu": (-5.0, 5.0), "sigma": (0.1, 2.0)},
    StatisticalGeneratorName.JUMP_DIFFUSION: {"mu": (0.0, 0.2), "sigma": (0.1, 1.0), "jump_lambda": (0.05, 0.5), "jump_mu": (0.0, 2.0), "jump_sigma": (0.5, 5.0)},
    StatisticalGeneratorName.HAWKES: {"mu": (0.01, 0.5), "alpha": (0.05, 2.0), "beta": (0.1, 2.0)},
    StatisticalGeneratorName.HMM: {"n_states": (2, 5), "mus": "auto", "sigmas": "auto", "transition": "auto"},
    StatisticalGeneratorName.MSAR: {"n_states": (2, 4), "coeffs": "auto", "noise_scale": (0.01, 1.0)},
    StatisticalGeneratorName.TAR: {"threshold": (-0.2, 0.2), "coeffs1": "auto", "coeffs2": "auto", "noise_scale": (0.01, 0.2)},
    StatisticalGeneratorName.RANDOM_WALK: {"drift": (-1, 1), "sigma": (0.1, 2.0)},
    StatisticalGeneratorName.SEASONAL_RANDOM_WALK: {"season": (2, 24), "sigma": (0.1, 2.0)},
    StatisticalGeneratorName.EGARCH: {"omega": (-1, 1), "alpha": (0, 1), "gamma": (0, 1), "beta": (0, 1)},
    StatisticalGeneratorName.TGARCH: {"omega": (-1, 1), "alpha_pos": (0, 1), "alpha_neg": (0, 1), "beta": (0, 1)},
    StatisticalGeneratorName.LEVY_FLIGHT: {"alpha": (0.5, 2.0), "scale": (0.1, 2.0)},
}

# --- 3) Chaotic attractors parameter space ---
chaotic_param_space = {
    ChaoticGeneratorName.LORENZ: {"dt": (0.005, 0.02), "sigma": (8.0, 14.0), "rho": (25.0, 35.0), "beta": (2.0, 3.0), "noise_scale": (0.0, 0.05)},
    ChaoticGeneratorName.ROSSLER: {"dt": (0.005, 0.02), "a": (0.1, 0.3), "b": (0.1, 0.3), "c": (4.0, 7.0), "noise_scale": (0.0, 0.05)},
    ChaoticGeneratorName.DUFFING: {"dt": (0.005, 0.02), "alpha": (0.5, 1.5), "beta": (-1.5, -0.5), "delta": (0.1, 0.3), "gamma": (0.1, 0.5), "omega": (0.5, 1.5), "noise_scale": (0.0, 0.05)},
    ChaoticGeneratorName.LOGISTIC_MAP: {"r": (3.5, 3.9), "x0": (0.1, 0.9), "burn_in": (50, 200)}
}

# --- 4) Mechanistic models parameter space ---
mechanistic_param_space = {
    MechanisticGeneratorName.PENDULUM: {"g": (9.81, 9.81), "L": (1.0, 1.0), "theta0": (0.01, 0.5), "omega0": (-1.0, 1.0), "dt": (0.01, 0.05)},
    MechanisticGeneratorName.LOTKA_VOLTERRA: {"alpha": (0.1, 1.0), "beta": (0.01, 0.5), "delta": (0.01, 0.5), "gamma": (0.1, 1.0), "x0": (5, 15), "y0": (1, 10), "dt": (0.01, 0.05)},
    MechanisticGeneratorName.SIR: {"beta": (0.1, 0.5), "gamma": (0.05, 0.2), "S0": (0.5, 1.0), "I0": (0.01, 0.5), "R0": (0.0, 0.1), "dt": (0.1, 1.0)},
    MechanisticGeneratorName.HEAT_EQUATION: {"n_space": (5, 20), "alpha": (0.01, 0.5), "dt": (0.01, 0.1), "dx": (0.5, 2.0)},
    MechanisticGeneratorName.LOCAL_LINEAR_TREND: {"sigma_level": (0.1, 2.0), "sigma_trend": (0.01, 0.5)},
    MechanisticGeneratorName.HOLT_WINTERS: {"alpha": (0.1, 1.0), "beta": (0.01, 0.5), "gamma": (0.01, 0.5), "season": (2, 24), "seasonal": ["add", "mul"]},
}

# --- 5) Gaussian-process parameter space ---
gaussian_param_space = {
    GaussianGeneratorName.GAUSSIAN_PROCESS: {"max_kernels": (1, 5)}
}

# --- 6) Merge into a global parameter space ---
global_param_space = {}
global_param_space.update(synthetic_param_space)
global_param_space.update(statistical_param_space)
global_param_space.update(chaotic_param_space)
global_param_space.update(mechanistic_param_space)
global_param_space.update(gaussian_param_space)

__all__ = [
    "synthetic_param_space", "statistical_param_space",
    "chaotic_param_space", "mechanistic_param_space",
    "gaussian_param_space", "global_param_space"
]
