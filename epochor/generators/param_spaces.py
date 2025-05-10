import numpy as np

# ─── 1) Synthetic kernels parameter space ───
synthetic_param_space = {
    "trend": {
        "slope": (-0.05, 0.05),
        "intercept": (-5, 5),
        "nonlinear": [True, False]
    },
    "seasonality": {
        "period": (3, 100),
        "amplitude": (0.1, 2.0),
        "harmonics": (1, 4),
        "phase": (0.0, 2*np.pi)
    },
    "noise": {
        "noise_type": ["gaussian", "uniform", "t", "laplace"],
        "scale": (0.1, 1.0),
        "clip_sigma": (0.1, 2.5)
    },
    "changepoint": {
        "n_breakpoints": (0, 3),
        "max_slope": (0.001, 0.02)
    },
    "fgn": {"hurst": (0.1, 0.95)},
    "event_decay": {
        "event_locs": "auto",
        "decay_rate": (0.85, 0.99),
        "magnitude": (1.0, 5.0)
    },
    "time_varying_freq": {
        "base_freq": (0.005, 0.05),
        "freq_slope": (1e-5, 5e-4)
    },
    "heteroskedastic_noise": {
        "base_std": (0.1, 1.5),
        "mod_freq": (0.01, 0.1),
        "noise_type": ["gaussian", "t"],
        "clip_sigma": (1.0, 2.0)
    },
    "autoregressive": {
        "coeffs": "auto",
        "noise_scale": (0.05, 0.3)
    },
    # deterministic waves & trends
    "sinusoid": {"period": (5, 200), "amplitude": (0.1, 2.0), "phase": (0.0, 2*np.pi)},
    "sum_of_sines": {"periods": "auto", "amps": "auto", "phases": "auto"},
    "sawtooth": {"period": (5, 200)},
    "triangle": {"period": (5, 200)},
    "square": {"period": (5, 200)},
    "polynomial_trend": {"coeffs": "auto"},
    "exponential_trend": {"a": (0.5, 2.0), "b": (0.001, 0.02)},
    "logistic_growth": {"K": (0.5, 2.0), "r": (0.01, 0.3), "x0": (0.01, 0.2)},
    "damped_harmonic": {"omega": (0.01, 0.5), "zeta": (0.001, 0.2), "x0": (0.5, 2.0), "v0": (-1.0, 1.0), "dt": (0.1, 1.0)}
}

# ─── 2) Statistical processes parameter space ───
statistical_param_space = {
    "ar":     {"p": (1, 4), "coeffs": "auto", "noise_scale": (0.01, 1.0)},
    "ma":     {"q": (1, 6), "coeffs": "auto", "noise_scale": (0.01, 1.0)},
    "arma":   {"p": (1, 4), "q": (1, 4), "coeffs": "auto", "noise_scale": (0.01, 1.0)},
    "arima":  {"p": (1, 4), "d": (0, 1), "q": (1, 4), "coeffs": "auto", "noise_scale": (0.01, 0.5)},
    "garch":  {"p": (1, 2), "q": (1, 2), "omega": (0.001, 0.05), "alpha": (0.2, 0.5), "beta": (0.4, 0.9)},
    "ou":     {"theta": (0.01, 1.0), "mu": (-5.0, 5.0), "sigma": (0.1, 2.0)},
    "jump_diff": {"mu": (0.0, 0.2), "sigma": (0.1, 1.0), "jump_lambda": (0.05, 0.5), "jump_mu": (0.0, 2.0), "jump_sigma": (0.5, 5.0)},
    "hawkes": {"mu": (0.01, 0.5), "alpha": (0.05, 2.0), "beta": (0.1, 2.0)},
    "hmm":    {"n_states": (2, 5), "mus": "auto", "sigmas": "auto", "transition": "auto"},
    "msar":   {"n_states": (2, 4), "coeffs": "auto", "noise_scale": (0.01, 1.0)},
    "tar":    {"threshold": (-0.2, 0.2), "coeffs1": "auto", "coeffs2": "auto", "noise_scale": (0.01, 0.2)}
}

# ─── 3) Chaotic attractors parameter space ───
chaotic_param_space = {
    "lorenz": {"dt": (0.005, 0.02), "sigma": (8.0, 14.0), "rho": (25.0, 35.0), "beta": (2.0, 3.0), "noise_scale": (0.0, 0.05)},
    "rossler": {"dt": (0.005, 0.02), "a": (0.1, 0.3), "b": (0.1, 0.3), "c": (4.0, 7.0), "noise_scale": (0.0, 0.05)},
    "duffing": {"dt": (0.005, 0.02), "alpha": (0.5, 1.5), "beta": (-1.5, -0.5), "delta": (0.1, 0.3), "gamma": (0.1, 0.5), "omega": (0.5, 1.5), "noise_scale": (0.0, 0.05)},
    "logistic_map": {"r": (3.5, 3.9), "x0": (0.1, 0.9), "burn_in": (50, 200)}
}

# ─── 4) Mechanistic models parameter space ───
mechanistic_param_space = {
    "pendulum": {"g": (9.81, 9.81), "L": (1.0, 1.0), "theta0": (0.01, 0.5), "omega0": (-1.0, 1.0), "dt": (0.01, 0.05)},
    "lotka_volterra": {"alpha": (0.1, 1.0), "beta": (0.01, 0.5), "delta": (0.01, 0.5), "gamma": (0.1, 1.0), "x0": (5, 15), "y0": (1, 10), "dt": (0.01, 0.05)},
    "sir": {"beta": (0.1, 0.5), "gamma": (0.05, 0.2), "S0": (0.5, 1.0), "I0": (0.01, 0.5), "R0": (0.0, 0.1), "dt": (0.1, 1.0)},
    "heat_equation": {"n_space": (5, 20), "alpha": (0.01, 0.5), "dt": (0.01, 0.1), "dx": (0.5, 2.0)}
}

# ─── 5) Gaussian-process parameter space ───
gaussian_param_space = {
    "gaussian_process": {"max_kernels": (1, 5)}
}

# ─── 6) Merge into a global parameter space ───
global_param_space: dict = {}
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