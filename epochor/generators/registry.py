"""
Central registry for time series generator functions.

This module defines dictionaries that map generator names to their
respective function implementations, categorized by the type of time
series they generate (e.g., blended, statistical, chaotic).
"""

# Import generator functions from their respective modules
from .v1.synthetic import (
    generate_trend,
    generate_seasonality,
    generate_noise,
    generate_changepoint_series,
    generate_fractional_gaussian_noise,
    generate_event_driven_decay,
    generate_time_varying_freq,
    generate_heteroskedastic_noise,
    generate_autoregressive_series,
    generate_sinusoid,
    generate_sum_of_sines,
    generate_sawtooth,
    generate_triangle,
    generate_square,
    generate_polynomial_trend,
    generate_exponential_trend,
    generate_logistic_growth,
    generate_damped_harmonic,
    generate_amplitude_modulated_sinusoid,
)
from .statistical import (
    generate_ar,
    generate_ma,
    generate_arma,
    generate_arima,
    generate_garch,
    generate_ou,
    generate_jump_diffusion,
    generate_hawkes,
    generate_hmm,
    generate_msar,
    generate_tar,
    generate_random_walk,
    generate_seasonal_random_walk,
    generate_egarch,
    generate_tgarch,
    generate_levy_flight,
)
from .chaotic import (
    generate_lorenz,
    generate_rossler,
    generate_duffing,
    generate_logistic_map,
)
from .mechanistic import (
    generate_pendulum,
    generate_lotka_volterra,
    generate_sir,
    generate_heat_equation,
    generate_local_linear_trend,
    generate_holt_winters,
)
from .gp import (
    generate_gaussian_process,
)

# Registries mapping generator names to function references
blended_registry = {
    "trend": generate_trend,
    "seasonality": generate_seasonality,
    "noise": generate_noise,
    "changepoint": generate_changepoint_series,
    "fgn": generate_fractional_gaussian_noise,
    "event_decay": generate_event_driven_decay,
    "time_varying_freq": generate_time_varying_freq,
    "heteroskedastic_noise": generate_heteroskedastic_noise,
    "autoregressive": generate_autoregressive_series,
    "sinusoid": generate_sinusoid,
    "sum_of_sines": generate_sum_of_sines,
    "sawtooth": generate_sawtooth,
    "triangle": generate_triangle,
    "square": generate_square,
    "polynomial_trend": generate_polynomial_trend,
    "exponential_trend": generate_exponential_trend,
    "logistic_growth": generate_logistic_growth,
    "damped_harmonic": generate_damped_harmonic,
    "amplitude_modulated_sinusoid": generate_amplitude_modulated_sinusoid, # New
}

statistical_registry = {
    "ar": generate_ar, # Added existing function ref
    "ma": generate_ma, # Added existing function ref
    "arma": generate_arma,
    "garch": generate_garch,
    "ou": generate_ou,
    "jump_diff": generate_jump_diffusion,
    "hawkes": generate_hawkes, # Was commented, now assuming it's an actual function
    "hmm": generate_hmm,
    "msar": generate_msar,
    "tar": generate_tar,
    "arima": generate_arima, # Added existing function ref
    "random_walk": generate_random_walk, # New
    "seasonal_random_walk": generate_seasonal_random_walk, # New
    "egarch": generate_egarch, # New
    "tgarch": generate_tgarch, # New
    "levy_flight": generate_levy_flight, # New
}

chaotic_registry = {
    "lorenz": generate_lorenz,
    "rossler": generate_rossler,
    "duffing": generate_duffing,
    "logistic_map": generate_logistic_map,
}

mechanistic_registry = {
    "pendulum": generate_pendulum,
    "lotka_volterra": generate_lotka_volterra,
    "sir": generate_sir,
    "heat_equation": generate_heat_equation,
    "local_linear_trend": generate_local_linear_trend, # New
    "holt_winters": generate_holt_winters, # New
}

gaussian_registry = {
    "gaussian_process": generate_gaussian_process
}

__all__ = [
    "blended_registry",
    "statistical_registry",
    "chaotic_registry",
    "mechanistic_registry",
    "gaussian_registry",
]
