"""
Central registry for time series generator functions.

This module defines dictionaries that map generator names to their
respective function implementations, categorized by the type of time
series they generate (e.g., blended, statistical, chaotic).

It uses Enum classes to avoid "magic strings" for generator names,
providing better type safety and code completion support.
"""
from enum import Enum

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

# --- Enum Definitions for Generator Names ---

class BlendedGeneratorName(str, Enum):
    TREND = "trend"
    SEASONALITY = "seasonality"
    NOISE = "noise"
    CHANGEPOINT = "changepoint"
    FGN = "fgn"
    EVENT_DECAY = "event_decay"
    TIME_VARYING_FREQ = "time_varying_freq"
    HETEROSKEDASTIC_NOISE = "heteroskedastic_noise"
    AUTOREGRESSIVE = "autoregressive"
    SINUSOID = "sinusoid"
    SUM_OF_SINES = "sum_of_sines"
    SAWTOOTH = "sawtooth"
    TRIANGLE = "triangle"
    SQUARE = "square"
    POLYNOMIAL_TREND = "polynomial_trend"
    EXPONENTIAL_TREND = "exponential_trend"
    LOGISTIC_GROWTH = "logistic_growth"
    DAMPED_HARMONIC = "damped_harmonic"
    AMPLITUDE_MODULATED_SINUSOID = "amplitude_modulated_sinusoid"

class StatisticalGeneratorName(str, Enum):
    AR = "ar"
    MA = "ma"
    ARMA = "arma"
    ARIMA = "arima"
    GARCH = "garch"
    OU = "ou"
    JUMP_DIFFUSION = "jump_diff"
    HAWKES = "hawkes"
    HMM = "hmm"
    MSAR = "msar"
    TAR = "tar"
    RANDOM_WALK = "random_walk"
    SEASONAL_RANDOM_WALK = "seasonal_random_walk"
    EGARCH = "egarch"
    TGARCH = "tgarch"
    LEVY_FLIGHT = "levy_flight"

class ChaoticGeneratorName(str, Enum):
    LORENZ = "lorenz"
    ROSSLER = "rossler"
    DUFFING = "duffing"
    LOGISTIC_MAP = "logistic_map"

class MechanisticGeneratorName(str, Enum):
    PENDULUM = "pendulum"
    LOTKA_VOLTERRA = "lotka_volterra"
    SIR = "sir"
    HEAT_EQUATION = "heat_equation"
    LOCAL_LINEAR_TREND = "local_linear_trend"
    HOLT_WINTERS = "holt_winters"

class GaussianGeneratorName(str, Enum):
    GAUSSIAN_PROCESS = "gaussian_process"

# --- Registries mapping Enum members to function references ---

blended_registry = {
    BlendedGeneratorName.TREND: generate_trend,
    BlendedGeneratorName.SEASONALITY: generate_seasonality,
    BlendedGeneratorName.NOISE: generate_noise,
    BlendedGeneratorName.CHANGEPOINT: generate_changepoint_series,
    BlendedGeneratorName.FGN: generate_fractional_gaussian_noise,
    BlendedGeneratorName.EVENT_DECAY: generate_event_driven_decay,
    BlendedGeneratorName.TIME_VARYING_FREQ: generate_time_varying_freq,
    BlendedGeneratorName.HETEROSKEDASTIC_NOISE: generate_heteroskedastic_noise,
    BlendedGeneratorName.AUTOREGRESSIVE: generate_autoregressive_series,
    BlendedGeneratorName.SINUSOID: generate_sinusoid,
    BlendedGeneratorName.SUM_OF_SINES: generate_sum_of_sines,
    BlendedGeneratorName.SAWTOOTH: generate_sawtooth,
    BlendedGeneratorName.TRIANGLE: generate_triangle,
    BlendedGeneratorName.SQUARE: generate_square,
    BlendedGeneratorName.POLYNOMIAL_TREND: generate_polynomial_trend,
    BlendedGeneratorName.EXPONENTIAL_TREND: generate_exponential_trend,
    BlendedGeneratorName.LOGISTIC_GROWTH: generate_logistic_growth,
    BlendedGeneratorName.DAMPED_HARMONIC: generate_damped_harmonic,
    BlendedGeneratorName.AMPLITUDE_MODULATED_SINUSOID: generate_amplitude_modulated_sinusoid,
}

statistical_registry = {
    StatisticalGeneratorName.AR: generate_ar,
    StatisticalGeneratorName.MA: generate_ma,
    StatisticalGeneratorName.ARMA: generate_arma,
    StatisticalGeneratorName.ARIMA: generate_arima,
    StatisticalGeneratorName.GARCH: generate_garch,
    StatisticalGeneratorName.OU: generate_ou,
    StatisticalGeneratorName.JUMP_DIFFUSION: generate_jump_diffusion,
    StatisticalGeneratorName.HAWKES: generate_hawkes,
    StatisticalGeneratorName.HMM: generate_hmm,
    StatisticalGeneratorName.MSAR: generate_msar,
    StatisticalGeneratorName.TAR: generate_tar,
    StatisticalGeneratorName.RANDOM_WALK: generate_random_walk,
    StatisticalGeneratorName.SEASONAL_RANDOM_WALK: generate_seasonal_random_walk,
    StatisticalGeneratorName.EGARCH: generate_egarch,
    StatisticalGeneratorName.TGARCH: generate_tgarch,
    StatisticalGeneratorName.LEVY_FLIGHT: generate_levy_flight,
}

chaotic_registry = {
    ChaoticGeneratorName.LORENZ: generate_lorenz,
    ChaoticGeneratorName.ROSSLER: generate_rossler,
    ChaoticGeneratorName.DUFFING: generate_duffing,
    ChaoticGeneratorName.LOGISTIC_MAP: generate_logistic_map,
}

mechanistic_registry = {
    MechanisticGeneratorName.PENDULUM: generate_pendulum,
    MechanisticGeneratorName.LOTKA_VOLTERRA: generate_lotka_volterra,
    MechanisticGeneratorName.SIR: generate_sir,
    MechanisticGeneratorName.HEAT_EQUATION: generate_heat_equation,
    MechanisticGeneratorName.LOCAL_LINEAR_TREND: generate_local_linear_trend,
    MechanisticGeneratorName.HOLT_WINTERS: generate_holt_winters,
}

gaussian_registry = {
    GaussianGeneratorName.GAUSSIAN_PROCESS: generate_gaussian_process
}

__all__ = [
    "blended_registry",
    "statistical_registry",
    "chaotic_registry",
    "mechanistic_registry",
    "gaussian_registry",
    "BlendedGeneratorName",
    "StatisticalGeneratorName",
    "ChaoticGeneratorName",
    "MechanisticGeneratorName",
    "GaussianGeneratorName",
]
