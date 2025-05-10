"""
V1 Synthetic Time Series Generators.

This module provides a collection of kernel functions for generating
various synthetic time series components, including trends, seasonality,
noise, changepoints, and other deterministic or stochastic patterns.
These kernels are typically used in combination to create blended time series.
"""
import numpy as np
import random
try:
    from fbm import FBM # type: ignore
except ImportError:
    FBM = None # type: ignore
from typing import Optional, List, Union, Tuple

# Primitive / "synthetic" kernels
def generate_trend(
    length: int,
    slope: float = 0.1,
    intercept: float = 0.0,
    nonlinear: bool = False
) -> np.ndarray:
    """Generates a linear or non-linear trend."""
    t = np.arange(length)
    return intercept + slope * (np.log1p(t) if nonlinear else t)

def generate_seasonality(
    length: int,
    period: float = 50,
    amplitude: float = 1.0,
    harmonics: int = 1,
    phase: float = 0.0
) -> np.ndarray:
    """Generates seasonality with specified harmonics."""
    t = np.arange(length)
    h = int(harmonics)
    out = np.zeros(length)
    for k in range(1, h + 1):
        if k == 0: continue # Avoid division by zero if amplitude/k later uses k=0
        out += (amplitude / k) * np.sin(2 * np.pi * k * t / period + phase)
    return out

def generate_noise(
    length: int,
    noise_type: str = 'gaussian',
    scale: float = 1.0,
    df: int = 5, 
    clip_sigma: Optional[float] = 2.0,
    clip_abs: Optional[float] = None
) -> np.ndarray:
    """Generates noise of a specified type (gaussian, uniform, t, laplace)."""
    if noise_type == 'gaussian':
        raw = np.random.normal(0, scale, length)
    elif noise_type == 'uniform':
        raw = np.random.uniform(-scale, scale, length)
    elif noise_type == 't':
        raw = np.random.standard_t(df=df, size=length) * scale
    elif noise_type == 'laplace':
        raw = np.random.laplace(0, scale, length)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")
    
    if clip_sigma is not None:
        μ, σ_val = raw.mean(), raw.std()
        if σ_val > 1e-6: 
            raw = np.clip(raw, μ - clip_sigma * σ_val, μ + clip_sigma * σ_val)
            
    if clip_abs is not None:
        raw = np.clip(raw, -clip_abs, clip_abs)
    return raw

# Advanced synthetic kernels
def generate_changepoint_series(
    length: int,
    n_breakpoints: int = 2,
    max_slope: float = 0.02,
    seed: Optional[int] = None
) -> np.ndarray:
    """Generates a series with random changepoints in slope."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    if n_breakpoints < 0 : n_breakpoints = 0
    y = np.zeros(length)
    if length == 0: return y

    if n_breakpoints > 0 and length > 1: 
         max_possible_bps = length -1 
         actual_n_breakpoints = min(n_breakpoints, max_possible_bps)
         if actual_n_breakpoints > 0:
            bps_intermediate = sorted(random.sample(range(1, length), actual_n_breakpoints))
         else:
            bps_intermediate = []
         bps = bps_intermediate + [length] 
    else: 
        bps = [length] 

    slopes = [random.uniform(-max_slope, max_slope) for _ in bps]
    
    start_idx = 0
    current_val = 0.0 
    for i, bp_idx in enumerate(bps):
        end_idx = bp_idx
        seg_len = end_idx - start_idx
        
        if seg_len <= 0:
            if end_idx == length and start_idx == length : 
                 pass
            elif start_idx < length : 
                 y[start_idx:] = current_val 
            start_idx = end_idx
            continue

        current_slope = slopes[i]
        time_in_segment = np.arange(seg_len)
        
        base_val = y[start_idx-1] if start_idx > 0 else 0.0

        segment_values = base_val + current_slope * time_in_segment
        
        actual_end_idx_for_slice = min(start_idx + seg_len, length)
        y[start_idx:actual_end_idx_for_slice] = segment_values[:actual_end_idx_for_slice-start_idx]
        
        if actual_end_idx_for_slice > 0:
            current_val = y[actual_end_idx_for_slice-1] 
        
        start_idx = end_idx
        if start_idx >=length: break

    return y

def generate_fractional_gaussian_noise(
    length: int,
    hurst: float = 0.7
) -> np.ndarray:
    """Generates Fractional Gaussian Noise (FGN) if 'fbm' library is available."""
    if FBM is None:
        return np.random.normal(0, 1, length) 
    f = FBM(n=length, hurst=hurst, length=1, method='daviesharte') 
    return f.fgn()

def generate_event_driven_decay(
    length: int,
    event_locs: Optional[List[int]] = None, 
    decay_rate: float = 0.95,
    magnitude: float = 5.0
) -> np.ndarray:
    """Generates a series with exponential decays triggered at event locations."""
    current_event_locs = [150, 350] if event_locs is None else event_locs
    s = np.zeros(length)
    for loc in current_event_locs:
        if 0 <= loc < length: 
            for i in range(loc, length):
                s[i] += magnitude * (decay_rate**(i - loc))
    return s

def generate_time_varying_freq(
    length: int,
    base_freq: float = 0.05,
    freq_slope: float = 1e-4
) -> np.ndarray:
    """Generates a sinusoid with time-varying frequency."""
    t = np.arange(length)
    freqs = base_freq + freq_slope * t
    freqs = np.maximum(freqs, 1e-6) 
    return np.sin(2 * np.pi * freqs * t)

def generate_heteroskedastic_noise(
    length: int,
    base_std: float = 0.5,
    mod_freq: float = 0.05,
    noise_type: str = "gaussian",
    clip_sigma: Optional[float] = 2.0, 
    df: int = 3 
) -> np.ndarray:
    """Generates noise with time-varying standard deviation."""
    mod = 1 + 0.5 * np.sin(2 * np.pi * mod_freq * np.arange(length))
    varying_std = base_std * mod
    
    if noise_type == "gaussian":
        raw = np.random.normal(0, 1, length) * varying_std 
    elif noise_type == "t":
        raw = np.random.standard_t(df=df, size=length) * varying_std
    else: 
        raise ValueError(f"Unsupported noise_type: {noise_type} for heteroskedastic_noise")

    if clip_sigma is not None: 
        μ, σ_global = raw.mean(), raw.std()
        if σ_global > 1e-6 : 
            raw = np.clip(raw, μ - clip_sigma * σ_global, μ + clip_sigma * σ_global)
    return raw

def generate_autoregressive_series(
    length: int,
    coeffs: Optional[List[float]] = None, 
    noise_scale: float = 0.1,
    seed: Optional[int] = None
) -> np.ndarray:
    """Generates a simple autoregressive series."""
    if seed is not None:
        np.random.seed(seed)
    current_coeffs = [0.8] if coeffs is None else coeffs
        
    order = len(current_coeffs)
    s = np.zeros(length)
    noise = np.random.normal(0, noise_scale, length)
    
    for t in range(order): 
        s[t] = noise[t] 

    for t in range(order, length):
        s[t] = sum(current_coeffs[i] * s[t - i - 1] for i in range(order)) + noise[t]
    return s

# Deterministic waves & trends
def generate_sinusoid(
    length: int,
    period: float = 50,
    amplitude: float = 1.0,
    phase: float = 0.0
) -> np.ndarray:
    """Generates a simple sinusoid wave."""
    t = np.arange(length)
    if period == 0: return np.zeros(length) 
    return amplitude * np.sin(2 * np.pi * t / period + phase)

def generate_sum_of_sines(
    length: int,
    periods: Union[List[float], Tuple[float, ...]] = (20.0, 50.0), 
    amps: Union[List[float], Tuple[float, ...]] = (1.0, 0.5),    
    phases: Optional[Union[List[float], Tuple[float, ...]]] = None 
) -> np.ndarray:
    """Generates a sum of multiple sine waves."""
    t = np.arange(length)
    
    current_periods = list(periods) if hasattr(periods, '__iter__') and not isinstance(periods, (str, bytes)) else [float(periods)]
    current_amps = list(amps) if hasattr(amps, '__iter__') and not isinstance(amps, (str, bytes)) else [float(amps)]
    
    if phases is None:
        current_phases = [0.0] * len(current_periods)
    else:
        current_phases = list(phases) if hasattr(phases, '__iter__') and not isinstance(phases, (str, bytes)) else [float(phases)]

    num_components = max(len(current_periods), len(current_amps), len(current_phases))
    
    def bcast(lst: List[float], target_len: int) -> List[float]:
        if not lst: return [0.0] * target_len 
        return (lst * (target_len // len(lst) + 1))[:target_len]

    final_periods = bcast(current_periods, num_components)
    final_amps = bcast(current_amps, num_components)
    final_phases = bcast(current_phases, num_components)
    
    y = np.zeros(length)
    for P, A, ph in zip(final_periods, final_amps, final_phases):
        if P == 0: continue 
        y += A * np.sin(2 * np.pi * t / P + ph)
    return y

def generate_sawtooth(
    length: int,
    period: float = 50
) -> np.ndarray:
    """Generates a sawtooth wave."""
    if period == 0: return np.zeros(length) 
    t = np.arange(length)
    return 2 * ((t / period) - np.floor(0.5 + t / period))

def generate_triangle(
    length: int,
    period: float = 50
) -> np.ndarray:
    """Generates a triangle wave."""
    if period == 0: return np.zeros(length)
    return np.abs(generate_sawtooth(length, period)) 

def generate_square(
    length: int,
    period: float = 50
) -> np.ndarray:
    """Generates a square wave."""
    if period == 0: return np.zeros(length) 
    return np.sign(np.sin(2 * np.pi * np.arange(length) / period))

def generate_polynomial_trend(
    length: int,
    coeffs: Optional[List[float]] = None 
) -> np.ndarray:
    """Generates a polynomial trend of a given order."""
    current_coeffs = [0.0001, 0.001, 1.0] if coeffs is None else coeffs 
    t = np.arange(length)
    order = len(current_coeffs) - 1
    y = np.zeros(length)
    for i, c_val in enumerate(current_coeffs):
        y += c_val * (t ** (order - i))
    return y

def generate_exponential_trend(
    length: int,
    a: float = 1.0,
    b: float = 0.01
) -> np.ndarray:
    """Generates an exponential trend: a * exp(b*t)."""
    t = np.arange(length)
    return a * np.exp(b * t)

def generate_logistic_growth(
    length: int,
    K: float = 1.0, 
    r: float = 0.1, 
    x0: float = 0.01 
) -> np.ndarray:
    """Generates a logistic growth curve."""
    x = np.zeros(length)
    if length == 0: return x
    x[0] = x0
    for t in range(1, length):
        if K == 0: 
             x[t] = x[t-1] * (1+r) 
        else:
             x[t] = x[t - 1] + r * x[t - 1] * (1 - x[t - 1] / K)
    return x

def generate_damped_harmonic(
    length: int,
    omega: float = 0.1, 
    zeta: float = 0.05, 
    x0: float = 1.0,    
    v0: float = 0.0,    
    dt: float = 1.0     
) -> np.ndarray:
    """Generates a damped harmonic oscillator series."""
    x = np.zeros(length)
    v = np.zeros(length)
    if length == 0: return x
    
    x[0], v[0] = x0, v0
    for t in range(1, length):
        a = -2 * zeta * omega * v[t - 1] - (omega**2) * x[t - 1]
        v[t] = v[t - 1] + a * dt
        x[t] = x[t - 1] + v[t] * dt 
    return x

# --- Appended new function ---
def generate_amplitude_modulated_sinusoid(
    length: int,
    base_amp: float = 1.0,
    mod_freq: float = 0.01,
    carrier_period: float = 50,
    seed: Optional[int] = None 
) -> np.ndarray:
    """
    Generates an amplitude‐modulated sinusoid.
      A(t) * sin(2π t / carrier_period)
      A(t) = base_amp * (1 + 0.5*sin(2π mod_freq * t))

    Args:
        length: The desired length of the time series.
        base_amp: Base amplitude of the carrier wave. Defaults to 1.0.
        mod_freq: Frequency of the modulating sine wave. Defaults to 0.01.
        carrier_period: Period of the carrier sine wave. Defaults to 50.
        seed: Optional random seed (currently not used as the function is deterministic).
              Defaults to None.

    Returns:
        A 1D numpy array representing the amplitude-modulated sinusoid.
    """
    if seed is not None:
        # np.random.seed(seed) # Not used
        # random.seed(seed)    # Not used
        pass 

    if carrier_period == 0:
        return np.zeros(length) 

    t = np.arange(length)
    A_t = base_amp * (1 + 0.5 * np.sin(2 * np.pi * mod_freq * t))
    carrier_wave = np.sin(2 * np.pi * t / carrier_period)
    
    return (A_t * carrier_wave).astype(float)


__all__ = [
    "generate_trend",
    "generate_seasonality",
    "generate_noise",
    "generate_changepoint_series",
    "generate_fractional_gaussian_noise",
    "generate_event_driven_decay",
    "generate_time_varying_freq",
    "generate_heteroskedastic_noise",
    "generate_autoregressive_series",
    "generate_sinusoid",
    "generate_sum_of_sines",
    "generate_sawtooth",
    "generate_triangle",
    "generate_square",
    "generate_polynomial_trend",
    "generate_exponential_trend",
    "generate_logistic_growth",
    "generate_damped_harmonic",
    "generate_amplitude_modulated_sinusoid", # New
]
