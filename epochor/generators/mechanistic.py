"""
Generators for time series data based on mechanistic or physical models.

This module includes functions to generate data from systems like
pendulums, Lotka-Volterra predator-prey models, SIR epidemic models,
the heat equation, local linear trends, and Holt-Winters smoothing.
"""
import numpy as np
import random # Added for seed usage
from typing import Optional, List # Added for type hints

def generate_pendulum(
    length: int, 
    g: float = 9.81, 
    L: float = 1.0, 
    theta0: float = 0.1, 
    omega0: float = 0.0, 
    dt: float = 0.01,
    seed: Optional[int] = None # Added seed for completeness, though not used if np.random not called
) -> np.ndarray:
    """Generates the angle of a simple pendulum over time."""
    # if seed is not None: # No stochastic elements in this version
    #     np.random.seed(seed)
    theta = np.zeros(length)
    omega = np.zeros(length)
    if length == 0: return theta
    theta[0], omega[0] = theta0, omega0
    for t in range(1, length):
        a = -(g / L) * np.sin(theta[t - 1])
        omega[t] = omega[t - 1] + a * dt
        theta[t] = theta[t - 1] + omega[t] * dt
    return theta

def generate_lotka_volterra(
    length: int, 
    alpha: float = 1.0, 
    beta: float = 0.1, 
    delta: float = 0.1,
    gamma: float = 1.0, 
    x0: float = 10, 
    y0: float = 5, 
    dt: float = 0.01,
    seed: Optional[int] = None # Added seed
) -> np.ndarray:
    """Generates time series for prey (x) and predator (y) populations."""
    # if seed is not None: # No stochastic elements
    #     np.random.seed(seed)
    xs = np.zeros(length)
    ys = np.zeros(length)
    if length == 0: return np.stack([xs, ys], axis=1)
    xs[0], ys[0] = x0, y0
    for t in range(1, length):
        dx = alpha * xs[t - 1] - beta * xs[t - 1] * ys[t - 1]
        dy = delta * xs[t - 1] * ys[t - 1] - gamma * ys[t - 1]
        xs[t] = xs[t - 1] + dx * dt
        ys[t] = ys[t - 1] + dy * dt
    return np.stack([xs, ys], axis=1)

def generate_sir(
    length: int, 
    beta: float = 0.3, # Infection rate
    gamma: float = 0.1, # Recovery rate
    S0: float = 0.9, 
    I0: float = 0.1, 
    R0: float = 0.0, 
    dt: float = 1.0,
    seed: Optional[int] = None # Added seed
) -> np.ndarray:
    """Generates time series for Susceptible, Infected, and Recovered populations."""
    # if seed is not None: # No stochastic elements
    #     np.random.seed(seed)
    S = np.zeros(length)
    I = np.zeros(length)
    R = np.zeros(length)
    if length == 0: return np.stack([S, I, R], axis=1)
    S[0], I[0], R[0] = S0, I0, R0
    for t in range(1, length):
        dS = -beta * S[t - 1] * I[t - 1]
        dI = beta * S[t - 1] * I[t - 1] - gamma * I[t - 1]
        dR = gamma * I[t - 1]
        S[t] = S[t - 1] + dS * dt
        I[t] = I[t - 1] + dI * dt
        R[t] = R[t - 1] + dR * dt
        # Basic check for non-negativity, though model implies it if params are physical
        S[t] = max(S[t], 0)
        I[t] = max(I[t], 0)
        R[t] = max(R[t], 0)
    return np.stack([S, I, R], axis=1)

def generate_heat_equation(
    length: int, # Number of time steps
    n_space: int = 10, # Number of spatial points
    alpha: float = 0.1, # Thermal diffusivity
    dt: float = 0.1, 
    dx: float = 1.0,
    seed: Optional[int] = None # For initial condition randomization
) -> np.ndarray:
    """
    Simulates the 1D heat equation u_t = alpha * u_xx using finite differences.
    Returns the temperature distribution over space at each time step.
    """
    if seed is not None:
        np.random.seed(seed)
        
    u = np.zeros((length, n_space))
    if length == 0 or n_space == 0: return u
        
    # Initial condition (e.g., random or a specific profile)
    u[0] = np.random.rand(n_space) 
    
    # Stability condition for explicit method: alpha * dt / dx^2 <= 0.5
    # Not enforced here, but user should be aware.

    for t in range(1, length):
        prev_u_t = u[t - 1, :]
        laplacian_u = np.zeros(n_space)
        if n_space > 2: # Need at least 3 points for central difference
            laplacian_u[1:-1] = (prev_u_t[2:] - 2 * prev_u_t[1:-1] + prev_u_t[:-2]) / (dx**2)
        
        # Boundary conditions (e.g., Dirichlet u=0, or Neumann du/dx=0)
        # Assuming Neumann (insulated ends) for simplicity if not specified:
        # laplacian_u[0] = (prev_u_t[1] - prev_u_t[0]) / dx**2 (approx, or better use ghost points)
        # laplacian_u[-1] = (prev_u_t[-2] - prev_u_t[-1]) / dx**2 (approx)
        # For this implementation, the ends are effectively fixed by the laplacian_u[1:-1] calculation
        # unless boundary conditions are explicitly handled for laplacian_u[0] and laplacian_u[-1].
        # If laplacian ends are 0, it means du/dt is 0 at ends if u[0] and u[-1] are not updated via BCs.

        u[t, :] = prev_u_t + alpha * laplacian_u * dt
    return u

# --- Appended new functions ---
def generate_local_linear_trend(
    length: int,
    sigma_level: float = 1.0, # Std dev for level noise
    sigma_trend: float = 0.1, # Std dev for trend noise
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generates a time series based on a local linear trend model.
    Level and slope components evolve as random walks:
      level_t = level_{t-1} + slope_{t-1} + ε_level_t
      slope_t = slope_{t-1} + ε_slope_t
    The output is the level component.

    Args:
        length: The desired length of the time series.
        sigma_level: Standard deviation of the noise in the level equation. Defaults to 1.0.
        sigma_trend: Standard deviation of the noise in the slope equation. Defaults to 0.1.
        seed: Optional random seed for reproducibility. Defaults to None.

    Returns:
        A 1D numpy array representing the level of the local linear trend.
    """
    if seed is not None:
        # random.seed(seed) # Not used by np.random
        np.random.seed(seed)
        
    level = np.zeros(length)
    slope = np.zeros(length)
    
    if length == 0:
        return level

    # Initial values for level and slope can be set to 0 or drawn from a prior
    # level[0] = 0.0 (or some initial_level)
    # slope[0] = 0.0 (or some initial_slope)

    for t in range(1, length):
        level_noise = np.random.normal(0, sigma_level)
        trend_noise = np.random.normal(0, sigma_trend)
        
        level[t] = level[t - 1] + slope[t - 1] + level_noise
        slope[t] = slope[t - 1] + trend_noise
        
    return level

def generate_holt_winters(
    length: int,
    alpha: float = 0.5,  # Smoothing factor for level
    beta: float = 0.1,   # Smoothing factor for trend
    gamma: float = 0.1,  # Smoothing factor for seasonality
    season_len: int = 12, # Length of the seasonal cycle
    seasonal_type: str = "add", # "add" or "mul" for additive/multiplicative seasonality
    seed: Optional[int] = None # For any stochastic initialization if added
) -> np.ndarray:
    """
    Generates a time series using Holt-Winters exponential smoothing.
    Supports additive or multiplicative seasonality.

    Args:
        length: The desired length of the time series.
        alpha: Smoothing parameter for the level (0 < alpha < 1). Defaults to 0.5.
        beta: Smoothing parameter for the trend (0 <= beta < 1). Defaults to 0.1.
        gamma: Smoothing parameter for the seasonality (0 <= gamma < 1). Defaults to 0.1.
        season_len: Length of the seasonal period. Must be >= 2. Defaults to 12.
        seasonal_type: Type of seasonality, "add" or "mul". Defaults to "add".
        seed: Optional random seed (currently for potential stochastic initial seasonals).
              Defaults to None.

    Returns:
        A 1D numpy array representing the Holt-Winters smoothed series.
    """
    if not (0 < alpha < 1 and 0 <= beta < 1 and 0 <= gamma < 1):
        # Relaxing strict inequality for beta and gamma to allow no trend/seasonality if 0
        if not (0 < alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1):
          raise ValueError("Alpha, beta, gamma must be in [0, 1] with alpha > 0.")
    if not isinstance(season_len, int) or season_len < 2:
        # season_len=1 means no seasonality, could be allowed if gamma=0.
        # For distinct seasonal components, season_len >= 2.
        raise ValueError("Season length must be an integer >= 2.")
    if seasonal_type not in ["add", "mul"]:
        raise ValueError("Seasonal type must be 'add' or 'mul'.")

    if seed is not None:
        # random.seed(seed) # Not used by np.random
        np.random.seed(seed) # For initializing seasonals if stochastic

    y = np.zeros(length)
    if length == 0: return y

    # Initialization (simple method: use first season_len values if available, or heuristic)
    # More advanced: estimate from first few full seasons.
    # For generation, we need to define initial level, trend, and seasonals.
    
    # Heuristic initialization:
    level = 0.0 # Or estimate from first few data points if this were fitting
    trend = 0.0 # Or estimate

    # Initialize seasonal components
    # Could be all zeros for additive, all ones for multiplicative, or based on data/random.
    # Let's use a simple constant initialization for generation.
    if seasonal_type == "add":
        seasonals = np.zeros(season_len)
        # Optionally, make them slightly varied if seed is used:
        # seasonals = np.random.normal(0, 0.1, season_len) if seed else np.zeros(season_len)
    else: # "mul"
        seasonals = np.ones(season_len)
        # seasonals = np.random.uniform(0.9, 1.1, season_len) if seed else np.ones(season_len)


    # Simulate the series generation
    # Holt-Winters requires some history for good initialization.
    # We are generating, so we define initial components.
    # y[0] value will depend on these initial components.
    
    for t in range(length):
        if t == 0:
            # First point based on initial level, trend, and first seasonal component
            if seasonal_type == "add":
                y[t] = level + trend + seasonals[0]
            else: # "mul"
                y[t] = (level + trend) * seasonals[0]
            # Update level, trend, seasonals based on this first pseudo-observation y[0]
            # Or, more simply, start updates from t=1 using y[0] as if it were observed.
            # Let's refine: level, trend, seasonals are for predicting y[t], then updated with y[t].
            # For generation, y[t] is the output, then components are updated.
            
            # A common way to initialize for generation:
            # Assume level_0, trend_0, seasonals_(-m+1)...s_0
            # y_1 = (level_0 + trend_0) * s_{1-m} (mul) or + s_{1-m} (add)
            # level_1 = alpha * (y_1 / s_{1-m}) + (1-alpha)*(level_0+trend_0)
            # etc.
            
            # Simpler for generation: initialize level/trend, then compute y[0]
            # The loop structure is typical for fitting. For generation, y[t] is the result.
            # The provided code structure is more like a filter update.
            # Let's adapt for generation by outputting y[t] based on components *before* update.
            
            # Initial state for components:
            current_level = 0.0 # Example initial level
            current_trend = 0.0 # Example initial trend
            current_seasonals = seasonals.copy()

            if seasonal_type == "add":
                y[t] = current_level + current_trend + current_seasonals[t % season_len]
            else:
                y[t] = (current_level + current_trend) * current_seasonals[t % season_len]
                if y[t] == 0 and current_seasonals[t % season_len] == 0 : y[t] = 1e-6 # avoid issues if both are 0 for mul

            # Update components based on this generated y[t]
            prev_level_for_update = current_level # Store L_{t-1} equivalent
            
            if seasonal_type == "add":
                current_level = alpha * (y[t] - current_seasonals[t % season_len]) + \
                                (1 - alpha) * (prev_level_for_update + current_trend)
                current_seasonals[t % season_len] = gamma * (y[t] - prev_level_for_update - current_trend) + \
                                           (1 - gamma) * current_seasonals[t % season_len]
            else: # "mul"
                # Avoid division by zero if current_seasonals is zero
                denom_s = current_seasonals[t % season_len] if current_seasonals[t % season_len] != 0 else 1e-6
                current_level = alpha * (y[t] / denom_s) + \
                                (1 - alpha) * (prev_level_for_update + current_trend)

                denom_lt = prev_level_for_update + current_trend if (prev_level_for_update + current_trend) != 0 else 1e-6
                if current_level == 0 and denom_lt == 0 and y[t] !=0 : # if current_level became 0 due to y[t]/denom_s=0 but prev_level+trend also 0
                    pass # this state is tricky.
                elif denom_lt == 0 and y[t] !=0 : # if prev level+trend is zero, seasonal update from y[t]/current_level
                     denom_cl = current_level if current_level !=0 else 1e-6
                     current_seasonals[t % season_len] = gamma * (y[t] / denom_cl) + \
                                               (1 - gamma) * current_seasonals[t % season_len]
                else:
                     current_seasonals[t % season_len] = gamma * (y[t] / denom_lt) + \
                                               (1 - gamma) * current_seasonals[t % season_len]


            current_trend = beta * (current_level - prev_level_for_update) + \
                            (1 - beta) * current_trend
            continue


        # For t > 0
        # Predict y[t] using L_{t-1}, T_{t-1}, S_{t-m}
        if seasonal_type == "add":
            y[t] = current_level + current_trend + current_seasonals[t % season_len]
        else: # "mul"
            y[t] = (current_level + current_trend) * current_seasonals[t % season_len]
            if y[t] == 0 and current_seasonals[t % season_len] == 0 and (current_level + current_trend) != 0 : y[t] = 1e-6


        # Update components L_t, T_t, S_t based on observed y[t]
        prev_level_for_update = current_level 
        
        if seasonal_type == "add":
            current_level = alpha * (y[t] - current_seasonals[t % season_len]) + \
                            (1 - alpha) * (prev_level_for_update + current_trend)
            # For S_t, use y[t] - (L_{t-1}+T_{t-1}) for additive. The formula often uses L_t, not L_{t-1}+T_{t-1}.
            # Standard: S_t = gamma*(y_t - L_t) + (1-gamma)*S_{t-m} or S_t = gamma*(y_t - (L_{t-1}+T_{t-1})) + (1-gamma)*S_{t-m}
            # Let's use y[t] - current_level (which is L_t)
            current_seasonals[t % season_len] = gamma * (y[t] - current_level) + \
                                       (1 - gamma) * current_seasonals[t % season_len]
        else: # "mul"
            denom_s = current_seasonals[t % season_len] if current_seasonals[t % season_len] != 0 else 1e-6
            current_level = alpha * (y[t] / denom_s) + \
                            (1 - alpha) * (prev_level_for_update + current_trend)
            
            denom_cl = current_level if current_level != 0 else 1e-6
            current_seasonals[t % season_len] = gamma * (y[t] / denom_cl) + \
                                           (1 - gamma) * current_seasonals[t % season_len]

        current_trend = beta * (current_level - prev_level_for_update) + \
                        (1 - beta) * current_trend
            
    return y

__all__ = [
    "generate_pendulum",
    "generate_lotka_volterra",
    "generate_sir",
    "generate_heat_equation",
    "generate_local_linear_trend", # New
    "generate_holt_winters", # New
]
