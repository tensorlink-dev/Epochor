"""
Generators for time series data based on various statistical models.

This module includes functions to generate data from processes such as
Autoregressive (AR), Moving Average (MA), ARMA, ARIMA, GARCH,
Ornstein-Uhlenbeck (OU), jump diffusion, Hawkes processes,
Hidden Markov Models (HMM), Markov Switching Autoregressive (MSAR),
and Threshold Autoregressive (TAR) models.
"""
import numpy as np
import random
from typing import List, Optional, Tuple, Union # Added Tuple, Union

def generate_ar(
    length: int,
    coeffs: Optional[List[float]] = None,
    noise_scale: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generates an Autoregressive (AR) process time series.

    Args:
        length: The desired length of the time series.
        coeffs: A list of AR coefficients. Defaults to [0.8].
        noise_scale: The standard deviation of the noise term. Defaults to 1.0.
        seed: Optional random seed for reproducibility. Defaults to None.

    Returns:
        A 1D numpy array representing the AR process.
    """
    if seed is not None:
        np.random.seed(seed)
    
    current_coeffs = [0.8] if coeffs is None else coeffs
        
    p = len(current_coeffs)
    x = np.zeros(length)
    e = np.random.normal(0, noise_scale, length)
    
    for t in range(p, length):
        x[t] = sum(current_coeffs[i] * x[t - i - 1] for i in range(p)) + e[t]
    return x

def generate_ma(
    length: int,
    coeffs: Optional[List[float]] = None,
    noise_scale: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generates a Moving Average (MA) process time series.

    Args:
        length: The desired length of the time series.
        coeffs: A list of MA coefficients. Defaults to [0.8].
        noise_scale: The standard deviation of the noise term. Defaults to 1.0.
        seed: Optional random seed for reproducibility. Defaults to None.

    Returns:
        A 1D numpy array representing the MA process.
    """
    if seed is not None:
        np.random.seed(seed)
        
    current_coeffs = [0.8] if coeffs is None else coeffs
        
    q = len(current_coeffs)
    e = np.random.normal(0, noise_scale, length + q)
    x = np.zeros(length)
    for t in range(length):
        x[t] = e[t + q] + sum(current_coeffs[i] * e[t + q - i - 1] for i in range(q))
    return x

def generate_arma(
    length: int,
    ar_coeffs: Optional[List[float]] = None,
    ma_coeffs: Optional[List[float]] = None,
    noise_scale: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generates an Autoregressive Moving Average (ARMA) process time series.

    Args:
        length: The desired length of the time series.
        ar_coeffs: A list of AR coefficients. Defaults to [0.5].
        ma_coeffs: A list of MA coefficients. Defaults to [0.3].
        noise_scale: The standard deviation of the noise term. Defaults to 1.0.
        seed: Optional random seed for reproducibility. Defaults to None.

    Returns:
        A 1D numpy array representing the ARMA process.
    """
    if seed is not None:
        np.random.seed(seed)

    current_ar_coeffs = [0.5] if ar_coeffs is None else ar_coeffs
    current_ma_coeffs = [0.3] if ma_coeffs is None else ma_coeffs
        
    p = len(current_ar_coeffs)
    q = len(current_ma_coeffs)
    
    e = np.random.normal(0, noise_scale, length + q)
    x_extended = np.zeros(length + p) # Use extended array to handle initial AR terms
    
    for t in range(p, length + p):
        ar_sum = sum(current_ar_coeffs[i] * x_extended[t - i - 1] for i in range(p))
        ma_sum = sum(current_ma_coeffs[i] * e[t - p + q - i - 1] for i in range(q)) # Adjusted MA error indexing
        x_extended[t] = ar_sum + e[t - p + q] + ma_sum # Adjusted error term e[t] index
        
    return x_extended[p:]


def generate_arima(
    length: int,
    ar_coeffs: Optional[List[float]] = None,
    ma_coeffs: Optional[List[float]] = None,
    d: int = 1,
    noise_scale: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generates an Autoregressive Integrated Moving Average (ARIMA) process time series.

    Args:
        length: The desired length of the time series.
        ar_coeffs: A list of AR coefficients for the ARMA part. Defaults to [0.5].
        ma_coeffs: A list of MA coefficients for the ARMA part. Defaults to [0.3].
        d: The order of differencing. Defaults to 1.
        noise_scale: The standard deviation of the noise term. Defaults to 1.0.
        seed: Optional random seed for reproducibility. Defaults to None.

    Returns:
        A 1D numpy array representing the ARIMA process.
    """
    # Seeding for ARMA part will be handled by generate_arma
    arma_component = generate_arma(length, ar_coeffs, ma_coeffs, noise_scale, seed)
    
    # Apply integration
    x = arma_component.copy()
    for _ in range(d):
        x = np.cumsum(x) # Integration is summation
    return x[:length] # Ensure final length is correct

def generate_garch(
    length: int,
    omega: float = 0.1,
    alpha: Union[float, List[float]] = 0.1, # alpha can be a list for GARCH(p,q)
    beta: Union[float, List[float]] = 0.8,   # beta can be a list for GARCH(p,q)
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generates a Generalized Autoregressive Conditional Heteroskedasticity (GARCH)
    process time series (specifically GARCH(1,1) if alpha/beta are floats).

    Args:
        length: The desired length of the time series.
        omega: The constant term in the GARCH variance equation. Defaults to 0.1.
        alpha: The coefficient(s) for the lagged squared error terms (ARCH term).
               Defaults to 0.1 (for GARCH(1,1)).
        beta: The coefficient(s) for the lagged conditional variance terms (GARCH term).
              Defaults to 0.8 (for GARCH(1,1)).
        seed: Optional random seed for reproducibility. Defaults to None.

    Returns:
        A 1D numpy array representing the GARCH process (epsilon_t values).
    """
    if seed is not None:
        np.random.seed(seed)

    # Ensure alpha and beta are lists for consistent processing
    current_alpha = [alpha] if isinstance(alpha, float) else alpha
    current_beta = [beta] if isinstance(beta, float) else beta
    
    p = len(current_alpha) # Order of ARCH terms
    q = len(current_beta)  # Order of GARCH terms
    
    eps = np.zeros(length)
    sigma2 = np.zeros(length) # Conditional variance

    # Initial variance (e.g., unconditional variance if sum(alpha)+sum(beta) < 1)
    # For simplicity, starting with a common heuristic if possible, else small positive value
    unconditional_var_denominator = 1 - sum(current_alpha) - sum(current_beta)
    if unconditional_var_denominator > 0:
        sigma2[0] = omega / unconditional_var_denominator
    else:
        sigma2[0] = omega / (1 - 0.95) # Fallback if sum is too high, avoid division by zero

    if sigma2[0] <=0 : sigma2[0] = 1e-4 # Ensure positive initial variance

    eps[0] = np.sqrt(sigma2[0]) * np.random.randn()

    for t in range(1, length):
        arch_sum = sum(current_alpha[i] * eps[t - i - 1]**2 for i in range(min(t, p)))
        garch_sum = sum(current_beta[j] * sigma2[t - j - 1] for j in range(min(t, q)))
        sigma2[t] = omega + arch_sum + garch_sum
        if sigma2[t] <= 0: sigma2[t] = 1e-4 # Ensure variance stays positive
        eps[t] = np.sqrt(sigma2[t]) * np.random.randn()
        
    return eps

def generate_ou(
    length: int,
    theta: float = 0.7,
    mu: float = 0.0,
    sigma: float = 0.3,
    dt: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generates an Ornstein-Uhlenbeck (OU) process time series.

    Args:
        length: The desired length of the time series.
        theta: The rate of mean reversion. Defaults to 0.7.
        mu: The mean of the process. Defaults to 0.0.
        sigma: The volatility of the process. Defaults to 0.3.
        dt: The time step. Defaults to 1.0.
        seed: Optional random seed for reproducibility. Defaults to None.

    Returns:
        A 1D numpy array representing the OU process.
    """
    if seed is not None:
        np.random.seed(seed)
        
    x = np.zeros(length)
    # Set initial value to mu or a draw from stationary distribution if known
    x[0] = mu 
    
    for t in range(1, length):
        dW = np.sqrt(dt) * np.random.randn() # Wiener process increment
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) * dt + sigma * dW
    return x

def generate_jump_diffusion(
    length: int,
    mu: float = 0.0,        # Drift of the diffusion part
    sigma: float = 0.2,     # Volatility of the diffusion part
    jump_lambda: float = 0.1,# Intensity of the Poisson process for jumps
    jump_mu: float = 0.0,   # Mean of the jump size
    jump_sigma: float = 1.0,# Standard deviation of the jump size
    dt: float = 1.0,        # Time step
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generates a jump diffusion process time series (Merton's model).

    The process is X_t = mu*dt + sigma*dW_t + dJ_t, where dJ_t is a compound Poisson process.

    Args:
        length: The desired length of the time series.
        mu: Drift of the continuous part. Defaults to 0.0.
        sigma: Volatility of the continuous part. Defaults to 0.2.
        jump_lambda: Arrival rate of jumps (Poisson intensity). Defaults to 0.1.
        jump_mu: Mean of the jump sizes (normally distributed). Defaults to 0.0.
        jump_sigma: Standard deviation of the jump sizes. Defaults to 1.0.
        dt: Time step. Defaults to 1.0.
        seed: Optional random seed for reproducibility. Defaults to None.

    Returns:
        A 1D numpy array representing the jump diffusion process.
    """
    if seed is not None:
        np.random.seed(seed)
        
    x = np.zeros(length)
    # x[0] can be set to a specific starting point if needed, e.g., 0
    
    for t in range(1, length):
        dW = np.sqrt(dt) * np.random.randn() # Diffusion part
        diffusion_increment = mu * dt + sigma * dW
        
        # Jump part
        num_jumps = np.random.poisson(jump_lambda * dt)
        jump_increment = 0.0
        if num_jumps > 0:
            jumps = np.random.normal(jump_mu, jump_sigma, num_jumps)
            jump_increment = np.sum(jumps)
            
        x[t] = x[t - 1] + diffusion_increment + jump_increment
    return x

def generate_hawkes(
    length: int,        # Number of time steps for discrete counts
    mu: float = 0.05,   # Base intensity
    alpha: float = 0.8, # Excitation factor
    beta: float = 1.0,  # Decay rate of the excitation kernel (exp(-beta*t))
    dt: float = 1.0,    # Time step for discretizing event counts
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generates event counts from a Hawkes process using Ogata's thinning algorithm.
    Returns discrete counts per interval dt over the specified length.

    Args:
        length: The number of discrete time intervals for which counts are returned.
        mu: Base intensity of the Hawkes process. Defaults to 0.05.
        alpha: Excitation factor (alpha >= 0). Defaults to 0.8.
        beta: Decay rate of the exponential kernel (beta > 0). Defaults to 1.0.
        dt: The size of the time intervals for counting events. Defaults to 1.0.
                The total time simulated is length * dt.
        seed: Optional random seed for reproducibility. Defaults to None.

    Returns:
        A 1D numpy array of event counts in each interval dt.
    """
    if seed is not None:
        random.seed(seed) # random for uniform draw
        np.random.seed(seed) # np.random for exponential draw

    event_times = []
    current_time = 0.0
    total_simulation_time = length * dt

    while current_time < total_simulation_time:
        # Calculate current intensity lambda_t
        intensity_at_t = mu + sum(alpha * np.exp(-beta * (current_time - tj)) for tj in event_times if tj < current_time)
        if intensity_at_t <= 0 : intensity_at_t = 1e-5 # Ensure positivity

        # Generate time to next candidate event from M (upper bound on intensity)
        # For simplicity, we adaptively use current_intensity_at_t * 1.1 as M if it's simple,
        # or a more complex upper bound estimation if needed.
        # A simple approach for simulation: generate from homogeneous Poisson with rate M > lambda_t
        # Here, we use a simpler method more direct for Ogata's algorithm
        
        # Time to next event from exponential distribution with current intensity
        time_to_next_event = np.random.exponential(1.0 / intensity_at_t)
        candidate_time = current_time + time_to_next_event
        
        # Thinning: accept candidate with probability lambda(candidate_time) / M
        # Here M is effectively intensity_at_t, so we check if a U(0,1) < lambda(candidate_time)/intensity_at_t_at_candidate_proposal
        # A more standard Ogata: find M_star >= lambda_t over [current_time, candidate_time]
        # then accept if U < lambda(candidate_time) / M_star
        
        # For this simplified version, we'll recalculate intensity at candidate_time
        # and accept if U*intensity_at_previous_event_time < intensity_at_candidate_time
        # This is not strictly Ogata's thinning but a common simulation approach.
        # A more robust Ogata thinning is preferred for accuracy.

        # Let's use a slightly more robust thinning based on current intensity_at_t as a lower bound for M.
        if candidate_time < total_simulation_time:
            # Calculate intensity at the candidate event time
            intensity_at_candidate = mu + sum(alpha * np.exp(-beta * (candidate_time - tj)) for tj in event_times if tj < candidate_time)
            if intensity_at_candidate <= 0: intensity_at_candidate = 1e-5

            # Accept the candidate event with probability intensity_at_candidate / (some M >= intensity_at_candidate)
            # Simplified: if random.uniform(0,1) * M_approx < intensity_at_candidate (where M_approx was used for drawing time_to_next_event)
            # Here, M_approx was intensity_at_t.
            if random.uniform(0, 1) * intensity_at_t < intensity_at_candidate : # Thinning step (simplified)
                 event_times.append(candidate_time)
            current_time = candidate_time # Move time forward
        else:
            current_time = total_simulation_time # End simulation
            
    # Discretize event times into counts per interval dt
    bins = np.arange(0, total_simulation_time + dt, dt)
    counts, _ = np.histogram(event_times, bins=bins)
    return counts[:length] # Ensure correct output length


def generate_hmm(
    length: int,
    n_states: int = 3,
    trans_mat: Optional[np.ndarray] = None, # Transition probability matrix (n_states x n_states)
    mus: Optional[List[float]] = None,      # Means of observations for each state
    sigmas: Optional[List[float]] = None,   # Std deviations of observations for each state
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generates a time series from a Hidden Markov Model (HMM) with Gaussian emissions.

    Args:
        length: The desired length of the time series.
        n_states: The number of hidden states. Defaults to 3.
        trans_mat: The transition probability matrix (rows must sum to 1).
                     If None, a default uniform transition matrix is used.
        mus: A list of mean values for the Gaussian emission of each state.
               If None, defaults to [0, 1, ..., n_states-1].
        sigmas: A list of standard deviations for the Gaussian emission of each state.
                  If None, defaults to [1.0, 1.0, ..., 1.0].
        seed: Optional random seed for reproducibility. Defaults to None.

    Returns:
        A 1D numpy array representing the HMM observations.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed) # For random.choice if used, np.random.choice is better

    # Default transition matrix (uniform transitions)
    current_trans_mat = (np.ones((n_states, n_states)) / n_states
                         if trans_mat is None else trans_mat)
    # Default emission means
    current_mus = (list(range(n_states))
                   if mus is None else mus)
    # Default emission standard deviations
    current_sigmas = ([1.0] * n_states
                      if sigmas is None else sigmas)

    if not (current_trans_mat.shape == (n_states, n_states) and \
            all(np.isclose(current_trans_mat.sum(axis=1), 1.0))):
        raise ValueError("Transition matrix must be n_states x n_states and rows must sum to 1.")
    if not (len(current_mus) == n_states and len(current_sigmas) == n_states):
        raise ValueError("Mus and Sigmas must be lists of length n_states.")

    states = np.zeros(length, dtype=int)
    observations = np.zeros(length)

    # Initial state (e.g., uniform or from stationary distribution)
    states[0] = np.random.choice(n_states) 
    observations[0] = np.random.normal(current_mus[states[0]], current_sigmas[states[0]])

    for t in range(1, length):
        states[t] = np.random.choice(n_states, p=current_trans_mat[states[t - 1]])
        observations[t] = np.random.normal(current_mus[states[t]], current_sigmas[states[t]])
        
    return observations

def generate_msar(
    length: int,
    n_states: int = 2,
    ar_coeffs: Optional[List[Union[float, List[float]]]] = None, # List of AR coeffs for each state. Each element can be a list itself for AR(p) per state.
    trans_mat: Optional[np.ndarray] = None, # Transition probability matrix
    noise_scale: float = 1.0,             # Std deviation of the noise term (common for all states)
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generates a Markov Switching Autoregressive (MSAR) process time series.
    Assumes AR(1) for each state if ar_coeffs elements are floats.

    Args:
        length: The desired length of the time series.
        n_states: The number of hidden states. Defaults to 2.
        ar_coeffs: A list where each element contains AR coefficient(s) for a state.
                     Example for 2 states, AR(1): [[0.5], [-0.2]].
                     If None, defaults to [[0.5 + 0.2*i] for i in range(n_states)].
        trans_mat: The transition probability matrix. If None, defaults to uniform.
        noise_scale: Standard deviation of the innovation noise. Defaults to 1.0.
        seed: Optional random seed for reproducibility. Defaults to None.

    Returns:
        A 1D numpy array representing the MSAR process.
    """
    if seed is not None:
        np.random.seed(seed)

    # Default AR coefficients: AR(1) for each state with varying coeffs
    if ar_coeffs is None:
        current_ar_coeffs = [[0.5 + 0.2 * i] for i in range(n_states)]
    else: # Ensure each element is a list for consistent processing of AR order
        current_ar_coeffs = [([c] if isinstance(c, float) else c) for c in ar_coeffs]


    # Default transition matrix (uniform transitions)
    current_trans_mat = (np.ones((n_states, n_states)) / n_states
                         if trans_mat is None else trans_mat)
    
    if not all(len(c) > 0 for c in current_ar_coeffs):
        raise ValueError("Each state must have at least one AR coefficient.")

    states = np.zeros(length, dtype=int)
    x = np.zeros(length)
    noise = np.random.normal(0, noise_scale, length)

    # Initial state
    states[0] = np.random.choice(n_states)
    # x[0] needs careful initialization, e.g. from unconditional mean or zero
    # For simplicity, start with noise if no prior AR terms
    max_order = max(len(c) for c in current_ar_coeffs)
    if max_order == 0 : x[0] = noise[0] # Should not happen with validation above

    for t in range(length):
        if t > 0: # Determine current state based on previous state
            states[t] = np.random.choice(n_states, p=current_trans_mat[states[t-1]])
        
        current_state_coeffs = current_ar_coeffs[states[t]]
        order = len(current_state_coeffs)
        
        ar_sum = 0.0
        if t >= order: # Ensure enough past values for AR terms
            ar_sum = sum(current_state_coeffs[i] * x[t - i - 1] for i in range(order))
        elif t > 0 : # Handle initial steps with fewer than 'order' past values if order > 1
             # Simplified: use available past terms, or assume 0 for unavailable ones
             available_order = min(t, order)
             ar_sum = sum(current_state_coeffs[i] * x[t - i - 1] for i in range(available_order))


        x[t] = ar_sum + noise[t]
        
    return x


def generate_tar(
    length: int,
    coeffs1: Optional[List[float]] = None,    # AR coefficients for regime 1 (X_{t-1} <= threshold)
    coeffs2: Optional[List[float]] = None,    # AR coefficients for regime 2 (X_{t-1} > threshold)
    threshold: float = 0.0,                 # Threshold value
    noise_scale: float = 1.0,               # Std deviation of the noise term
    delay: int = 1,                         # Delay lag for the threshold variable (d in X_{t-d})
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generates a Threshold Autoregressive (TAR) process time series.
    Assumes a single threshold and two regimes. The threshold variable is X_{t-delay}.

    Args:
        length: The desired length of the time series.
        coeffs1: AR coefficients for the first regime (X_{t-delay} <= threshold). Defaults to [0.5].
        coeffs2: AR coefficients for the second regime (X_{t-delay} > threshold). Defaults to [-0.5].
        threshold: The threshold value. Defaults to 0.0.
        noise_scale: Standard deviation of the innovation noise. Defaults to 1.0.
        delay: The lag of the series to be used as the threshold variable. Must be >= 1. Defaults to 1.
        seed: Optional random seed for reproducibility. Defaults to None.

    Returns:
        A 1D numpy array representing the TAR process.
    """
    if seed is not None:
        np.random.seed(seed)
    if delay < 1:
        raise ValueError("Delay parameter 'd' must be at least 1.")

    current_coeffs1 = [0.5] if coeffs1 is None else coeffs1
    current_coeffs2 = [-0.5] if coeffs2 is None else coeffs2
    
    order1 = len(current_coeffs1)
    order2 = len(current_coeffs2)
    max_order = max(order1, order2, delay) # Max lag needed for AR terms and threshold variable

    x = np.zeros(length)
    noise = np.random.normal(0, noise_scale, length)

    # Initialize first max_order points (e.g., with noise or zeros)
    # For simplicity, let them be affected by noise, assuming x starts around 0
    for t in range(max_order):
        x[t] = noise[t] 

    for t in range(max_order, length):
        ar_sum = 0.0
        threshold_variable = x[t - delay]
        
        if threshold_variable <= threshold: # Regime 1
            ar_sum = sum(current_coeffs1[i] * x[t - i - 1] for i in range(order1))
        else: # Regime 2
            ar_sum = sum(current_coeffs2[i] * x[t - i - 1] for i in range(order2))
            
        x[t] = ar_sum + noise[t]
        
    return x

# ... (all existing statistical generator functions from the read_file output end here) ...

# --- Appended new functions ---
def generate_random_walk(
    length: int, 
    drift: float = 0.0, 
    sigma: float = 1.0, 
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generates a Random Walk time series.
    x_t = x_{t-1} + drift + ε_t, where ε_t ~ N(0, σ^2).
    If x_0 is assumed to be 0, then x_t = sum_{i=1 to t} (drift + ε_i).

    Args:
        length: The desired length of the time series.
        drift: The drift component for each step of the random walk. Defaults to 0.0.
        sigma: The standard deviation of the noise term for each step. Defaults to 1.0.
        seed: Optional random seed for reproducibility. Defaults to None.

    Returns:
        A 1D numpy array representing the Random Walk.
    """
    if seed is not None:
        # random.seed(seed) # Not strictly necessary as np.random is used primarily
        np.random.seed(seed)
    # Increments are normal with mean `drift` and scale `sigma`
    increments = np.random.normal(loc=drift, scale=sigma, size=length)
    return np.cumsum(increments)

def generate_seasonal_random_walk(
    length: int, 
    season: int = 12, 
    sigma: float = 1.0, 
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generates a Seasonal Random Walk time series.
    x_t = x_{t-season} + ε_t, where ε_t ~ N(0, σ^2).

    Args:
        length: The desired length of the time series.
        season: The length of the season. Must be positive. Defaults to 12.
        sigma: The standard deviation of the noise term. Defaults to 1.0.
        seed: Optional random seed for reproducibility. Defaults to None.

    Returns:
        A 1D numpy array representing the Seasonal Random Walk.
    """
    if seed is not None:
        # random.seed(seed) # Not strictly necessary
        np.random.seed(seed)
    if not isinstance(season, int) or season <= 0:
        raise ValueError("Season length must be a positive integer.")
        
    x = np.zeros(length)
    if length == 0:
        return x
        
    # Initialize the first season with random values (e.g., from N(0, sigma))
    for t in range(min(length, season)):
        x[t] = np.random.normal(loc=0, scale=sigma)
    
    # Generate subsequent values
    for t in range(season, length):
        x[t] = x[t - season] + np.random.normal(loc=0, scale=sigma)
    return x

def generate_egarch(
    length: int, 
    omega: float = 0.0, 
    alpha: float = 0.1, 
    gamma: float = 0.1, 
    beta: float = 0.8, 
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generates an EGARCH(1,1) process time series.
    log σ_t^2 = ω + α(|Z_{t-1}| - E|Z|) + γ Z_{t-1} + β log σ_{t-1}^2, 
    where Z_t = ε_t/σ_t ~ N(0,1) and E|Z| = sqrt(2/π).

    Args:
        length: The desired length of the time series.
        omega: Constant term in the log-variance equation. Defaults to 0.0.
        alpha: Coefficient for the magnitude of lagged standardized residual. Defaults to 0.1.
        gamma: Coefficient for the sign/leverage of lagged standardized residual. Defaults to 0.1.
        beta: Coefficient for the lagged log conditional variance. Defaults to 0.8.
        seed: Optional random seed for reproducibility. Defaults to None.

    Returns:
        A 1D numpy array representing the EGARCH process innovations (ε_t).
    """
    if seed is not None:
        np.random.seed(seed) 
        
    eps = np.zeros(length)
    log_sigma2 = np.zeros(length) 

    if length == 0:
        return eps

    log_sigma2[0] = omega / (1 - beta) if abs(1 - beta) > 1e-6 else np.log(1e-4) 
    if np.isnan(log_sigma2[0]) or np.isinf(log_sigma2[0]): 
        log_sigma2[0] = np.log(1e-4)

    sigma_0 = np.sqrt(np.exp(log_sigma2[0]))
    eps[0] = sigma_0 * np.random.randn() 

    expected_abs_z = np.sqrt(2.0 / np.pi) 

    for t in range(1, length):
        std_prev = np.sqrt(np.exp(log_sigma2[t-1]))
        z_prev = eps[t-1] / std_prev if std_prev > 1e-8 else 0.0 
        
        log_sigma2[t] = (
            omega
            + alpha * (np.abs(z_prev) - expected_abs_z) 
            + gamma * z_prev  
            + beta * log_sigma2[t-1]
        )
        
        log_sigma2[t] = np.clip(log_sigma2[t], -50, 50)

        current_sigma = np.sqrt(np.exp(log_sigma2[t]))
        eps[t] = current_sigma * np.random.randn()
    return eps

def generate_tgarch(
    length: int, 
    omega: float = 0.0, 
    alpha_pos: float = 0.1, 
    alpha_neg: float = 0.1, 
    beta: float = 0.8, 
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generates a Threshold GARCH (TGARCH/GJR-GARCH) (1,1) process time series.
    σ_t^2 = ω + (α_pos * I{ε_{t-1}>0} + α_neg * I{ε_{t-1}<0}) * ε_{t-1}^2 + β σ_{t-1}^2.

    Args:
        length: The desired length of the time series.
        omega: Constant term in the variance equation. Defaults to 0.0.
        alpha_pos: Coefficient for squared positive lagged residuals. Defaults to 0.1.
        alpha_neg: Coefficient for squared negative lagged residuals (leverage effect). Defaults to 0.1.
        beta: Coefficient for lagged conditional variance. Defaults to 0.8.
        seed: Optional random seed for reproducibility. Defaults to None.

    Returns:
        A 1D numpy array representing the TGARCH process innovations (ε_t).
    """
    if seed is not None:
        np.random.seed(seed) 
        
    eps = np.zeros(length)
    sigma2 = np.zeros(length) 

    if length == 0:
        return eps

    avg_alpha_effect = (alpha_pos + alpha_neg) / 2.0
    denominator = 1.0 - avg_alpha_effect - beta
    
    if denominator > 1e-6:
        sigma2[0] = omega / denominator
    else:
        sigma2[0] = omega / (1.0 - 0.95) if omega > 0 else 1e-4 

    if sigma2[0] <= 0: sigma2[0] = 1e-4 
    
    eps[0] = np.sqrt(sigma2[0]) * np.random.randn()

    for t in range(1, length):
        eps_lagged_sq = eps[t-1]**2
        indicator_pos = 1.0 if eps[t-1] > 0 else 0.0 
        indicator_neg = 1.0 if eps[t-1] < 0 else 0.0 
        
        sigma2[t] = (
            omega
            + alpha_pos * eps_lagged_sq * indicator_pos
            + alpha_neg * eps_lagged_sq * indicator_neg
            + beta * sigma2[t-1]
        )
        if sigma2[t] <= 0: sigma2[t] = 1e-4 
        eps[t] = np.sqrt(sigma2[t]) * np.random.randn()
    return eps

def generate_levy_flight(
    length: int, 
    alpha: float = 1.7, # Stability parameter (0 < alpha <= 2)
    scale: float = 1.0, # Scale parameter (c > 0)
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generates Lévy‐flight increments: symmetric α‐stable increments.
    Uses the Chambers-Mallows-Stuck (CMS) method for SαS random variables.
    The characteristic function is exp(-|scale*k|^alpha).

    Args:
        length: The number of increments to generate.
        alpha: Stability parameter (characteristic exponent), 0 < alpha <= 2.
               alpha=2 corresponds to Gaussian, alpha=1 to Cauchy.
               Defaults to 1.7.
        scale: Scale parameter (c), must be positive. Determines the spread.
               Defaults to 1.0.
        seed: Optional random seed for reproducibility. Defaults to None.

    Returns:
        A 1D numpy array of Lévy flight increments.
    """
    if not (0 < alpha <= 2):
        raise ValueError("Stability parameter alpha must be in (0, 2].")
    if scale <= 0:
        raise ValueError("Scale parameter must be positive.")
        
    if seed is not None:
        # random.seed(seed) # For np.random.uniform and np.random.exponential
        np.random.seed(seed)
        
    # Generate uniform random numbers in (-pi/2, pi/2)
    theta = np.random.uniform(low=-np.pi / 2.0, high=np.pi / 2.0, size=length)
    # Generate standard exponential random numbers
    w = np.random.exponential(scale=1.0, size=length)

    if alpha == 1: # Cauchy distribution
        # X = scale * tan(theta)
        return scale * np.tan(theta)
    elif alpha == 2: # Gaussian distribution (scaled Wiener process increments)
        # X = sqrt(2) * scale * N(0,1) ~ N(0, 2*scale^2)
        # The CMS method for alpha=2 gives N(0, 2*scale^2).
        # To get N(0, scale^2), need to adjust.
        # However, the Levy flight definition usually implies a specific variance scaling.
        # If CMS results in N(0, 2c^2), then X = N(0, sqrt(2)*c)
        # Let's stick to the direct formula result, which for alpha=2 should be Gaussian.
        # CMS for alpha=2: sin(2theta)/(cos(theta))^(1/2) * (cos(-theta)/W)^(-1/2)
        # = 2sin(theta)cos(theta)/sqrt(cos(theta)) * sqrt(W/cos(-theta))
        # = 2sin(theta)sqrt(cos(theta)) * sqrt(W/cos(theta)) = 2sin(theta)sqrt(W)
        # This is related to Box-Muller. N(0, var=2*scale^2)
        # We want N(0,1)*scale for standard Wiener increments.
        # The formula simplifies to N(0, sqrt(2)*scale).
        # It's often simpler to just draw from Gaussian directly for alpha=2 if that's the goal.
        # The provided formula is general.
        # For alpha=2, (1-alpha)/alpha = -1/2. (cos((1-alpha)theta)/w)**((1-alpha)/alpha) = (cos(-theta)/w)**(-1/2) = sqrt(w/cos(theta))
        # sin(alpha*theta)/(cos(theta)**(1/alpha)) = sin(2theta)/sqrt(cos(theta)) = 2sin(theta)sqrt(cos(theta))
        # Product = 2sin(theta)sqrt(w). This gives a Gaussian with variance 2.
        # So, result is N(0, sqrt(2)*scale).
        return scale * np.random.normal(loc=0, scale=np.sqrt(2.0), size=length)


    # General case for S_alpha(1, 0, 0) then scale by `scale`
    # (S_alpha(sigma, beta, mu) = sigma * S_alpha(1, beta, 0) + mu)
    # Here beta=0 (symmetric)
    
    # Numerator: sin(alpha * theta)
    # Denominator part 1: (cos(theta))**(1/alpha)
    # Denominator part 2 related term: ((np.cos((1-alpha)*theta) / w)**((1-alpha)/alpha))
    
    # Handle alpha = 1 separately because (1-alpha)/alpha term would be problematic.
    # For alpha != 1:
    term1 = np.sin(alpha * theta) / (np.cos(theta)**(1.0/alpha))
    term2_base = np.cos((1.0 - alpha) * theta) / w
    term2 = term2_base**((1.0 - alpha) / alpha)
    
    return scale * term1 * term2

# --- Update __all__ ---
__all__ = [
    "generate_ar",
    "generate_ma",
    "generate_arma",
    "generate_arima",
    "generate_garch",
    "generate_ou",
    "generate_jump_diffusion",
    "generate_hawkes",
    "generate_hmm",
    "generate_msar",
    "generate_tar",
    "generate_random_walk", # New
    "generate_seasonal_random_walk", # New
    "generate_egarch", # New
    "generate_tgarch", # New
    "generate_levy_flight", # New
]
