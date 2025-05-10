"""
Generators for time series data based on chaotic systems.
"""

import numpy as np
import random

def generate_lorenz(
 length: int,
 dt: float = 0.01,
 sigma: float = 10.0,
 rho: float = 28.0,
 beta: float = 8/3,
 x0: float = 1.0,
 y0: float = 1.0,
 z0: float = 1.0,
 noise_scale: float = 0.0,
 burn_in: int = 100,
 ic_jitter: float = 0.1,
 normalize: bool = True,
 seed: int | None = None
) -> np.ndarray:
 """
 Lorenz attractor → x-coordinate.
 • burn_in: discard the first ‘burn_in’ steps.
 • ic_jitter: ±jitter on (x0,y0,z0).
 • normalize: subtract mean & divide by std.
 • noise_scale: uniform multiplier jitter after normalization.
 """
 if seed is not None:
 random.seed(seed)
 np.random.seed(seed)

 # jitter initial conditions
 x = x0 + random.uniform(-ic_jitter, ic_jitter)
 y = y0 + random.uniform(-ic_jitter, ic_jitter)
 z = z0 + random.uniform(-ic_jitter, ic_jitter)

 # simulate burn_in + length
 N = length + burn_in
 out = np.empty(N)
 for i in range(N):
 dx = sigma * (y - x)
 dy = x * (rho - z) - y
 dz = x * y - beta * z
 x += dx * dt
 y += dy * dt
 z += dz * dt
 out[i] = x

 ts = out[burn_in:]

 # optional uniform‐jitter noise
 if noise_scale > 0:
 ts = ts * np.random.uniform(1 - noise_scale, 1 + noise_scale, size=length)

 # normalize
 if normalize:
 ts = (ts - ts.mean()) / ts.std()

 return ts


def generate_rossler(
 length: int,
 dt: float = 0.01,
 a: float = 0.2,
 b: float = 0.2,
 c: float = 5.7,
 x0: float = 0.0,
 y0: float = 1.0,
 z0: float = 1.0,
 noise_scale: float = 0.0,
 burn_in: int = 100,
 ic_jitter: float = 0.1,
 normalize: bool = True,
 seed: int | None = None
) -> np.ndarray:
 """
 Rössler attractor → x-coordinate, with burn-in, IC jitter, normalization, and multiplier‐noise.
 """
 if seed is not None:
 random.seed(seed)
 np.random.seed(seed)

 x = x0 + random.uniform(-ic_jitter, ic_jitter)
 y = y0 + random.uniform(-ic_jitter, ic_jitter)
 z = z0 + random.uniform(-ic_jitter, ic_jitter)

 N = length + burn_in
 out = np.empty(N)
 for i in range(N):
 dx = -y - z
 dy = x + a * y
 dz = b + z * (x - c)
 x += dx * dt
 y += dy * dt
 z += dz * dt
 out[i] = x

 ts = out[burn_in:]

 if noise_scale > 0:
 ts = ts * np.random.uniform(1 - noise_scale, 1 + noise_scale, size=length)

 if normalize:
 ts = (ts - ts.mean()) / ts.std()

 return ts


def generate_duffing(
 length: int,
 dt: float = 0.01,
 alpha: float = 1.0,
 beta: float = -1.0,
 delta: float = 0.2,
 gamma: float = 0.3,
 omega: float = 1.2,
 x0: float = 0.0,
 v0: float = 0.0,
 noise_scale: float = 0.0,
 burn_in: int = 100,
 ic_jitter: float = 0.1,
 normalize: bool = True,
 seed: int | None = None
) -> np.ndarray:
 """
 Duffing oscillator → x-coordinate, with burn-in, IC jitter, normalization, and multiplier‐noise.
 """
 if seed is not None:
 random.seed(seed)
 np.random.seed(seed)

 x = x0 + random.uniform(-ic_jitter, ic_jitter)
 v = v0 + random.uniform(-ic_jitter, ic_jitter)

 N = length + burn_in
 out = np.empty(N)
 for i in range(N):
 dx = v
 dv = -delta * v - alpha * x - beta * x**3 + gamma * np.cos(omega * i * dt)
 x += dx * dt
 v += dv * dt
 out[i] = x

 ts = out[burn_in:]

 if noise_scale > 0:
 ts = ts * np.random.uniform(1 - noise_scale, 1 + noise_scale, size=length)

 if normalize:
 ts = (ts - ts.mean()) / ts.std()

 return ts


def generate_logistic_map(
 length: int,
 r: float = 3.9,
 x0: float = 0.5,
 burn_in: int = 100,
 ic_jitter: float = 0.1,
 normalize: bool = True,
 seed: int | None = None
) -> np.ndarray:
 """
 Logistic map: x_{t+1} = r x_t (1-x_t), with burn-in, IC jitter, normalization.
 """
 if seed is not None:
 random.seed(seed)
 np.random.seed(seed)

 x = x0 + random.uniform(-ic_jitter, ic_jitter)
 N = length + burn_in
 out = np.empty(N)
 for i in range(N):
 x = r * x * (1 - x)
 out[i] = x

 ts = out[burn_in:]

 if normalize:
 ts = (ts - ts.mean()) / ts.std()

 return ts

__all__ = [
 "generate_lorenz",
 "generate_rossler",
 "generate_duffing",
 "generate_logistic_map",
]
