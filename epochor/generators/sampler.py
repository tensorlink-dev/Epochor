import numpy as np
import random

from .param_spaces import global_param_space, synthetic_param_space # Assuming param_spaces.py is in the same directory or a sibling directory
from .registry import blended_registry, statistical_registry, chaotic_registry, mechanistic_registry, gaussian_registry # Assuming registry.py is in the same directory or a sibling directory

SEASONAL_KERNS = {
    "seasonality", "sinusoid", "sum_of_sines",
    "sawtooth", "triangle", "square"
}

def sample_params(name, length):
    """
    Samples parameters for ANY generator by looking up its spec
    in the merged global_param_space.
    """
    if name not in global_param_space:
        raise KeyError(f"No parameter space defined for '{name}'")
    spec = global_param_space[name]
    cfg = {}

    for k,v in spec.items():
        # 1) Sum-of-sines needs special handling:
        if name=="sum_of_sines" and k in ("periods","amps","phases"):
            # pick between 1–3 components
            n = random.randint(1, 3)
            if k=="periods":
                lo,hi = synthetic_param_space["sinusoid"]["period"]
                cfg[k] = [round(random.uniform(lo,hi),2) for _ in range(n)]
            elif k=="amps":
                lo,hi = synthetic_param_space["sinusoid"]["amplitude"]
                cfg[k] = [round(random.uniform(lo,hi),2) for _ in range(n)]
            else:  # phases
                cfg[k] = [round(random.uniform(0, 2*np.pi),2) for _ in range(n)]
            continue

        # 2) Everything else “auto”:
        if v=="auto":
            if k=="event_locs":
                cfg[k] = sorted(random.sample(range(1,length), 2))
            elif name=="hmm" and k=="mus":
                cfg[k] = [round(random.uniform(-1,1),2) for _ in range(cfg.get("n_states",3))]
            elif name=="hmm" and k=="sigmas":
                cfg[k] = [round(random.uniform(0.1,2.0),2) for _ in range(cfg.get("n_states",3))]
            elif k in ("coeffs","coeffs1","coeffs2"):
                order = max(cfg.get("p",1), cfg.get("q",1), 1)
                cfg[k] = [round(random.uniform(-0.9,0.9),3) for _ in range(order)]
            elif k=="transition":
                n = cfg.get("n_states",3)
                cfg[k] = np.ones((n,n)) / n
            else:
                raise ValueError(f"Cannot auto-sample '{k}' for '{name}'")
            continue

        # 3) Integer ranges
        if isinstance(v, tuple) and all(isinstance(z,int) for z in v):
            cfg[k] = random.randint(v[0], v[1])
            continue

        # 4) Continuous ranges
        if isinstance(v, tuple):
            cfg[k] = round(random.uniform(v[0], v[1]), 4)
            continue

        # 5) Categorical choices
        cfg[k] = random.choice(v)

    # ── post-fix for GARCH so α+β <1 ──
    if name=="garch":
        α, β = cfg["alpha"], cfg["beta"]
        if α + β >= 0.99:
            cfg["beta"] = round(0.99 - α, 4)

    return cfg

def random_blended_config(
    length=500,
    seed=None,
    min_k=1,
    max_k=5,
    peak_k=3,
    spread=1.0,
    force_seasonal=False
):
    """
    Build a blended config of k ~ N(peak_k, spread) rounded to [min_k,max_k].
    Always includes 'noise' + (k-1) others.
    If force_seasonal=True, ensures at least one seasonal kernel is included.
    Weights are drawn U(0.2,1.0) then normalized to sum to 1.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 1) pick k
    k = int(round(np.random.normal(peak_k, spread)))
    k = max(min_k, min(max_k, k))

    # 2) always include noise
    blend = [{
        "name": "noise",
        "weight": random.uniform(0.2, 1.0),
        "params": sample_params("noise", length)
    }]

    # 3) prepare pool of other kernels
    pool = [nm for nm in blended_registry if nm != "noise"]
    n_to_add = k - 1

    # 4) if forcing seasonality, grab one first
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

    # 5) fill the rest at random
    if n_to_add > 0:
        for nm in random.sample(pool, min(n_to_add, len(pool))):
            blend.append({
                "name": nm,
                "weight": random.uniform(0.2, 1.0),
                "params": sample_params(nm, length)
            })

    # 6) normalize weights
    total_w = sum(item["weight"] for item in blend)
    for item in blend:
        item["weight"] /= total_w

    return {
        "length": length,
        "blend_kernels": blend
    }



def random_statistical_config(length=500, seed=None):
    if seed is not None: random.seed(seed); np.random.seed(seed)
    name = random.choice(list(statistical_registry))
    return {"name":name,"length":length,"params":sample_params(name,length)}

def random_chaotic_config(length=500, seed=None):
    if seed is not None: random.seed(seed); np.random.seed(seed)
    name = random.choice(list(chaotic_registry))
    return {"name":name,"length":length,"params":sample_params(name,length)}

def random_mechanistic_config(length=500, seed=None):
    if seed is not None: random.seed(seed); np.random.seed(seed)
    name = random.choice(list(mechanistic_registry))
    return {"name":name,"length":length,"params":sample_params(name,length)}
def random_gaussian_config(length=500, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    # always the one
    name = "gaussian_process"
    params = sample_params(name, length)
    return {"name": name, "length": length, "params": params}

__all__ = [
    "sample_params",
    "random_blended_config",
    "random_statistical_config",
    "random_chaotic_config",
    "random_mechanistic_config",
    "random_gaussian_config",
]