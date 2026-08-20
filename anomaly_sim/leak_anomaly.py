"""
LEAK class scenario generator (the 'true positive' class).

A leak is a single-node constant additional demand. The observation is the
sensor-subset pressure delta + field-grade noise.
"""
from __future__ import annotations
import numpy as np


def observe(dp_full: dict, sensors: list, noise_std: float, rng) -> dict:
    """Sample sensor readings = true Δp at sensors + Gaussian noise."""
    return {s: float(dp_full.get(s, 0.0) + rng.normal(0.0, noise_std)) for s in sensors}


def make_leak_scenario(twin, node_to_partition, sensors, leak_node, rate_Ls,
                       period="day_normal", noise_std=0.02, rng=None):
    rng = rng if rng is not None else np.random.default_rng(0)
    dp = twin.leak_delta(leak_node, rate_Ls, period)
    if dp is None:
        return None
    return {
        "true_class": "leak",
        "leak_node": str(leak_node),
        "true_region": int(node_to_partition.get(str(leak_node), -1)),
        "rate_Ls": float(rate_Ls),
        "demand_period": period,
        "noise_std": float(noise_std),
        "observation": observe(dp, sensors, noise_std, rng),
        "full_dp": dp,                 # all-node field, for active sensing
        "max_abs_dp": float(max((abs(v) for v in dp.values()), default=0.0)),
    }
