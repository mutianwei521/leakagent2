"""
DEMAND-ANOMALY class (a non-leak confounder).

A demand anomaly is a DMA/zone-wide proportional over-draw (festival,
firefighting, large illegal tap distributed across consumers), NOT a single
pipe leak. Hydraulically it lowers pressure across a whole partition more
broadly/uniformly than a single-node leak, which is the signature the
Supervisor must use to refuse a (false) leak dispatch.
"""
from __future__ import annotations
import numpy as np
from .leak_anomaly import observe


def dma_demand_modifier(dma_nodes, factor: float):
    def _mod(wc):
        for n in dma_nodes:
            try:
                for dem in wc.get_node(n).demand_timeseries_list:
                    if dem.base_value is not None:
                        dem.base_value *= factor
            except Exception:
                continue
    return _mod


def make_demand_anomaly_scenario(twin, node_to_partition, sensors, partition,
                                 factor=1.6, period="day_normal",
                                 noise_std=0.02, rng=None):
    rng = rng if rng is not None else np.random.default_rng(0)
    dma_nodes = [n for n, p in node_to_partition.items() if p == partition]
    if not dma_nodes:
        return None
    dp = twin.delta(dma_demand_modifier(dma_nodes, factor), period)
    if dp is None:
        return None
    return {
        "true_class": "demand_anomaly",
        "leak_node": None,
        "true_region": int(partition),       # the over-drawing zone
        "demand_factor": float(factor),
        "demand_period": period,
        "noise_std": float(noise_std),
        "observation": observe(dp, sensors, noise_std, rng),
        "full_dp": dp,
        "max_abs_dp": float(max((abs(v) for v in dp.values()), default=0.0)),
    }
