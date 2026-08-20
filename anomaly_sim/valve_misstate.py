"""
VALVE / PRV / PUMP MIS-STATE class (the operationally dangerous confounder).

A closed/throttled pipe, a wrong PRV setting, or a tripped pump causes large,
structured, often network-wide pressure shifts that can masquerade as a big
leak. Dispatching a repair crew for a mis-set valve is exactly the failure mode
the Supervisor must catch.
"""
from __future__ import annotations
import numpy as np
import wntr
from .leak_anomaly import observe

try:
    from wntr.network import LinkStatus
    _CLOSED = LinkStatus.Closed
except Exception:                # pragma: no cover - fallback for older wntr
    _CLOSED = "Closed"


def close_pipe_modifier(pipe_name):
    def _mod(wc):
        link = wc.get_link(pipe_name)
        try:
            link.initial_status = _CLOSED
        except Exception:
            link.status = _CLOSED
    return _mod


def make_valve_misstate_scenario(twin, node_to_partition, sensors, pipe_name,
                                 period="day_normal", noise_std=0.02, rng=None):
    rng = rng if rng is not None else np.random.default_rng(0)
    dp = twin.delta(close_pipe_modifier(pipe_name), period)
    if dp is None:
        return None
    # the "region" is where the closed asset sits (the higher-Δp endpoint's partition)
    return {
        "true_class": "valve_misstate",
        "leak_node": None,
        "asset": str(pipe_name),
        "true_region": -1,
        "demand_period": period,
        "noise_std": float(noise_std),
        "observation": observe(dp, sensors, noise_std, rng),
        "full_dp": dp,
        "max_abs_dp": float(max((abs(v) for v in dp.values()), default=0.0)),
    }
