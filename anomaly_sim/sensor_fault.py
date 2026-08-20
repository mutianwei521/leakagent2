"""
SENSOR-FAULT class (a non-leak confounder that lives in measurement space).

The hydraulic state is essentially NORMAL, but 1-3 sensor channels read wrong
(bias/drift, stuck, or excess noise). The signature: a few sensors look
anomalous while their hydraulic neighbours read ~0 — which no physical leak can
reproduce. The Supervisor should diagnose 'sensor fault' (recalibrate), NOT
dispatch a repair crew.
"""
from __future__ import annotations
import numpy as np
from .leak_anomaly import observe


def apply_sensor_fault(obs: dict, faulty_sensors, mode="bias", magnitude=0.6, rng=None):
    rng = rng if rng is not None else np.random.default_rng(0)
    out = dict(obs)
    for s in faulty_sensors:
        if mode == "bias":          # fixed offset / slow drift
            out[s] = float(obs.get(s, 0.0) + magnitude)
        elif mode == "stuck":       # frozen channel
            out[s] = 0.0
        elif mode == "noise":       # inflated noise / dropout
            out[s] = float(obs.get(s, 0.0) + rng.normal(0.0, magnitude))
    return out


def make_sensor_fault_scenario(twin, node_to_partition, sensors, faulty_sensors,
                               mode="bias", magnitude=0.8, period="day_normal",
                               noise_std=0.02, rng=None):
    """Normal hydraulic state + a corrupted reading on `faulty_sensors`."""
    rng = rng if rng is not None else np.random.default_rng(0)
    # normal state => baseline Δp is ~0; observation is just noise + the fault
    normal_obs = {s: float(rng.normal(0.0, noise_std)) for s in sensors}
    obs = apply_sensor_fault(normal_obs, faulty_sensors, mode=mode,
                             magnitude=magnitude, rng=rng)
    return {
        "true_class": "sensor_fault",
        "leak_node": None,
        "true_region": -1,
        "faulty_sensors": [str(s) for s in faulty_sensors],
        "fault_mode": mode,
        "fault_magnitude": float(magnitude),
        "demand_period": period,
        "noise_std": float(noise_std),
        "observation": obs,
        "full_dp": {},                 # normal hydraulics: any new sensor reads ~0
        "max_abs_dp": float(max((abs(v) for v in obs.values()), default=0.0)),
    }
