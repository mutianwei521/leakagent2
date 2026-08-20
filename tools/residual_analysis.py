"""
ResidualTool (T1): compare an observation to a twin-predicted sensor response
and return a physical-consistency verdict + a Gaussian likelihood.

  residual  r = observed Δp - predicted Δp   (at sensors)
  z         = r / sigma_field
  Mahalanobis-per-dof = sqrt( mean(z^2) )
  loglik    = -0.5 * sum(z^2) - m*log(sigma*sqrt(2*pi))
  passed    = Mahalanobis-per-dof <= pass_threshold

`mask` lets a sensor-fault hypothesis "explain away" specific sensors: those
channels are excluded from the residual (the fault is the explanation there).
"""
from __future__ import annotations
import numpy as np

from .base import Tool


class ResidualTool(Tool):
    name = "residual_analysis"
    cost = 0.0

    def __init__(self, sensor_std=0.15, pass_threshold=3.0):
        self.sensor_std = float(sensor_std)
        self.pass_threshold = float(pass_threshold)

    def analyze(self, observation, predicted_dp, sensors, mask=None, sensor_std=None):
        sigma = float(sensor_std if sensor_std is not None else self.sensor_std)
        use = [s for s in sensors if not (mask and s in mask)]
        if not use:
            return {"rmse": 0.0, "mahalanobis_per_dof": 0.0, "loglik": 0.0,
                    "passed": True, "n_used": 0}
        r = np.array([observation.get(s, 0.0) - predicted_dp.get(s, 0.0) for s in use],
                     dtype=float)
        m = len(r)
        z = r / sigma
        rmse = float(np.sqrt(np.mean(r ** 2)))
        mahal = float(np.sqrt(np.mean(z ** 2)))
        loglik = float(-0.5 * np.sum(z ** 2) - m * np.log(sigma * np.sqrt(2 * np.pi)))
        return {
            "rmse": rmse,
            "mahalanobis_per_dof": mahal,
            "loglik": loglik,
            "passed": bool(mahal <= self.pass_threshold),
            "n_used": m,
            "max_abs_residual": float(np.max(np.abs(r))),
        }

    def run(self, ev, ctx, **kwargs):
        return self.analyze(ev.observation, kwargs["predicted_dp"], ctx.sensors,
                            mask=kwargs.get("mask"),
                            sensor_std=kwargs.get("sensor_std"))
