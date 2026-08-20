"""
DigitalTwinTool (T3): predict the sensor-space pressure response of a hypothesis
by forward-simulating it in the WNTR twin. This is how competing hypotheses are
actively FALSIFIED: if even the best-fit version of a hypothesis cannot
reproduce the observation, its likelihood collapses. All sims are memoized.
"""
from __future__ import annotations
import numpy as np

from .base import Tool
from anomaly_sim.demand_anomaly import dma_demand_modifier
from anomaly_sim.valve_misstate import close_pipe_modifier


class DigitalTwinTool(Tool):
    name = "digital_twin"
    cost = 1.0
    DEFAULT_RATE_GRID = (2.0, 5.0, 10.0, 20.0, 35.0, 50.0)   # cover all severities
    DEFAULT_FACTOR_GRID = (1.3, 1.6, 1.9, 2.2, 2.5)

    def __init__(self, rate_grid=None, factor_grid=None, registry=None):
        self.rate_grid = tuple(rate_grid) if rate_grid else self.DEFAULT_RATE_GRID
        self.factor_grid = tuple(factor_grid) if factor_grid else self.DEFAULT_FACTOR_GRID
        self.registry = registry

    def _charge(self, n=1):
        if self.registry is not None:
            self.registry.charge("twin_sims", n)

    def _at_sensors(self, dp, sensors):
        return {s: float(dp.get(s, 0.0)) for s in sensors}

    @staticmethod
    def _sse(observation, pred, sensors):
        return sum((observation.get(s, 0.0) - pred[s]) ** 2 for s in sensors)

    # ---- leak (single node) ----------------------------------------------
    def predict_leak(self, ctx, node, period, observation=None, rate=None):
        rates = [rate] if rate is not None else list(self.rate_grid)
        best = None
        for r in rates:
            dp = ctx.twin.leak_delta(node, r, period)
            self._charge(1)
            if dp is None:
                continue
            pred = self._at_sensors(dp, ctx.sensors)
            err = 0.0 if observation is None else self._sse(observation, pred, ctx.sensors)
            if best is None or err < best[2]:
                best = (pred, r, err)
            if observation is None:
                break
        if best is None:
            return {"predicted_dp": {s: 0.0 for s in ctx.sensors}, "rate": None,
                    "node": str(node), "reproducible": False}
        pred, r, _ = best
        return {"predicted_dp": pred, "rate": r, "node": str(node), "reproducible": True}

    # ---- leak (partition = best over candidate nodes x rates) ------------
    def predict_leak_partition(self, ctx, candidate_nodes, period, observation=None,
                               rate_grid=None, sensors=None):
        rates = list(rate_grid or self.rate_grid)
        sensors = sensors if sensors is not None else ctx.sensors
        best = None
        for node in (candidate_nodes or []):
            for r in rates:
                dp = ctx.twin.leak_delta(node, r, period)
                self._charge(1)
                if dp is None:
                    continue
                pred = self._at_sensors(dp, sensors)
                err = 0.0 if observation is None else self._sse(observation, pred, sensors)
                if best is None or err < best[3]:
                    best = (pred, node, r, err)
        if best is None:
            return {"predicted_dp": {s: 0.0 for s in ctx.sensors}, "node": None,
                    "rate": None, "reproducible": False}
        pred, node, r, _ = best
        return {"predicted_dp": pred, "node": str(node), "rate": r, "reproducible": True}

    # ---- demand anomaly (DMA-wide over-draw) -----------------------------
    def predict_demand_anomaly(self, ctx, partition, period, observation=None):
        dma_nodes = ctx.nodes_in(partition)
        if not dma_nodes:
            return {"predicted_dp": {s: 0.0 for s in ctx.sensors}, "factor": None,
                    "reproducible": False}
        best = None
        for f in self.factor_grid:
            ck = f"demand|{ctx.name}|{partition}|{f}|{period}"
            dp = ctx.twin.delta(dma_demand_modifier(dma_nodes, f), period, cache_key=ck)
            self._charge(1)
            if dp is None:
                continue
            pred = self._at_sensors(dp, ctx.sensors)
            err = 0.0 if observation is None else self._sse(observation, pred, ctx.sensors)
            if best is None or err < best[2]:
                best = (pred, f, err)
        if best is None:
            return {"predicted_dp": {s: 0.0 for s in ctx.sensors}, "factor": None,
                    "reproducible": False}
        pred, f, _ = best
        return {"predicted_dp": pred, "factor": f, "reproducible": True}

    # ---- valve / pipe mis-state -----------------------------------------
    def predict_valve_misstate(self, ctx, pipe_name, period):
        ck = f"valve|{ctx.name}|{pipe_name}|{period}"
        dp = ctx.twin.delta(close_pipe_modifier(pipe_name), period, cache_key=ck)
        self._charge(1)
        if dp is None:
            return {"predicted_dp": {s: 0.0 for s in ctx.sensors}, "reproducible": False}
        return {"predicted_dp": self._at_sensors(dp, ctx.sensors), "reproducible": True}

    # ---- sensor fault (measurement-space, no sim) ------------------------
    def predict_sensor_fault(self, ctx, faulty_sensors, observation):
        """A sensor fault explains exactly the faulty channels and predicts NO
        hydraulic anomaly elsewhere (predicted=obs on faulty, 0 on the rest)."""
        faulty = set(str(s) for s in faulty_sensors)
        pred = {s: (observation.get(s, 0.0) if s in faulty else 0.0) for s in ctx.sensors}
        return {"predicted_dp": pred, "faulty": sorted(faulty), "reproducible": True}

    def predict_none(self, ctx):
        return {"predicted_dp": {s: 0.0 for s in ctx.sensors}, "reproducible": True}
