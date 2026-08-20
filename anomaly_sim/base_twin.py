"""
HydraulicTwin: a thin, cross-network-safe WNTR/EPANET forward-simulation wrapper.

This is the falsification instrument of the whole system: the Executor tests
each competing hypothesis by injecting it here and comparing the predicted
sensor response to the observation.

Key fix vs the prior repo: leak injection no longer hard-codes the EXA7-only
demand pattern '881'. A leak is injected as a CONSTANT additional demand
(pattern=None), which is both physically appropriate (a fixed extra draw) and
portable to City H / L-Town. `leak_pattern` may be set per-network if a
pattern-modulated leak is ever desired.
"""
from __future__ import annotations
import copy
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import wntr

DEFAULT_PERIODS = {"night_low": 0.7, "day_normal": 1.0, "evening_peak": 1.2}


class HydraulicTwin:
    def __init__(self, inp_file: str, period_multipliers: dict | None = None,
                 leak_pattern=None):
        self.inp_file = inp_file
        self.wn = wntr.network.WaterNetworkModel(inp_file)
        self.period_multipliers = dict(period_multipliers or DEFAULT_PERIODS)
        self.leak_pattern = leak_pattern        # None => constant-demand leak
        self._baseline_cache: dict = {}
        self._delta_cache: dict = {}      # memoized (modifier-signature, period) -> delta dict
        self.junctions = list(self.wn.junction_name_list)
        self._junction_set = set(str(x) for x in self.junctions)
        self.sim_failures = 0

    # ---- internals --------------------------------------------------------
    def _scaled_copy(self, mult: float):
        wc = copy.deepcopy(self.wn)
        if mult != 1.0:
            for jn in wc.junction_name_list:
                for dem in wc.get_node(jn).demand_timeseries_list:
                    if dem.base_value is not None:
                        dem.base_value *= mult
        return wc

    @staticmethod
    def _avg_pressure(results) -> dict:
        return results.node["pressure"].mean(axis=0).to_dict()

    # Pressures beyond this (m) signal a disconnected/degenerate solve (e.g. a
    # pipe closure that isolates part of the network) -> treat as a sim failure.
    MAX_PHYSICAL_PRESSURE = 1.0e4

    def _run(self, wc):
        """Run EPANET; return avg-pressure dict or None on failure/degeneracy."""
        try:
            results = wntr.sim.EpanetSimulator(wc).run_sim()
            p = self._avg_pressure(results)
            vals = np.array(list(p.values()), dtype=float)
            if vals.size == 0 or not np.all(np.isfinite(vals)) \
                    or np.max(np.abs(vals)) > self.MAX_PHYSICAL_PRESSURE:
                self.sim_failures += 1
                return None
            return p
        except Exception:
            self.sim_failures += 1
            return None

    # ---- public API -------------------------------------------------------
    def baseline(self, period: str = "day_normal") -> dict:
        if period not in self._baseline_cache:
            mult = self.period_multipliers[period]
            self._baseline_cache[period] = self._run(self._scaled_copy(mult)) or {}
        return self._baseline_cache[period]

    def simulate(self, modifier=None, period: str = "day_normal"):
        """`modifier(wn_copy)` mutates a demand-scaled deepcopy in place."""
        wc = self._scaled_copy(self.period_multipliers[period])
        if modifier is not None:
            modifier(wc)
        return self._run(wc)

    def delta(self, modifier, period: str = "day_normal", cache_key=None):
        """Pressure delta (modified - baseline). Returns dict or None.
        If `cache_key` is given, the result is memoized (sims are deterministic)."""
        if cache_key is not None and cache_key in self._delta_cache:
            return self._delta_cache[cache_key]
        base = self.baseline(period)
        modp = self.simulate(modifier, period)
        out = None
        if modp is not None and base:
            out = {n: modp[n] - base[n] for n in base if n in modp}
        if cache_key is not None:
            self._delta_cache[cache_key] = out
        return out

    # ---- leak injection (constant additional demand) ----------------------
    def leak_modifier(self, node: str, rate_Ls: float):
        pat = self.leak_pattern

        def _mod(wc):
            # wntr's Demands.to_ts silently substitutes the model's DEFAULT
            # pattern when the pattern argument is None, so a "constant" leak
            # would inherit e.g. KY4's diurnal pattern '1'. Register an explicit
            # constant pattern instead (no-op on networks without a default).
            p = pat
            if p is None:
                if "LEAK_CONST" not in wc.pattern_name_list:
                    wc.add_pattern("LEAK_CONST", [1.0])
                p = "LEAK_CONST"
            obj = wc.get_node(node)
            obj.add_demand(rate_Ls / 1000.0, p, category="leak")
        return _mod

    def save_cache(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self._delta_cache, f)

    def load_cache(self, path):
        import os
        import pickle
        if os.path.exists(path):
            with open(path, "rb") as f:
                self._delta_cache.update(pickle.load(f))
        return len(self._delta_cache)

    def leak_delta(self, node: str, rate_Ls: float, period: str = "day_normal"):
        if str(node) not in self._junction_set:    # only junctions can leak
            return None
        ck = f"leak|{node}|{float(rate_Ls)}|{period}"
        return self.delta(self.leak_modifier(node, rate_Ls), period, cache_key=ck)
