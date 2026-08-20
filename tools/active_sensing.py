"""
ActiveSensingTool (T5): pick the next measurement(s) that best DISCRIMINATE the
surviving leak-partition hypotheses. We choose hidden nodes where the twin
predicts the largest spread of pressure response across the competing
hypotheses — the measurement most likely to resolve a wrong-partition ambiguity.
(A digital-twin surrogate of expected-information-gain; see roadmap 3.4.)
"""
from __future__ import annotations

from .base import Tool


class ActiveSensingTool(Tool):
    name = "active_sensing"
    cost = 0.0

    def discriminative_nodes(self, ctx, leak_hyps, period, n_extra=3, pool_cap=300):
        # predicted full-node response of each competing leak hypothesis (cached)
        preds = []
        for h in leak_hyps:
            fn, fr = h.params.get("fit_node"), h.params.get("fit_rate")
            if fn is None or fr is None:
                continue
            dp = ctx.twin.leak_delta(fn, fr, period)
            if dp:
                preds.append(dp)
        if len(preds) < 2:
            return []
        sensors = set(ctx.sensors)
        pids = {h.partition for h in leak_hyps}
        pool, seen = [], set()
        for p in pids:
            for n in ctx.nodes_in(p):
                if n not in sensors and n not in seen:
                    seen.add(n)
                    pool.append(n)
        pool = pool[:pool_cap]
        scored = []
        for x in pool:
            vals = [dp.get(x, 0.0) for dp in preds]
            scored.append((x, max(vals) - min(vals)))   # spread across hypotheses
        scored.sort(key=lambda t: -t[1])
        return [x for x, s in scored[:n_extra] if s > 1e-6]
