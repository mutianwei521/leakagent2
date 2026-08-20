"""
BayesianEngine (T4): a proper posterior over the competing-hypothesis space.

  log P(h | E) = log prior(h) + sum_tools loglik_tool(h)   (then softmax-normalize)

The log-likelihoods come from the digital-twin + residual fit of each hypothesis
(physically distinct measurements -> conditional independence is a reasonable
approximation; see roadmap risk notes). The existence mass = total leak-class
posterior is the quantity the Supervisor's existence gate (G1) reads.
"""
from __future__ import annotations
import math
from collections import defaultdict


class BayesianEngine:
    DEFAULT_BASE_RATES = {
        "leak": 0.50, "demand_anomaly": 0.20, "sensor_fault": 0.15,
        "valve_misstate": 0.10, "none": 0.05,
    }

    def __init__(self, base_rates=None):
        self.base_rates = dict(self.DEFAULT_BASE_RATES)
        if base_rates:
            self.base_rates.update(base_rates)

    def assign_priors(self, hypotheses):
        groups = defaultdict(list)
        for h in hypotheses:
            groups[h.kind].append(h)
        for kind, hs in groups.items():
            share = self.base_rates.get(kind, 0.05) / max(1, len(hs))
            for h in hs:
                h.prior = share
        total = sum(h.prior for h in hypotheses) or 1.0
        for h in hypotheses:
            h.prior /= total

    def update(self, hypotheses, loglik: dict):
        logpost = {}
        for h in hypotheses:
            logpost[h.key()] = math.log(max(h.prior, 1e-12)) + float(loglik.get(h.key(), 0.0))
        mx = max(logpost.values())
        exps = {k: math.exp(v - mx) for k, v in logpost.items()}
        z = sum(exps.values()) or 1.0
        for h in hypotheses:
            h.posterior = exps[h.key()] / z

    @staticmethod
    def summarize(hypotheses):
        s = sorted(hypotheses, key=lambda h: -h.posterior)
        top1 = s[0] if s else None
        top2 = s[1] if len(s) > 1 else None
        margin = (top1.posterior - (top2.posterior if top2 else 0.0)) if top1 else 0.0
        existence = sum(h.posterior for h in hypotheses if h.kind == "leak")
        ent = -sum(h.posterior * math.log(max(h.posterior, 1e-12)) for h in hypotheses)
        return existence, top1, top2, margin, ent / math.log(2)
