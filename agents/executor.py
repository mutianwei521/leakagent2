"""
ExecutorAgent: the hydraulic diagnostic agent.

Round 1 seeds COMPETING hypotheses (leak / demand-anomaly / sensor-fault /
valve-misstate / none) and, for each, predicts the sensor response in the
digital twin and scores the residual -> a Gaussian log-likelihood. The Bayesian
engine then fuses priors + likelihoods into a posterior over the full space.

This is a deterministic, reproducible core. An LLM can later ORCHESTRATE tool
choice / hypothesis phrasing on top of it (llm_client), but the numbers always
come from the twin + residual, never from the LLM — the structural guard
against fabricated metrics.
"""
from __future__ import annotations
import math
import numpy as np

from schemas.evidence import Hypothesis, ToolResult


def _k_params(h) -> int:
    """Effective # of fitted free parameters (for a BIC/Occam complexity penalty).
    The leak fits node+rate (2); demand fits a factor (1); a valve/none is a
    discrete hypothesis with no continuous fit (0); a sensor fault 'explains' one
    free reading per masked channel."""
    if h.kind == "leak":
        return 2
    if h.kind == "demand_anomaly":
        return 1
    if h.kind == "sensor_fault":
        return len(h.params.get("faulty", []))
    return 0


class ExecutorAgent:
    def __init__(self, ctx, graphrag, twin_tool, residual_tool, bayesian,
                 top_k_leak=3, rate_grid=(2.0, 5.0, 10.0, 20.0, 35.0, 50.0), leak_cands=6,
                 sensor_anom_sigmas=4.0, max_sensor_fault=3, noise_std=None,
                 differential=True, occam=True):
        self.ctx = ctx
        self.differential = differential   # if False: leak/none only (ablation)
        self.occam = occam                 # if False: no BIC penalty (ablation)
        self.graphrag = graphrag
        self.twin = twin_tool
        self.residual = residual_tool
        self.bayesian = bayesian
        self.top_k_leak = top_k_leak
        self.rate_grid = tuple(rate_grid)
        self.leak_cands = leak_cands
        self.sensor_anom_sigmas = sensor_anom_sigmas
        self.max_sensor_fault = max_sensor_fault
        self.noise_std = noise_std if noise_std is not None else ctx.noise_sensor_std
        self._plan = None      # optional per-case LLM plan: which non-leak families to test

    def set_plan(self, plan):
        self._plan = plan

    # ---- hypothesis seeding ----------------------------------------------
    def _valve_candidates(self, pid, limit=2):
        if pid is None:
            return []
        wn, n2c, out = self.ctx.twin.wn, self.ctx.node_to_partition, []
        for pn in wn.pipe_name_list:
            l = wn.get_link(pn)
            if n2c.get(l.start_node_name) == pid or n2c.get(l.end_node_name) == pid:
                out.append(pn)
            if len(out) >= limit:
                break
        return out

    def seed(self, ev):
        obs = ev.observation
        ranked = self.graphrag.rank_partitions(obs, top_k=self.top_k_leak)
        hyps = []
        for pid, _ in ranked:
            kb = self.graphrag.candidate_nodes(pid, max_nodes=2)
            deg = self.ctx.top_degree_nodes(pid, self.leak_cands)
            cand = list(dict.fromkeys([str(c) for c in kb] + [str(d) for d in deg]))
            hyps.append(Hypothesis(kind="leak", partition=int(pid),
                                   node=(cand[0] if cand else None),
                                   params={"candidate_nodes": cand}))
        if self.differential:        # non-leak hypotheses (the differential head)
            p = self._plan or {}
            if p.get("include_demand", True):
                for pid, _ in ranked:    # a demand-anomaly alternative for every retrieved zone
                    hyps.append(Hypothesis(kind="demand_anomaly", partition=int(pid)))
            if p.get("include_sensor", True):
                sig = self.noise_std
                anom = sorted([s for s in self.ctx.sensors
                               if abs(obs.get(s, 0.0)) > self.sensor_anom_sigmas * sig],
                              key=lambda s: -abs(obs.get(s, 0.0)))
                if 1 <= len(anom):
                    faulty = anom[: self.max_sensor_fault]
                    hyps.append(Hypothesis(kind="sensor_fault", node=",".join(faulty),
                                           params={"faulty": faulty}))
            if p.get("include_valve", True):
                top_pid = ranked[0][0] if ranked else None
                for pp in self._valve_candidates(top_pid, limit=2):
                    hyps.append(Hypothesis(kind="valve_misstate", node=str(pp)))
        hyps.append(Hypothesis(kind="none"))
        self.bayesian.assign_priors(hyps)
        ev.hypotheses = hyps
        return ev

    # ---- one diagnostic round --------------------------------------------
    def step(self, ev, round_idx=1):
        if not ev.hypotheses:
            self.seed(ev)
        obs = ev.observation
        period = ev.demand_period or "day_normal"
        sigma = self.noise_std
        loglik = {}
        for h in ev.hypotheses:
            if h.kind == "leak":
                pred = self.twin.predict_leak_partition(
                    self.ctx, h.params.get("candidate_nodes") or ([h.node] if h.node else []),
                    period, observation=obs, rate_grid=self.rate_grid)
                h.params["fit_rate"] = pred.get("rate")
                h.params["fit_node"] = pred.get("node")
                r = self.residual.analyze(obs, pred["predicted_dp"], self.ctx.sensors,
                                          sensor_std=sigma)
            elif h.kind == "demand_anomaly":
                pred = self.twin.predict_demand_anomaly(self.ctx, h.partition, period,
                                                        observation=obs)
                r = self.residual.analyze(obs, pred["predicted_dp"], self.ctx.sensors,
                                          sensor_std=sigma)
            elif h.kind == "valve_misstate":
                pred = self.twin.predict_valve_misstate(self.ctx, h.node, period)
                r = self.residual.analyze(obs, pred["predicted_dp"], self.ctx.sensors,
                                          sensor_std=sigma)
            elif h.kind == "sensor_fault":
                pred = self.twin.predict_sensor_fault(self.ctx, h.params.get("faulty", []), obs)
                r = self.residual.analyze(obs, pred["predicted_dp"], self.ctx.sensors,
                                          sensor_std=sigma)
            else:  # none
                pred = self.twin.predict_none(self.ctx)
                r = self.residual.analyze(obs, pred["predicted_dp"], self.ctx.sensors,
                                          sensor_std=sigma)
            h.params["residual"] = r
            # BIC/Occam penalty: more-flexible hypotheses must fit substantially
            # better to win (stops an over-parameterized leak from mimicking a
            # DMA demand anomaly at sparse sensors).
            penalty = (0.5 * _k_params(h) * math.log(max(len(self.ctx.sensors), 2))
                       if self.occam else 0.0)
            h.params["bic_penalty"] = penalty
            loglik[h.key()] = r["loglik"] - penalty

        self.bayesian.update(ev.hypotheses, loglik)
        existence, top1, top2, margin, ent = self.bayesian.summarize(ev.hypotheses)
        ev.existence_mass = existence
        ev.top1, ev.top2, ev.margin, ev.entropy_bits = top1, top2, margin, ent
        ev.candidate_region = (self.ctx.nodes_in(top1.partition)
                               if top1 and top1.kind == "leak" else [])
        # mark alternatives that the residual physically rules out
        for h in ev.hypotheses:
            if h is top1:
                continue
            rr = h.params.get("residual", {})
            if not rr.get("passed", True):
                h.status = "falsified"
                h.falsified_by = [f"residual_mahal/dof={rr.get('mahalanobis_per_dof', 0):.1f}"]
        ev.rounds = round_idx
        ev.add_tool_result(ToolResult(
            tool="executor_round", cost=0.0,
            payload={"ranked": self.graphrag.rank_partitions(obs, top_k=5),
                     "existence": round(existence, 3), "margin": round(margin, 3)},
            loglik={k: round(v, 2) for k, v in loglik.items()}, round_idx=round_idx))
        return ev

    def incorporate_feedback(self, required_evidence):
        # hook for active-sensing-driven multi-round (P4); deterministic core is 1 round
        self._last_feedback = required_evidence

    def active_sense_verify(self, ev, full_dp, period, rng, active_tool,
                            n_extra=3, top_k=6, strategy="discriminative",
                            legacy_margin=False):
        """P4 verification round: reveal the most discriminative hidden node(s) and
        RE-RANK the surviving leak partitions over the augmented sensor set. This
        resolves wrong-partition ambiguities that the residual gate cannot (both
        zones fit the 30 sensors well, but disagree at the revealed node).
        strategy="random" is the ablation control: reveal n_extra random hidden
        nodes instead of the max-discriminability choice."""
        if ev.top1 is None or ev.top1.kind != "leak":
            return ev
        leaks = sorted([h for h in ev.hypotheses if h.kind == "leak"],
                       key=lambda h: -h.posterior)[:top_k]
        if len(leaks) < 2:
            return ev
        if strategy == "random":
            hidden = sorted(set(full_dp) - set(self.ctx.sensors))
            extra = (list(rng.choice(hidden, size=min(n_extra, len(hidden)),
                                     replace=False)) if hidden else [])
        else:
            extra = active_tool.discriminative_nodes(self.ctx, leaks, period, n_extra=n_extra)
        if not extra:
            return ev
        aug = list(self.ctx.sensors) + extra
        obs = dict(ev.observation)
        for x in extra:
            obs[x] = float(full_dp.get(x, 0.0) + rng.normal(0.0, self.noise_std))

        scored = []
        for h in leaks:
            pred = self.twin.predict_leak_partition(
                self.ctx, h.params.get("candidate_nodes") or ([h.node] if h.node else []),
                period, observation=obs, rate_grid=self.rate_grid, sensors=aug)
            r = self.residual.analyze(obs, pred["predicted_dp"], aug, sensor_std=self.noise_std)
            ll = r["loglik"] - 0.5 * 2 * math.log(max(len(aug), 2))
            h.params["fit_node"], h.params["fit_rate"] = pred.get("node"), pred.get("rate")
            h.params["residual"] = r
            scored.append((h, ll, r))
        mx = max(s[1] for s in scored)
        exps = [math.exp(s[1] - mx) for s in scored]
        z = sum(exps) or 1.0
        post = [e / z for e in exps]
        order = sorted(range(len(scored)), key=lambda i: -post[i])
        best = scored[order[0]][0]
        second_post = post[order[1]] if len(order) > 1 else 0.0
        if legacy_margin:
            # pre-2026-08 behaviour, kept only for like-for-like comparison: the
            # margin became a within-leak quantity while the hypotheses' posteriors
            # stayed at their pre-sensing values, so the package was no longer
            # self-consistent and G3 no longer measured "any family".
            ev.top1 = best
            ev.margin = post[order[0]] - second_post
        else:
            # Active sensing re-fits the LEAK hypotheses only, so it may only move
            # mass WITHIN the leak family: keep the family total (existence mass,
            # set by the non-leak evidence) fixed and redistribute it in proportion
            # to the re-fitted within-leak posteriors; leaks outside the top_k
            # re-fit set are set to zero (they were dominated before sensing and
            # received no new evidence). Then top1/top2/margin are re-derived over
            # the FULL hypothesis space, so margin is again P(top1) - P(best of any
            # family), as Eq. S5 and predicate G3 define it, and the evidence
            # package the supervisor and auditor read is self-consistent.
            leak_mass = sum(h.posterior for h in ev.hypotheses if h.kind == "leak")
            refit = {id(s[0]) for s in scored}
            for h in ev.hypotheses:
                if h.kind == "leak" and id(h) not in refit:
                    h.posterior = 0.0
            for i, (h, _ll, _r) in enumerate(scored):
                h.posterior = leak_mass * post[i]
            existence, top1, top2, margin, ent = self.bayesian.summarize(ev.hypotheses)
            ev.existence_mass = existence
            ev.top1, ev.top2, ev.margin, ev.entropy_bits = top1, top2, margin, ent
            best = ev.top1 if (ev.top1 is not None and ev.top1.kind == "leak") else best
        ev.candidate_region = self.ctx.nodes_in(best.partition)
        ev.safety["active_sensors_used"] = len(extra)
        ev.add_tool_result(ToolResult(tool="active_sensing", cost=0.0,
                                      payload={"revealed": extra,
                                               "new_top1": f"P{best.partition}",
                                               "margin": round(ev.margin, 3)},
                                      round_idx=2))
        return ev
