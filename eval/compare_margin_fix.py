# -*- coding: utf-8 -*-
"""Quantify the effect of the active-sensing margin fix on the EXA7 headline
pipeline: same scenarios, same seeds, legacy vs. self-consistent package.

Reports, per seed and pooled: leak-partition accuracy, decision precision and
coverage at the operating point (max-coverage end of the residual sweep), the
100%-precision coverage, and how many acted/abstain decisions flip.
Run:  <wds_rag python> eval/compare_margin_fix.py [n_seeds]
"""
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from data.context import build_context
from schemas.evidence import EvidencePackage
from tools.graphrag_tool import GraphRAGTool
from tools.digital_twin import DigitalTwinTool
from tools.residual_analysis import ResidualTool
from tools.bayesian import BayesianEngine
from tools.active_sensing import ActiveSensingTool
from agents.executor import ExecutorAgent
import eval.metrics as M
from eval.run_experiments import generate, stat_row, PERIOD, leak_acc, op_point, SEEDS

NOISE = 0.05


def run(ctx, gr, dt, asens, seed, legacy):
    rt = ResidualTool(sensor_std=NOISE, pass_threshold=3.0)
    ex = ExecutorAgent(ctx, gr, dt, rt, BayesianEngine(), top_k_leak=len(ctx.partitions),
                       leak_cands=50, noise_std=NOISE)
    rng = np.random.default_rng(seed)
    arng = np.random.default_rng(seed + 1)
    rows = []
    for s in generate(ctx, gr, rng, NOISE):
        ev = EvidencePackage(observation=s["observation"], demand_period=PERIOD, network=ctx.name)
        ex.step(ev, 1)
        ex.active_sense_verify(ev, s.get("full_dp", {}), PERIOD, arng, asens,
                               n_extra=3, top_k=6, legacy_margin=legacy)
        rows.append(stat_row(ev, gr, s))
    return rows


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    ctx = build_context(os.path.join(ROOT, "configs", "exa7.yaml"))
    gr = GraphRAGTool(ctx.cfg["sensor_fingerprints_file"], ctx.sensors)
    dt = DigitalTwinTool()
    asens = ActiveSensingTool()
    out = {"seeds": [], "legacy": [], "fixed": [], "flips": []}
    for seed in SEEDS[:n]:
        L = run(ctx, gr, dt, asens, seed, legacy=True)
        F = run(ctx, gr, dt, asens, seed, legacy=False)
        opL, opF = op_point(L), op_point(F)
        # decisions at the reported gate (0.5/0.12/0.20/60, max-coverage residual gate of each run)
        # decisions at the fixed pre-registered gate of the paper (0.5/0.12/0.20/60, rho<=3)
        actL = [M.gate_act(r, 0.5, 0.12, 3.0, 0.20, 60) for r in L]
        actF = [M.gate_act(r, 0.5, 0.12, 3.0, 0.20, 60) for r in F]
        gL = gF = 3.0
        flips = sum(int(a != b) for a, b in zip(actL, actF))
        rec = lambda rows, op: {"leak_acc": round(leak_acc(rows), 4),
                                "precision": round(op["precision"], 4), "coverage": round(op["coverage"], 4),
                                "cov_at_100": round(op["coverage_at_100prec"], 4),
                                "gate": gL if rows is L else gF}
        out["seeds"].append(seed)
        out["legacy"].append(rec(L, opL))
        out["fixed"].append(rec(F, opF))
        out["flips"].append(flips)
        print(f"seed {seed}: legacy {out['legacy'][-1]}  |  fixed {out['fixed'][-1]}  |  decisions flipped {flips}/{len(L)}")
    for k in ("leak_acc", "precision", "coverage", "cov_at_100"):
        a = np.array([x[k] for x in out["legacy"]]); b = np.array([x[k] for x in out["fixed"]])
        print(f"{k:12s} legacy {a.mean():.4f} +- {a.std():.4f}   fixed {b.mean():.4f} +- {b.std():.4f}")
    json.dump(out, open(os.path.join(ROOT, "artifacts", "results_margin_fix_comparison.json"), "w"), indent=2)
    print("saved -> artifacts/results_margin_fix_comparison.json")


if __name__ == "__main__":
    main()
