# -*- coding: utf-8 -*-
"""
Standard in-silico protocol on an arbitrary onboarded network (city_h, city_d):
the identical scenario generator, noise model (sigma = 0.05 m), seeds and
metrics as the EXA7 headline run and the KY4 transfer (leak_cands = 25,
matching the persistent per-network leak-response caches).

Run:  <wds_rag python> eval/run_standard_net.py <net>
Output: artifacts/results_<net>_standard.json
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
from eval.run_experiments import generate, stat_row, leak_acc, op_point, agg, PERIOD

NOISE, SEEDS, LEAK_CANDS = 0.05, [17, 42, 101, 2025, 31337], 25


def run_once(ctx, gr, dt, asens, noise, seed):
    rt = ResidualTool(sensor_std=noise, pass_threshold=3.0)
    ex = ExecutorAgent(ctx, gr, dt, rt, BayesianEngine(), top_k_leak=len(ctx.partitions),
                       leak_cands=LEAK_CANDS, noise_std=noise)
    rng = np.random.default_rng(seed)
    arng = np.random.default_rng(seed + 1)
    scens = generate(ctx, gr, rng, noise)
    rows_pre, rows = [], []
    for s in scens:
        ev = EvidencePackage(observation=s["observation"], demand_period=PERIOD, network=ctx.name)
        ex.step(ev, 1)
        rows_pre.append(stat_row(ev, gr, s))
        ex.active_sense_verify(ev, s.get("full_dp", {}), PERIOD, arng, asens, n_extra=3, top_k=6)
        rows.append(stat_row(ev, gr, s))
    return rows_pre, rows


def main():
    net = sys.argv[1] if len(sys.argv) > 1 else "city_h"
    print("=" * 70)
    print(f"STANDARD IN-SILICO PROTOCOL: {net} (sigma={NOISE}, seeds={SEEDS})")
    print("=" * 70)
    ctx = build_context(os.path.join(ROOT, "configs", f"{net}.yaml"))
    gr = GraphRAGTool(ctx.cfg["sensor_fingerprints_file"], ctx.sensors)
    dt, asens = DigitalTwinTool(), ActiveSensingTool()

    per = {"forced_retrieval_top1": [], "leak_partition_acc_no_active": [],
           "leak_partition_acc_with_active": [], "decision_precision_at_maxcov": [],
           "coverage_at_maxcov": [], "coverage_at_100pct_precision": [],
           "false_dispatch_at_maxcov": []}
    n_scen = None
    for sd in SEEDS:
        rows_pre, rows = run_once(ctx, gr, dt, asens, NOISE, sd)
        n_scen = len(rows)
        ret = float(np.mean([1.0 if r["retrieval_top1"] == r["true_region"] else 0.0
                             for r in rows if r["true_class"] == "leak"]))
        op = op_point(rows)
        per["forced_retrieval_top1"].append(ret)
        per["leak_partition_acc_no_active"].append(leak_acc(rows_pre))
        per["leak_partition_acc_with_active"].append(leak_acc(rows))
        per["decision_precision_at_maxcov"].append(op["precision"])
        per["coverage_at_maxcov"].append(op["coverage"])
        per["coverage_at_100pct_precision"].append(op["coverage_at_100prec"])
        per["false_dispatch_at_maxcov"].append(op["false_dispatch"])
        print(f"  seed {sd}: retrieval {ret*100:.1f}%  twin-fit {leak_acc(rows)*100:.1f}%  "
              f"precision {op['precision']*100:.1f}% @ {op['coverage']*100:.1f}%", flush=True)
        for f in ("temp.bin", "temp.inp", "temp.rpt"):
            try:
                os.remove(os.path.join(ROOT, f))
            except OSError:
                pass

    out = {"provenance": {"network": net, "n_junctions": len(ctx.node_to_partition),  # all partitioned nodes incl. sources (SI reports as "nodes")
                          "partition_k": len(ctx.partitions), "n_sensors": len(ctx.sensors),
                          "n_scenarios_per_seed": n_scen, "noise_m": NOISE, "seeds": SEEDS,
                          "leak_cands": LEAK_CANDS,
                          "protocol": "identical generator/metrics to eval/run_experiments.py"},
           "multiseed_sigma0.05": {k: agg(v) for k, v in per.items()}}
    path = os.path.join(ROOT, "artifacts", f"results_{net}_standard.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=2)
    print(f"\nsaved -> artifacts/results_{net}_standard.json")
    ms = out["multiseed_sigma0.05"]
    for k, v in ms.items():
        print(f"  {k:34s}: {v['mean']*100:5.1f} ± {v['std']*100:4.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
