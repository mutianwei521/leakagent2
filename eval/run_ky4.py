"""
Second in-silico network: KY4 (public Kentucky benchmark, 959 junctions),
evaluated with the identical pipeline, scenario generator, noise model
(sigma = 0.05 m) and metrics as the EXA7 headline run (eval/run_experiments.py).
The only differences are the network itself and the leak-candidate breadth
(leak_cands = 25 per zone, matching the persistent leak-response cache built by
data/ky4_build_library.py, the same setting as the L-Town and City H legs).

Run:  python eval/run_ky4.py    (via the wds_rag interpreter; needs the cache)
Output: artifacts/results_ky4.json
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
from eval import metrics as M
from eval.run_experiments import generate, stat_row, leak_acc, op_point, agg, PERIOD

NOISE, SEEDS, LEAK_CANDS = 0.05, [17, 42, 101, 2025, 31337], 25
CACHE = os.path.join(ROOT, "data", "ky4", "leak_cache.pkl")


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
    print("=" * 70)
    print("SECOND IN-SILICO NETWORK: KY4 (public benchmark, sigma=0.05)")
    print("=" * 70)
    ctx = build_context(os.path.join(ROOT, "configs", "ky4.yaml"))
    gr = GraphRAGTool(ctx.cfg["sensor_fingerprints_file"], ctx.sensors)
    dt, asens = DigitalTwinTool(), ActiveSensingTool()
    nload = ctx.twin.load_cache(CACHE)
    print(f"loaded twin leak library: {nload} cached sims.", flush=True)

    leak_pre, leak_post, prec, cov, cov100, fd, retr = [], [], [], [], [], [], []
    n_scen = None
    for sd in SEEDS:
        rp, rw = run_once(ctx, gr, dt, asens, NOISE, sd)
        n_scen = len(rw)
        leak_pre.append(leak_acc(rp))
        leak_post.append(leak_acc(rw))
        op = op_point(rw)
        prec.append(op["precision"])
        cov.append(op["coverage"])
        cov100.append(op["coverage_at_100prec"])
        fd.append(op["false_dispatch"])
        retr.append(M.forced_top1_accuracy(rw)["top1_accuracy"])
        print(f"   seed {sd}: retr {retr[-1]*100:.1f}%  leak_acc {leak_acc(rw)*100:.1f}%  "
              f"prec@maxcov {op['precision']*100:.1f}%  cov {op['coverage']*100:.1f}%  "
              f"cov@100%prec {op['coverage_at_100prec']*100:.1f}%", flush=True)

    out = {"provenance": {"network": "ky4", "source": "Kentucky research database of WDS models, "
                          "vendored from the pip-installed WNTR package",
                          "n_junctions": int(ctx.twin.wn.num_junctions),
                          "n_pipes": int(ctx.twin.wn.num_pipes),
                          "partition_k": len(ctx.partitions),
                          "n_sensors": len(ctx.sensors),
                          "n_scenarios_per_seed": n_scen,
                          "noise_m": NOISE, "seeds": SEEDS, "leak_cands": LEAK_CANDS,
                          "protocol": "identical generator/metrics to eval/run_experiments.py"},
           "multiseed_sigma0.05": {
               "forced_retrieval_top1": agg(retr),
               "leak_partition_acc_no_active": agg(leak_pre),
               "leak_partition_acc_with_active": agg(leak_post),
               "decision_precision_at_maxcov": agg(prec),
               "coverage_at_maxcov": agg(cov),
               "coverage_at_100pct_precision": agg(cov100),
               "false_dispatch_at_maxcov": agg(fd)}}
    json.dump(out, open(os.path.join(ROOT, "artifacts", "results_ky4.json"), "w",
                        encoding="utf-8"), indent=2)

    ms = out["multiseed_sigma0.05"]
    print(f"\n--- KY4 HEADLINE (mean +/- s.d. over {len(SEEDS)} seeds, sigma={NOISE}) ---")
    for k in ("forced_retrieval_top1", "leak_partition_acc_no_active",
              "leak_partition_acc_with_active", "decision_precision_at_maxcov",
              "coverage_at_maxcov", "coverage_at_100pct_precision"):
        v = ms[k]
        print(f"  {k:34s}: {v['mean']*100:5.1f} +/- {v['std']*100:4.1f}%")
    print("\nsaved -> artifacts/results_ky4.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
