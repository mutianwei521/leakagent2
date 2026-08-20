"""
City D controlled severity sweep: the companion to the register-anchored eval.

The register eval shows the system abstains on ~all REAL leaks because they are tiny
(median 0.094 L/s -> ~0.003 m signature, far below the 0.15 m noise floor). This sweep
asks the complementary question: on the SAME City D network, at what leak size does
the system start to localize? At each rate it injects a leak at a fixed, seeded set of
candidate nodes, adds field noise, and diagnoses -- giving the coverage / decision-
precision / forced-accuracy gradient vs leak severity. Overlaid with the real register's
severity distribution (in the figure), it shows the real leak population sits almost
entirely BELOW the rate at which pressure localization becomes reliable. This separates
a real-leak-SIZE limit from any system limitation.

Uses only cached candidate Δp (rates 2..50 L/s already in leak_cache.pkl) -> no new sims.

Run:  python eval/run_city_d_severity.py     (after the library build; via wds_rag python)
"""
import os
import sys
import json
import statistics
import warnings
warnings.filterwarnings("ignore")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from data.context import build_context
from schemas.contract import GoalContract
from tools.graphrag_tool import GraphRAGTool
from tools.digital_twin import DigitalTwinTool
from tools.residual_analysis import ResidualTool
from tools.bayesian import BayesianEngine
from agents.executor import ExecutorAgent
from agents.supervisor import SupervisorAgent
from run_diagnosis import diagnose

SENSOR_STD = 0.15
LEAK_CANDS = 25
RATES = (2.0, 5.0, 10.0, 20.0, 35.0, 50.0)   # cached candidate rates -> no new sims
N_TEST = 120                                  # seeded test nodes per rate
SEED = 11


def main():
    print("=" * 72)
    print(f"CITY_D CONTROLLED SEVERITY SWEEP (sensor_std={SENSOR_STD} m, {N_TEST} nodes/rate)")
    print("=" * 72)
    ctx = build_context(os.path.join(ROOT, "configs", "city_d.yaml"))
    gr = GraphRAGTool(ctx.cfg["sensor_fingerprints_file"], ctx.sensors)
    from eval.run_city_d import RATE_GRID     # same register-leg hypothesis grid
    ex = ExecutorAgent(ctx, gr, DigitalTwinTool(),
                       ResidualTool(sensor_std=SENSOR_STD, pass_threshold=3.0), BayesianEngine(),
                       top_k_leak=len(ctx.partitions), leak_cands=LEAK_CANDS,
                       rate_grid=RATE_GRID, noise_std=SENSOR_STD)
    sv = SupervisorAgent(GoalContract(min_leak_probability=0.5, min_top1_margin=0.10,
                                      max_candidate_region_nodes=80, alt_falsify_prob=0.30,
                                      max_residual_mahalanobis=3.0))
    ctx.twin.load_cache(os.path.join(ROOT, "data", "city_d", "leak_cache.pkl"))
    n2c = ctx.node_to_partition

    # fixed seeded test set drawn from the per-zone leak candidates (cached at all RATES)
    cand = []
    for z in ctx.partitions:
        cand += ctx.top_degree_nodes(z, LEAK_CANDS)
    cand = list(dict.fromkeys(cand))
    rng = np.random.RandomState(SEED)
    test_nodes = list(rng.choice(cand, size=min(N_TEST, len(cand)), replace=False))
    print(f"{len(test_nodes)} seeded test nodes from {len(cand)} candidates", flush=True)

    sweep = []
    for r in RATES:
        nrng = np.random.RandomState(SEED * 7919 + int(r * 10))
        acted = correct = forced = 0
        dps = []
        for node in test_nodes:
            dp = ctx.twin.leak_delta(node, r, "day_normal")   # cache hit
            if dp is None:
                continue
            clean = {s: float(dp.get(s, 0.0)) for s in ctx.sensors}
            dps.append(max((abs(v) for v in clean.values()), default=0.0))
            obs = {s: clean[s] + float(nrng.normal(0.0, SENSOR_STD)) for s in ctx.sensors}
            tz = n2c.get(str(node))
            res = diagnose(obs, ctx, ex, sv, "day_normal", case_id=f"{node}@{r}")
            ev = res.evidence
            # genuinely forced choice (always defined): retrieval top-1 zone,
            # matching the register leg's forced metric
            retrieval_top1 = int(gr.rank_partitions(obs, top_k=1)[0][0])
            if retrieval_top1 == tz:
                forced += 1
            if res.acted:
                acted += 1
                if res.accepted_partition == tz:
                    correct += 1
        m = len(test_nodes)
        sweep.append({"rate_Ls": r, "n": m,
                      "median_clean_max_dp_m": round(float(statistics.median(dps)) if dps else 0.0, 4),
                      "forced_top1_zone_acc": forced / m if m else 0.0,
                      "coverage": acted / m if m else 0.0,
                      "decision_precision_on_acted": (correct / acted) if acted else 1.0})
        print(f"  {r:5.1f} L/s: med|Δp|={sweep[-1]['median_clean_max_dp_m']:.4f}m "
              f"forced={sweep[-1]['forced_top1_zone_acc']*100:4.0f}% "
              f"cover={sweep[-1]['coverage']*100:4.0f}% "
              f"prec={sweep[-1]['decision_precision_on_acted']*100:4.0f}%", flush=True)

    out = {"provenance": {"network": "City D municipal WDN", "model": "nominal dataset/city_d.inp",
                          "data": "controlled monosize leak sweep on candidate nodes; "
                                  "twin Δp (cached) + Gaussian field noise (sigma=0.15 m)",
                          "purpose": "detectability gradient vs leak severity, to contrast with the "
                                     "real register's tiny-leak distribution"},
           "sensor_std_m": SENSOR_STD, "n_test_nodes": len(test_nodes), "sweep": sweep}
    json.dump(out, open(os.path.join(ROOT, "artifacts", "results_city_d_severity.json"), "w",
                        encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\nsaved -> artifacts/results_city_d_severity.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
