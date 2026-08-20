"""
Extended statistics (real runs, EXA7): per-severity localization, seed-aggregated
differential confusion, pooled McNemar vs retrieval, and the no-differential-head
ablation (quantifies the false dispatches the differential diagnosis prevents).

Run:  python eval/run_full_stats.py     (via the wds_rag interpreter)
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
from anomaly_sim.leak_anomaly import make_leak_scenario
from anomaly_sim.demand_anomaly import make_demand_anomaly_scenario
from anomaly_sim.sensor_fault import make_sensor_fault_scenario
from anomaly_sim.valve_misstate import make_valve_misstate_scenario
import eval.metrics as M

PERIOD, NOISE = "day_normal", 0.05
SEEDS = [17, 42, 101, 2025, 31337]
LEAK_RATES = [2.0, 5.0, 10.0, 20.0, 35.0, 50.0]
CLASSES = ["leak", "demand_anomaly", "sensor_fault", "valve_misstate", "none"]


def gen_scenarios(ctx, gr, rng):
    sc = []
    parts = ctx.partitions[:10]
    for pid in parts:
        kb = gr.candidate_nodes(pid, max_nodes=1)
        pn = ctx.partition_nodes_by_degree.get(pid, [])
        nodes = [n for n in [kb[0] if kb else None, str(rng.choice(pn)) if pn else None] if n]
        for node in nodes:
            for r in LEAK_RATES:
                s = make_leak_scenario(ctx.twin, ctx.node_to_partition, ctx.sensors, node, r,
                                       PERIOD, noise_std=NOISE, rng=rng)
                if s:
                    sc.append(s)
    for pid in parts:
        for f in (1.6, 2.2):
            s = make_demand_anomaly_scenario(ctx.twin, ctx.node_to_partition, ctx.sensors, pid,
                                             factor=f, period=PERIOD, noise_std=NOISE, rng=rng)
            if s:
                sc.append(s)
    for _ in range(15):
        k = int(rng.integers(1, 3))
        faulty = list(rng.choice(ctx.sensors, size=k, replace=False))
        sc.append(make_sensor_fault_scenario(ctx.twin, ctx.node_to_partition, ctx.sensors, faulty,
                  mode=str(rng.choice(["bias", "stuck"])), magnitude=float(rng.choice([0.5, 1.0, 1.5])),
                  period=PERIOD, noise_std=NOISE, rng=rng))
    pipes = list(ctx.twin.wn.pipe_name_list); rng.shuffle(pipes); got = 0
    for pn in pipes:
        if got >= 15:
            break
        s = make_valve_misstate_scenario(ctx.twin, ctx.node_to_partition, ctx.sensors, pn,
                                         period=PERIOD, noise_std=NOISE, rng=rng)
        if s and s["max_abs_dp"] < 200.0:
            sc.append(s); got += 1
    return sc


def row(ev, gr, s):
    t1 = ev.top1
    rr = (t1.params.get("residual") if t1 else {}) or {}
    return {"true_class": s["true_class"], "true_region": int(s.get("true_region", -1)),
            "rate": s.get("rate_Ls"), "top1_kind": t1.kind if t1 else "none",
            "pred_partition": int(t1.partition) if t1 and t1.kind == "leak" else None,
            "retrieval_top1": int(gr.rank_partitions(ev.observation, 1)[0][0]),
            "existence": float(ev.existence_mass), "margin": float(ev.margin),
            "top1_mahal": float(rr.get("mahalanobis_per_dof", 999.0)),
            "best_alt": float(max((h.posterior for h in ev.hypotheses if h is not t1), default=0.0)),
            "region": int(len(ev.candidate_region))}


def run_variant(ctx, gr, dt, asens, sc, seed, differential):
    rt = ResidualTool(sensor_std=NOISE, pass_threshold=3.0)
    ex = ExecutorAgent(ctx, gr, dt, rt, BayesianEngine(), top_k_leak=len(ctx.partitions),
                       leak_cands=50, noise_std=NOISE, differential=differential)
    arng = np.random.default_rng(seed + 1)
    rows = []
    for s in sc:
        ev = EvidencePackage(observation=s["observation"], demand_period=PERIOD, network=ctx.name)
        ex.step(ev, 1)
        ex.active_sense_verify(ev, s.get("full_dp", {}), PERIOD, arng, asens)
        rows.append(row(ev, gr, s))
    return rows


def acted(r):
    return (r["top1_kind"] == "leak" and r["existence"] >= 0.5 and r["margin"] >= 0.12
            and r["best_alt"] <= 0.20 and r["top1_mahal"] <= 3.0)


def main():
    print("=" * 70)
    print("EXTENDED STATISTICS (real runs, EXA7, σ=0.05)")
    print("=" * 70)
    ctx = build_context(os.path.join(ROOT, "configs", "exa7.yaml"))
    gr = GraphRAGTool(ctx.cfg["sensor_fingerprints_file"], ctx.sensors)
    dt, asens = DigitalTwinTool(), ActiveSensingTool()

    full_rows, nodiff_rows = [], []
    persev_by_seed = {}   # rate -> [per-seed leak-partition accuracy], for the figure dots
    for sd in SEEDS:
        sc = gen_scenarios(ctx, gr, np.random.default_rng(sd))
        fr = run_variant(ctx, gr, dt, asens, sc, sd, True)
        full_rows += fr
        nodiff_rows += run_variant(ctx, gr, dt, asens, sc, sd, False)
        pv = {}
        for r in [x for x in fr if x["true_class"] == "leak"]:
            pv.setdefault(r["rate"], [0, 0]); pv[r["rate"]][1] += 1
            if r["top1_kind"] == "leak" and r["pred_partition"] == r["true_region"]:
                pv[r["rate"]][0] += 1
        for rate, (c, n) in pv.items():
            persev_by_seed.setdefault(rate, []).append(c / n if n else 0.0)
        print(f"  seed {sd} done ({len(sc)} scenarios)")

    # per-severity leak accuracy (pooled over seeds)
    persev = {}
    for r in [x for x in full_rows if x["true_class"] == "leak"]:
        persev.setdefault(r["rate"], [0, 0])
        persev[r["rate"]][1] += 1
        if r["top1_kind"] == "leak" and r["pred_partition"] == r["true_region"]:
            persev[r["rate"]][0] += 1
    persev = {f"{k} L/s": {"acc": c / n, "n": n, "values": persev_by_seed.get(k, [])}
              for k, (c, n) in sorted(persev.items())}

    # seed-aggregated 4x5 differential confusion (pooled counts)
    idx = {c: i for i, c in enumerate(CLASSES)}
    conf = [[0] * len(CLASSES) for _ in CLASSES]
    for r in full_rows:
        conf[idx[r["true_class"]]][idx[r["top1_kind"]]] += 1

    # pooled McNemar vs retrieval on leak subset
    n10 = n01 = 0
    for r in [x for x in full_rows if x["true_class"] == "leak"]:
        sysok = acted(r) and r["pred_partition"] == r["true_region"]
        retok = r["retrieval_top1"] == r["true_region"]
        n10 += int(sysok and not retok); n01 += int(retok and not sysok)
    chi2 = ((abs(n10 - n01) - 1) ** 2) / (n10 + n01) if (n10 + n01) else 0.0

    # no-differential ablation: false dispatch from confounders acted as leaks
    def confounder_false_dispatch(rows):
        conf_cases = [r for r in rows if r["true_class"] != "leak"]
        bad = sum(1 for r in conf_cases if acted(r))   # acted on a non-leak => false dispatch
        return bad, len(conf_cases)
    fd_full = confounder_false_dispatch(full_rows)
    fd_nodiff = confounder_false_dispatch(nodiff_rows)

    # loosened contract (analytical on full rows): drop margin + alt predicates
    def loosened(r):
        return r["top1_kind"] == "leak" and r["existence"] >= 0.5 and r["top1_mahal"] <= 3.0
    loose_acted = [r for r in full_rows if loosened(r)]
    loose_fd = sum(1 for r in loose_acted if not (r["true_class"] == "leak"
                   and r["pred_partition"] == r["true_region"]))

    out = {"provenance": {"network": "exa7", "noise_m": NOISE, "seeds": SEEDS,
                          "n_full": len(full_rows)},
           "per_severity_leak_acc": persev,
           "differential_confusion_pooled": {"classes": CLASSES, "matrix": conf},
           "mcnemar_vs_retrieval_pooled": {"n10_system_only": n10, "n01_retrieval_only": n01,
                                           "chi2_cc": chi2, "significant_p<0.05": chi2 > 3.841},
           "ablation_no_differential": {
               "full_confounder_false_dispatch": fd_full,
               "no_diff_confounder_false_dispatch": fd_nodiff},
           "ablation_loosened_contract": {
               "acted": len(loose_acted),
               "false_dispatch": loose_fd,
               "precision": (1 - loose_fd / len(loose_acted)) if loose_acted else 1.0}}
    json.dump(out, open(os.path.join(ROOT, "artifacts", "results_stats.json"), "w",
                        encoding="utf-8"), indent=2)

    print("\n--- per-severity leak-partition accuracy (pooled) ---")
    for k, v in persev.items():
        print(f"   {k:8s}: {v['acc']*100:5.1f}%  (n={v['n']})")
    print(f"\nMcNemar vs retrieval (pooled leaks): n10={n10} n01={n01} chi2={chi2:.1f} "
          f"sig={chi2>3.841}")
    print(f"no-differential ablation: confounder false-dispatch  full {fd_full[0]}/{fd_full[1]}  "
          f"-> no-diff {fd_nodiff[0]}/{fd_nodiff[1]}")
    print(f"loosened-contract: precision {out['ablation_loosened_contract']['precision']*100:.1f}% "
          f"on {len(loose_acted)} acted (vs full contract ~98%)")
    print("\n--- pooled differential confusion (rows=true, cols=pred) ---")
    print("           " + "  ".join(f"{c[:6]:>6}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"  {c[:9]:9s} " + "  ".join(f"{conf[i][j]:6d}" for j in range(len(CLASSES))))
    print("\nsaved -> artifacts/results_stats.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
