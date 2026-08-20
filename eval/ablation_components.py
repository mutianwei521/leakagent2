"""
Two component ablations on EXA7, run on IDENTICAL scenarios (5 seeds, sigma=0.05):

A) Occam/BIC penalty OFF (executor occam=False): quantifies the Methods claim
   that without the penalty the over-flexible leak family spuriously out-fits
   genuine zone-wide demand anomalies. Metrics: demand->leak mis-typing count,
   confounder false-dispatches at the nominal gate, decision precision/coverage.

B) Active-sensing RANDOM-REVEAL control (strategy="random"): reveals n_extra
   random hidden nodes instead of the max-discriminability choice, isolating the
   contribution of the discriminability criterion. Metric: leak-partition
   accuracy (none vs random vs discriminative).

Arms share the same generated scenarios per seed, so differences are purely the
ablated component. Output: artifacts/results_ablation_components.json.

Run:  python eval/ablation_components.py   (via the wds_rag interpreter)
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
from eval.run_full_stats import gen_scenarios, row, SEEDS, NOISE, PERIOD

GATE = dict(t_exist=0.5, min_margin=0.12, alt_max=0.20, max_mahal=3.0, max_region=60)


def run_arm(ctx, gr, dt, asens, sc, seed, occam=True, strategy="discriminative"):
    rt = ResidualTool(sensor_std=NOISE, pass_threshold=3.0)
    ex = ExecutorAgent(ctx, gr, dt, rt, BayesianEngine(), top_k_leak=len(ctx.partitions),
                       leak_cands=50, noise_std=NOISE, occam=occam)
    arng = np.random.default_rng(seed + 1)
    pre, post = [], []
    for s in sc:
        ev = EvidencePackage(observation=s["observation"], demand_period=PERIOD, network=ctx.name)
        ex.step(ev, 1)
        pre.append(row(ev, gr, s))
        ex.active_sense_verify(ev, s.get("full_dp", {}), PERIOD, arng, asens,
                               strategy=strategy)
        post.append(row(ev, gr, s))
    return pre, post


def summarize(rows_by_seed):
    """Pooled metrics over seeds for one arm (post-active rows)."""
    allr = [r for rows in rows_by_seed for r in rows]
    n = len(allr)
    dem = [r for r in allr if r["true_class"] == "demand_anomaly"]
    dem2leak = sum(1 for r in dem if r["top1_kind"] == "leak")
    conf = [r for r in allr if r["true_class"] != "leak"]
    acted = [r for r in allr if M.gate_act(r, GATE["t_exist"], GATE["min_margin"],
                                           GATE["max_mahal"], GATE["alt_max"],
                                           GATE["max_region"], mode="full")]
    fd_conf = sum(1 for r in acted if r["true_class"] != "leak")
    correct = sum(1 for r in acted if M._is_correct(r))
    leaks = [r for r in allr if r["true_class"] == "leak"]
    leak_acc = (sum(1 for r in leaks if r["top1_kind"] == "leak"
                    and r["pred_partition"] == r["true_region"]) / len(leaks)) if leaks else 0.0
    return {"n_events": n,
            "demand_to_leak_mistype": [dem2leak, len(dem)],
            "confounder_false_dispatch": [fd_conf, len(conf)],
            "n_acted": len(acted),
            "decision_precision": (correct / len(acted)) if acted else 1.0,
            "coverage": len(acted) / n if n else 0.0,
            "leak_partition_acc": leak_acc}


def main():
    print("=" * 70)
    print("COMPONENT ABLATIONS: Occam penalty OFF + random-reveal active sensing")
    print("=" * 70)
    ctx = build_context(os.path.join(ROOT, "configs", "exa7.yaml"))
    gr = GraphRAGTool(ctx.cfg["sensor_fingerprints_file"], ctx.sensors)
    dt, asens = DigitalTwinTool(), ActiveSensingTool()

    arms = {"full": [], "no_occam": [], "random_reveal": []}
    pre_full = []
    for sd in SEEDS:
        sc = gen_scenarios(ctx, gr, np.random.default_rng(sd))
        p1, r1 = run_arm(ctx, gr, dt, asens, sc, sd, occam=True, strategy="discriminative")
        _, r2 = run_arm(ctx, gr, dt, asens, sc, sd, occam=False, strategy="discriminative")
        _, r3 = run_arm(ctx, gr, dt, asens, sc, sd, occam=True, strategy="random")
        pre_full.append(p1)
        arms["full"].append(r1)
        arms["no_occam"].append(r2)
        arms["random_reveal"].append(r3)
        print(f"  seed {sd} done ({len(sc)} scenarios x 3 arms)", flush=True)

    out = {"provenance": {"network": "exa7", "noise_m": NOISE, "seeds": SEEDS,
                          "gate": GATE,
                          "protocol": "identical scenarios per seed across arms; "
                                      "post-active rows unless noted"},
           "full": summarize(arms["full"]),
           "no_occam": summarize(arms["no_occam"]),
           "random_reveal": summarize(arms["random_reveal"]),
           "no_active_leak_acc": summarize(pre_full)["leak_partition_acc"]}
    json.dump(out, open(os.path.join(ROOT, "artifacts", "results_ablation_components.json"),
                        "w", encoding="utf-8"), indent=2)

    for arm in ("full", "no_occam", "random_reveal"):
        s = out[arm]
        print(f"\n[{arm}] demand->leak {s['demand_to_leak_mistype'][0]}/{s['demand_to_leak_mistype'][1]}"
              f"  conf-FD {s['confounder_false_dispatch'][0]}/{s['confounder_false_dispatch'][1]}"
              f"  precision {s['decision_precision']*100:.1f}%  cov {s['coverage']*100:.1f}%"
              f"  leak-acc {s['leak_partition_acc']*100:.1f}%")
    print(f"\nleak-acc: none {out['no_active_leak_acc']*100:.1f}%  "
          f"random {out['random_reveal']['leak_partition_acc']*100:.1f}%  "
          f"discriminative {out['full']['leak_partition_acc']*100:.1f}%")
    print("\nsaved -> artifacts/results_ablation_components.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
