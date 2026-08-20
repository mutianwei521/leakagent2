"""
One-at-a-time sensitivity analysis of the goal-contract thresholds (EXA7).

The supervisor's acceptance gate is pure arithmetic over the per-event evidence
(existence mass, top-1 margin, best-alternative posterior, residual Mahalanobis,
candidate-region size), so sensitivity is evaluated exactly: the per-event
evidence rows are generated ONCE with the standard pipeline (the same
gen_scenarios / run_variant machinery as eval/run_full_stats.py, 5 seeds,
sigma = 0.05 m), and the accept/abstain decision is then re-evaluated
analytically for each perturbed threshold. Nominal values are the EXA7
selective-evaluation operating gate: t_exist = 0.5, min_margin = 0.12,
alt_max = 0.20, max_mahal = 3.0, max_region = 60.

Decision precision counts a dispatch as correct only if the event is a true
leak localized to the true zone (identical to eval/metrics.py); coverage is
the acted fraction of all mixed events (leaks + confounders).

Run:  python eval/contract_sensitivity.py   (via the wds_rag interpreter)
Output: artifacts/results_sensitivity.json  (consumed by gen_si.py, Table S11)
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
from tools.graphrag_tool import GraphRAGTool
from tools.digital_twin import DigitalTwinTool
from tools.active_sensing import ActiveSensingTool
from eval import metrics as M
from eval.run_full_stats import gen_scenarios, run_variant, SEEDS, NOISE

# nominal EXA7 selective-evaluation gate (matches op_point / risk_coverage_mahal)
NOMINAL = {"t_exist": 0.5, "min_margin": 0.12, "alt_max": 0.20,
           "max_mahal": 3.0, "max_region": 60}

GRIDS = {
    "t_exist":    [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "min_margin": [0.04, 0.08, 0.12, 0.16, 0.20, 0.24],
    "alt_max":    [0.10, 0.15, 0.20, 0.25, 0.30],
    "max_mahal":  [2.0, 2.5, 3.0, 3.5, 4.0],
    "max_region": [25, 40, 60, 80, 100],
}


def eval_gate(rows_by_seed, **thr):
    """Pooled precision/coverage plus the per-seed precision spread for one gate."""
    pooled_acted = pooled_correct = pooled_n = 0
    per_seed_prec = []
    for rows in rows_by_seed:
        acted = [r for r in rows if M.gate_act(r, thr["t_exist"], thr["min_margin"],
                                               thr["max_mahal"], thr["alt_max"],
                                               thr["max_region"], mode="full")]
        correct = sum(1 for r in acted if M._is_correct(r))
        pooled_acted += len(acted)
        pooled_correct += correct
        pooled_n += len(rows)
        if acted:
            per_seed_prec.append(correct / len(acted))
    prec = (pooled_correct / pooled_acted) if pooled_acted else 1.0
    sd = float(np.std(per_seed_prec, ddof=1)) if len(per_seed_prec) > 1 else 0.0
    return {"n_acted": pooled_acted, "coverage": pooled_acted / pooled_n,
            "precision": prec, "precision_per_seed_sd": sd,
            "n_seeds_with_actions": len(per_seed_prec)}


def main():
    print("=" * 70)
    print("GOAL-CONTRACT THRESHOLD SENSITIVITY (one-at-a-time, EXA7, sigma=0.05)")
    print("=" * 70)
    ctx = build_context(os.path.join(ROOT, "configs", "exa7.yaml"))
    gr = GraphRAGTool(ctx.cfg["sensor_fingerprints_file"], ctx.sensors)
    dt, asens = DigitalTwinTool(), ActiveSensingTool()

    rows_by_seed = []
    for sd in SEEDS:
        sc = gen_scenarios(ctx, gr, np.random.default_rng(sd))
        rows_by_seed.append(run_variant(ctx, gr, dt, asens, sc, sd, True))
        print(f"  seed {sd} done ({len(sc)} scenarios)")

    nom = eval_gate(rows_by_seed, **NOMINAL)
    print(f"\nnominal gate {NOMINAL}:")
    print(f"  precision {nom['precision']*100:.1f}%  coverage {nom['coverage']*100:.1f}%  "
          f"acted {nom['n_acted']}")

    sweeps = {}
    for param, grid in GRIDS.items():
        pts = []
        for v in grid:
            thr = dict(NOMINAL)
            thr[param] = v
            r = eval_gate(rows_by_seed, **thr)
            r["value"] = v
            pts.append(r)
            print(f"  {param}={v:<5}: precision {r['precision']*100:5.1f}%  "
                  f"coverage {r['coverage']*100:5.1f}%  acted {r['n_acted']}")
        sweeps[param] = pts

    # summary stability numbers for the manuscript sentence
    allpts = [p for pts in sweeps.values() for p in pts]
    pmin = min(p["precision"] for p in allpts)
    pmax = max(p["precision"] for p in allpts)
    print(f"\nprecision across ALL one-at-a-time settings: "
          f"min {pmin*100:.1f}%  max {pmax*100:.1f}%  (nominal {nom['precision']*100:.1f}%)")

    out = {"provenance": {"network": "exa7", "noise_m": NOISE, "seeds": SEEDS,
                          "n_events_pooled": sum(len(r) for r in rows_by_seed),
                          "protocol": "rows generated once (run_full_stats machinery); "
                                      "gate re-evaluated analytically per threshold"},
           "nominal": {**NOMINAL, **nom},
           "precision_min_across_sweeps": pmin,
           "precision_max_across_sweeps": pmax,
           "sweeps": sweeps}
    os.makedirs(os.path.join(ROOT, "artifacts"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "artifacts", "results_sensitivity.json"), "w",
                        encoding="utf-8"), indent=2)
    print("\nsaved -> artifacts/results_sensitivity.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
