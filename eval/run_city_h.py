"""
City H REGISTER-ANCHORED SEMI-SYNTHETIC evaluation.

The 2018-2020 City H leak register is the only real City H data (diameter/severity/
type/DMA per event). Real SCADA pressure is confidential and unavailable, so each
synthetic leak's sensor signature is twin-simulated on the NOMINAL model + field-grade
noise (built by data/city_h_anchor_corpus.py). External real-MEASUREMENT validity is
carried by the L-Town benchmark, NOT by this leg.

This deliberately confronts the system with the REAL leak-severity distribution: the
median real leak is ~0.094 L/s (54% < 0.1 L/s, 75% < 0.3 L/s), far below the pressure-detectability
floor of a sparse sensor network. The honest, accountable outcome is therefore to ABSTAIN
on the undetectable bulk and act only on the detectable tail -- the exact opposite of the
the predecessor system's unreproduced 67% claim (real run was ~41%, here re-derived from scratch).

Run:  python eval/run_city_h.py     (after setup + library + corpus; via wds_rag python)
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
from eval._frontier import risk_coverage, contract_vs_scalar

SENSOR_STD = 0.15
ACTIONABLE_DP = 0.45       # ~3 sigma; clean max|Δp| above this => individually detectable
LEAK_CANDS = 25
RATE_BANDS = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.30), (0.30, 1.0), (1.0, 1e9)]
N_FALSE_ALARM = 50         # pure-noise (no-leak) controls


def detectability(rows):
    clean = [r["max_abs_dp_clean_m"] for r in rows]
    n = len(rows)
    def frac(th):
        return sum(1 for c in clean if c >= th) / n if n else 0.0
    return {"actionable_dp_threshold_m": ACTIONABLE_DP,
            "median_clean_max_dp_m": float(statistics.median(clean)) if clean else 0.0,
            "p90_clean_max_dp_m": float(np.percentile(clean, 90)) if clean else 0.0,
            "max_clean_max_dp_m": float(max(clean)) if clean else 0.0,
            "frac_clean_dp_ge_0.15m": frac(0.15),
            "frac_clean_dp_ge_0.45m": frac(ACTIONABLE_DP),
            "n_clean_dp_ge_0.45m": sum(1 for c in clean if c >= ACTIONABLE_DP)}


def by_severity(rows):
    out = []
    for lo, hi in RATE_BANDS:
        bucket = [r for r in rows if lo <= r["rate_Ls"] < hi]
        if not bucket:
            continue
        k = len(bucket)
        acted = [r for r in bucket if r["outcome"] == "ACTED"]
        forced = sum(1 for r in bucket if r["top1_zone"] in r["true_zones"])
        out.append({"rate_band_Ls": f"[{lo:g},{hi:g})" if hi < 1e9 else f">={lo:g}",
                    "n": k, "median_clean_max_dp_m": round(float(statistics.median(
                        [r["max_abs_dp_clean_m"] for r in bucket])), 4),
                    "forced_top1_zone_acc": forced / k, "n_acted": len(acted),
                    "coverage": len(acted) / k,
                    "decision_precision_on_acted": (sum(1 for r in acted
                        if r["accepted_zone"] in r["true_zones"]) / len(acted)) if acted else 1.0})
    return out


def diagnose_one(e, ctx, ex, sv):
    dp = {str(k): float(v) for k, v in e["obs_dp"].items()}
    true_zones = set(e["true_zones"])
    res = diagnose(dp, ctx, ex, sv, "day_normal", case_id=str(e["id"]))
    ev = res.evidence
    t1 = ev.top1
    top1_zone = int(t1.partition) if (t1 and t1.kind == "leak") else None
    rr = (t1.params.get("residual") if t1 else {}) or {}
    best_alt = max((h.posterior for h in ev.hypotheses if h is not t1), default=0.0)
    pred = res.accepted_partition
    correct = res.acted and (pred in true_zones)
    return {"id": e["id"], "type": e["type"], "dma": e["dma"], "diameter_mm": e["diameter_mm"],
            "rate_Ls": e["rate_Ls"], "severity_tpd": e["severity_tpd"],
            "true_zones": sorted(true_zones), "max_abs_dp_clean_m": e["max_abs_dp_clean_m"],
            "max_abs_dp_m": round(max((abs(v) for v in dp.values()), default=0.0), 3),
            "outcome": res.outcome, "accepted_zone": pred, "pred_zone": pred,
            "top1_zone": top1_zone, "correct": bool(correct),
            "existence": round(ev.existence_mass, 3), "margin": round(ev.margin, 3),
            "top1_mahal": round(float(rr.get("mahalanobis_per_dof", 999.0)), 3),
            "best_alt": round(float(best_alt), 3)}


def main():
    print("=" * 72)
    print(f"CITY_H REGISTER-ANCHORED SEMI-SYNTHETIC EVALUATION (sensor_std={SENSOR_STD} m)")
    print("=" * 72)
    ctx = build_context(os.path.join(ROOT, "configs", "city_h.yaml"))
    gr = GraphRAGTool(ctx.cfg["sensor_fingerprints_file"], ctx.sensors)
    ex = ExecutorAgent(ctx, gr, DigitalTwinTool(),
                       ResidualTool(sensor_std=SENSOR_STD, pass_threshold=3.0), BayesianEngine(),
                       top_k_leak=len(ctx.partitions), leak_cands=LEAK_CANDS, noise_std=SENSOR_STD)
    sv = SupervisorAgent(GoalContract(min_leak_probability=0.5, min_top1_margin=0.10,
                                      max_candidate_region_nodes=80, alt_falsify_prob=0.30,
                                      max_residual_mahalanobis=3.0))
    nload = ctx.twin.load_cache(os.path.join(ROOT, "data", "city_h", "leak_cache.pkl"))
    print(f"loaded twin leak library: {nload} cached sims.", flush=True)

    corpus = json.load(open(os.path.join(ROOT, "data", "city_h", "leak_corpus.json"),
                            encoding="utf-8"))["leaks"]
    print(f"register-anchored corpus: {len(corpus)} synthetic leaks.", flush=True)

    rows = []
    for i, e in enumerate(corpus):
        rows.append(diagnose_one(e, ctx, ex, sv))
        if (i + 1) % 100 == 0:
            nact = sum(1 for r in rows if r["outcome"] == "ACTED")
            print(f"   diagnosed {i+1}/{len(corpus)} (acted so far: {nact})", flush=True)

    # ---- no-leak false-alarm control (pure field noise, no injected leak) ----
    rng = np.random.RandomState(SEED := 7)
    fa_dispatched = 0
    for j in range(N_FALSE_ALARM):
        obs = {s: float(rng.normal(0.0, SENSOR_STD)) for s in ctx.sensors}
        res = diagnose(obs, ctx, ex, sv, "day_normal", case_id=f"noise{j}")
        if res.acted:
            fa_dispatched += 1

    acted = [r for r in rows if r["outcome"] == "ACTED"]
    correct = [r for r in acted if r["correct"]]
    n = len(rows)
    summary = {
        "n_leaks_evaluated": n, "n_acted": len(acted),
        "coverage": len(acted) / n if n else 0.0,
        "decision_precision_on_acted": len(correct) / len(acted) if acted else 1.0,
        "forced_top1_zone_accuracy": sum(1 for r in rows if r["top1_zone"] in r["true_zones"]) / n if n else 0.0,
        "false_alarm_dispatch_rate": fa_dispatched / N_FALSE_ALARM,
        "n_false_alarm_controls": N_FALSE_ALARM,
        "sensor_std_m": SENSOR_STD,
    }
    rc = risk_coverage(rows)
    cvs = contract_vs_scalar(rows)
    det = detectability(rows)
    sev = by_severity(rows)

    out = {"provenance": {
                "network": "City H municipal WDN", "model": "nominal dataset/city_h.inp",
                "data": "register-anchored SEMI-SYNTHETIC: real 2018-2020 leak register drives "
                        "leak rate/diameter/type/DMA per event; pressures are twin-simulated + "
                        "Gaussian field noise (sigma=0.15 m) because real SCADA is confidential",
                "mapping": "distribution_only (register has no key to the model's node IDs)",
                "dp_method": "twin Δp at register-derived rate + field noise (NOT a real measurement)",
                "honesty_note": "external real-measurement validity is carried by the L-Town leg, "
                                "not by City H; the predecessor system's 67% claim could not be reproduced "
                                "(real ~41%) and is inherited NOWHERE here."},
           "summary": summary, "risk_coverage": rc, "contract_vs_scalar_threshold": cvs,
           "detectability": det, "by_severity_band": sev, "rows": rows}
    os.makedirs(os.path.join(ROOT, "artifacts"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "artifacts", "results_city_h.json"), "w",
                        encoding="utf-8"), indent=2, ensure_ascii=False)

    print("\n--- CITY_H REGISTER-ANCHORED RESULT ---")
    print(f"  synthetic leaks (one per real register event): {n}")
    print(f"  real leak severities: median {statistics.median([r['rate_Ls'] for r in rows]):.3f} L/s, "
          f"max {max(r['rate_Ls'] for r in rows):.3f} L/s")
    print(f"  detectability: median clean max|Δp| {det['median_clean_max_dp_m']:.3f} m; "
          f"{det['frac_clean_dp_ge_0.45m']*100:.1f}% of real leaks exceed the {ACTIONABLE_DP} m actionable floor")
    print(f"  forced top-1 zone accuracy (act on all): {summary['forced_top1_zone_accuracy']*100:.1f}%")
    print(f"  ACCOUNTABLE: coverage {summary['coverage']*100:.1f}%  "
          f"decision-precision on acted {summary['decision_precision_on_acted']*100:.1f}%  "
          f"false-alarm dispatch {summary['false_alarm_dispatch_rate']*100:.1f}%")
    print("\n  by severity band (L/s):")
    print(f"    {'band':>12s} {'n':>4s} {'med|Δp|m':>9s} {'forced':>7s} {'cover':>6s} {'prec':>6s}")
    for b in sev:
        print(f"    {b['rate_band_Ls']:>12s} {b['n']:>4d} {b['median_clean_max_dp_m']:>9.4f} "
              f"{b['forced_top1_zone_acc']*100:>6.0f}% {b['coverage']*100:>5.0f}% "
              f"{b['decision_precision_on_acted']*100:>5.0f}%")
    if cvs["matched_coverage_acted"] > 0:
        print(f"\n  contract vs scalar @ {cvs['matched_coverage_acted']} acted: "
              f"{cvs['contract_precision']*100:.0f}% vs {cvs['scalar_existence_precision']*100:.0f}%")
    print("\nsaved -> artifacts/results_city_h.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
