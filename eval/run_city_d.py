"""
City D FIELD-REGISTER evaluation (City D district).

Field ingredients that are REAL: the utility's own EPANET model of the district
(vendored unmodified, SHA-256 recorded), 194 audited 2025 leak work orders with
real loss volumes, and an audited real-location node mapping (geocoded address ->
nearest numeric junction, three-CRS cross-check, per-record confidence). What is
NOT real: no SCADA pressure/flow exists for these events, so sensor signatures are
twin-simulated + field noise (sigma=0.15 m); the model vintage is 2016 vs
2025 leaks and no recalibration is claimed. These limits are stated in Methods.

Run:  python eval/run_city_d.py     (after setup + library + anchor corpus)
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
# Leak-rate hypothesis grid: the standard-protocol grid (2-50 L/s) extended down
# to cover the register's severity range (quantiles 0.028/0.106/0.382/1.15 L/s at
# 25/50/75/95%). Chosen from the register RATE distribution alone, before any
# evaluation; a hypothesis space that excludes the data's severities would
# systematically disfavor the leak family.
RATE_GRID = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 35.0, 50.0)
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
        forced = sum(1 for r in bucket if r["retrieval_top1"] in r["true_zones"])
        out.append({"rate_band_Ls": f"[{lo:g},{hi:g})" if hi < 1e9 else f">={lo:g}",
                    "n": k, "median_clean_max_dp_m": round(float(statistics.median(
                        [r["max_abs_dp_clean_m"] for r in bucket])), 4),
                    "forced_top1_zone_acc": forced / k, "n_acted": len(acted),
                    "coverage": len(acted) / k,
                    "decision_precision_on_acted": (sum(1 for r in acted
                        if r["accepted_zone"] in r["true_zones"]) / len(acted)) if acted else 1.0})
    return out


def diagnose_one(e, ctx, ex, sv, gr):
    dp = {str(k): float(v) for k, v in e["obs_dp"].items()}
    # FAIL LOUD on sensor mismatch: every tool defaults missing channels to 0.0,
    # so a corpus/fingerprints sensor-set drift would silently evaluate an
    # all-zero observation (this happened; see commit history).
    missing = [s for s in ctx.sensors if s not in dp]
    if missing:
        raise RuntimeError(f"event {e['id']}: {len(missing)}/{len(ctx.sensors)} sensors "
                           f"absent from corpus obs_dp (e.g. {missing[:3]}); corpus and "
                           "sensor_fingerprints are out of sync - regenerate the corpus.")
    true_zones = set(e["true_zones"])
    res = diagnose(dp, ctx, ex, sv, "day_normal", case_id=str(e["id"]))
    ev = res.evidence
    t1 = ev.top1
    top1_zone = int(t1.partition) if (t1 and t1.kind == "leak") else None
    # Genuinely FORCED zone choices (always defined), matching the manuscript's
    # "forced to answer" semantics: retrieval ranking and best leak hypothesis.
    retrieval_top1 = int(gr.rank_partitions(dp, top_k=1)[0][0])
    leak_hyps = [h for h in ev.hypotheses if h.kind == "leak"]
    twinfit_top1 = int(max(leak_hyps, key=lambda h: h.posterior).partition) if leak_hyps else None
    rr = (t1.params.get("residual") if t1 else {}) or {}
    best_alt = max((h.posterior for h in ev.hypotheses if h is not t1), default=0.0)
    pred = res.accepted_partition
    correct = res.acted and (pred in true_zones)
    return {"id": e["id"], "type": e["type"], "dma": e["dma"], "diameter_mm": e["diameter_mm"],
            "rate_Ls": e["rate_Ls"], "severity_tpd": e["severity_tpd"],
            "true_zones": sorted(true_zones), "max_abs_dp_clean_m": e["max_abs_dp_clean_m"],
            "max_abs_dp_m": round(max((abs(v) for v in dp.values()), default=0.0), 3),
            "outcome": res.outcome, "accepted_zone": pred, "pred_zone": pred,
            "top1_zone": top1_zone, "retrieval_top1": retrieval_top1,
            "twinfit_top1": twinfit_top1, "correct": bool(correct),
            "existence": round(ev.existence_mass, 3), "margin": round(ev.margin, 3),
            "top1_mahal": round(float(rr.get("mahalanobis_per_dof", 999.0)), 3),
            "best_alt": round(float(best_alt), 3)}


def main():
    print("=" * 72)
    print(f"CITY_D FIELD-REGISTER EVALUATION (audited real-location mapping) (sensor_std={SENSOR_STD} m)")
    print("=" * 72)
    ctx = build_context(os.path.join(ROOT, "configs", "city_d.yaml"))
    gr = GraphRAGTool(ctx.cfg["sensor_fingerprints_file"], ctx.sensors)
    ex = ExecutorAgent(ctx, gr, DigitalTwinTool(),
                       ResidualTool(sensor_std=SENSOR_STD, pass_threshold=3.0), BayesianEngine(),
                       top_k_leak=len(ctx.partitions), leak_cands=LEAK_CANDS,
                       rate_grid=RATE_GRID, noise_std=SENSOR_STD)
    sv = SupervisorAgent(GoalContract(min_leak_probability=0.5, min_top1_margin=0.10,
                                      max_candidate_region_nodes=80, alt_falsify_prob=0.30,
                                      max_residual_mahalanobis=3.0))
    nload = ctx.twin.load_cache(os.path.join(ROOT, "data", "city_d", "leak_cache.pkl"))
    print(f"loaded twin leak library: {nload} cached sims.", flush=True)

    cfile = json.load(open(os.path.join(ROOT, "data", "city_d", "scenario_corpus.json"),
                           encoding="utf-8"))
    corpus = cfile["leaks"]
    corpus_sensors = set(map(str, cfile["provenance"].get("sensor_nodes", [])))
    if corpus_sensors and corpus_sensors != set(ctx.sensors):
        raise RuntimeError("scenario_corpus sensor_nodes differ from the live sensor set "
                           f"({len(corpus_sensors & set(ctx.sensors))}/{len(ctx.sensors)} shared); "
                           "regenerate the corpus before evaluating.")
    print(f"field scenario corpus: {len(corpus)} synthetic leaks.", flush=True)

    rows = []
    for i, e in enumerate(corpus):
        rows.append(diagnose_one(e, ctx, ex, sv, gr))
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
        # forced = genuinely forced choices (always defined); the manuscript's
        # "retrieval localizer" number is forced_top1_zone_accuracy.
        "forced_top1_zone_accuracy": sum(1 for r in rows if r["retrieval_top1"] in r["true_zones"]) / n if n else 0.0,
        "forced_twinfit_zone_accuracy": sum(1 for r in rows if r["twinfit_top1"] in r["true_zones"]) / n if n else 0.0,
        "top1_hypothesis_leak_in_true_zone": sum(1 for r in rows if r["top1_zone"] in r["true_zones"]) / n if n else 0.0,
        "false_alarm_dispatch_rate": fa_dispatched / N_FALSE_ALARM,
        "n_false_alarm_controls": N_FALSE_ALARM,
        "sensor_std_m": SENSOR_STD,
    }
    rc = risk_coverage(rows)
    cvs = contract_vs_scalar(rows)
    det = detectability(rows)
    sev = by_severity(rows)

    out = {"provenance": {
                "network": "City D district WDN", "model": "utility data/city_d/city_d.inp (title date withheld, unmodified)",
                "data": "FIELD register, audited mapping: 2025 work orders supply rate/diameter/"
                        "type/DMA AND the audited real-location node; pressures are twin-simulated + "
                        "Gaussian field noise (sigma=0.15 m) because no SCADA exists for these events",
                "mapping": "audited real-location (NOT distribution-placed); per-record anchor "
                           "confidence and merge distance preserved in the corpus",
                "limits": ["model topology/equipment vintage 2016 (demands updated to 2025); no pressure/flow calibration claimed",
                           "coordinate datum unknown (provisional projected CRS), mapping cross-checked",
                           "work-order window is a proxy, not a measured leak duration"],
                "sensors": list(ctx.sensors),
                "sensor_fingerprints_sha256_16": __import__("hashlib").sha256(
                    open(ctx.cfg["sensor_fingerprints_file"], "rb").read()).hexdigest()[:16],
                "rate_grid_Ls": list(RATE_GRID)},
           "summary": summary, "risk_coverage": rc, "contract_vs_scalar_threshold": cvs,
           "detectability": det, "by_severity_band": sev, "rows": rows}
    os.makedirs(os.path.join(ROOT, "artifacts"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "artifacts", "results_city_d.json"), "w",
                        encoding="utf-8"), indent=2, ensure_ascii=False)

    print("\n--- CITY_D FIELD-REGISTER RESULT ---")
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
    print("\nsaved -> artifacts/results_city_d.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
