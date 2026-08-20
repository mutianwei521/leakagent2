"""
L-Town / BattLeDIM REAL-DATA evaluation (the journal-gate leg).

For each labeled leak (a pipe, start/peak time, type), the observed sensor
pressure deviation is a diurnal-matched ONSET STEP — mean pressure over the day
after the leak starts minus the day before, by hour-of-day — which isolates the
new leak's contribution from pre-existing concurrent leaks and diurnal demand.
The executor-supervisor diagnoses it against the NOMINAL L-TOWN.inp model (a real
sim-to-real gap), and we check whether the predicted zone contains the leak pipe.

This is genuine real-measurement data; we expect the system to ABSTAIN on the
weak/ambiguous events and act on the clear ones — the accountable-diagnosis story.

Run:  python eval/run_ltown.py     (via the wds_rag interpreter)
"""
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from data.context import build_context
from schemas.evidence import EvidencePackage
from schemas.contract import GoalContract
from tools.graphrag_tool import GraphRAGTool
from tools.digital_twin import DigitalTwinTool
from tools.residual_analysis import ResidualTool
from tools.bayesian import BayesianEngine
from agents.executor import ExecutorAgent
from agents.supervisor import SupervisorAgent
from run_diagnosis import diagnose
from eval._frontier import risk_coverage, contract_vs_scalar

LT = os.environ.get("LTOWN_SCADA_DIR", r"data/ltown/scada")
SENSOR_STD = 0.15         # onset-step noise floor (m); large values make real leaks look like noise
WIN_H = 24


def load_leaks():
    cfg = yaml.safe_load(open(os.path.join(LT, "dataset_configuration.yaml"), encoding="utf-8"))
    out = []
    for item in cfg.get("leakages", []):
        if not isinstance(item, str) or item.strip().startswith("#"):
            continue
        parts = [p.strip() for p in item.split(",")]
        if len(parts) < 6:
            continue
        out.append({"pipe": parts[0], "start": pd.Timestamp(parts[1]), "end": pd.Timestamp(parts[2]),
                    "diam": float(parts[3]), "type": parts[4], "peak": pd.Timestamp(parts[5])})
    return out


def load_pressures(year):
    df = pd.read_csv(os.path.join(LT, f"{year}_SCADA_Pressures.csv"), sep=";", decimal=",")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=False, errors="coerce")
    return df.set_index("Timestamp").sort_index()


def onset_dp(P, sensors, start, win_h=WIN_H):
    pre = P[(P.index >= start - pd.Timedelta(hours=win_h)) & (P.index < start)]
    post = P[(P.index >= start) & (P.index < start + pd.Timedelta(hours=win_h))]
    if len(pre) < 12 or len(post) < 12:
        return None
    dp = {}
    for s in sensors:
        if s not in P.columns:
            continue
        ph = pre[s].groupby(pre.index.hour).mean()
        qh = post[s].groupby(post.index.hour).mean()
        common = ph.index.intersection(qh.index)
        dp[s] = float((qh[common] - ph[common]).mean()) if len(common) else 0.0
    return dp


# risk_coverage() and contract_vs_scalar() now live in eval/_frontier.py
# (shared verbatim with eval/run_city_h.py so the two legs cannot drift).


def main():
    print("=" * 72)
    print(f"L-TOWN / BattLeDIM REAL-DATA EVALUATION (nominal model, sensor_std={SENSOR_STD} m)")
    print("=" * 72)
    ctx = build_context(os.path.join(ROOT, "configs", "ltown.yaml"))
    gr = GraphRAGTool(ctx.cfg["sensor_fingerprints_file"], ctx.sensors)
    LEAK_CANDS = 25
    ex = ExecutorAgent(ctx, gr, DigitalTwinTool(),
                       ResidualTool(sensor_std=SENSOR_STD, pass_threshold=3.0), BayesianEngine(),
                       top_k_leak=len(ctx.partitions), leak_cands=LEAK_CANDS, noise_std=SENSOR_STD)
    sv = SupervisorAgent(GoalContract(min_leak_probability=0.5, min_top1_margin=0.10,
                                      max_candidate_region_nodes=80, alt_falsify_prob=0.30,
                                      max_residual_mahalanobis=3.0))
    wn = ctx.twin.wn
    n2c = ctx.node_to_partition
    leaks = load_leaks()
    print(f"{len(leaks)} labeled leaks.", flush=True)

    # load the pre-built twin leak-response library (data/ltown/leak_cache.pkl)
    nload = ctx.twin.load_cache(os.path.join(ROOT, "data", "ltown", "leak_cache.pkl"))
    print(f"loaded twin leak library: {nload} cached sims.", flush=True)

    P = {2018: load_pressures(2018), 2019: load_pressures(2019)}
    rows = []
    for i, lk in enumerate(leaks):
        yr = lk["start"].year
        try:
            l = wn.get_link(lk["pipe"])
            true_zones = {n2c.get(str(l.start_node_name)), n2c.get(str(l.end_node_name))}
            true_zones.discard(None)
        except Exception:
            continue
        dp = onset_dp(P[yr], ctx.sensors, lk["start"])
        if dp is None:
            continue
        maxdp = max((abs(v) for v in dp.values()), default=0.0)
        res = diagnose(dp, ctx, ex, sv, "day_normal", case_id=lk["pipe"])
        pred = res.accepted_partition
        ev = res.evidence
        t1 = ev.top1
        top1_zone = int(t1.partition) if (t1 and t1.kind == "leak") else None  # top-1 regardless of accept
        rr = (t1.params.get("residual") if t1 else {}) or {}
        best_alt = max((h.posterior for h in ev.hypotheses if h is not t1), default=0.0)
        correct = res.acted and (pred in true_zones)
        rows.append({"pipe": lk["pipe"], "type": lk["type"], "true_zones": sorted(true_zones),
                     "max_abs_dp_m": round(maxdp, 3), "outcome": res.outcome,
                     "pred_zone": pred, "top1_zone": top1_zone, "correct": bool(correct),
                     "existence": round(ev.existence_mass, 3), "margin": round(ev.margin, 3),
                     "top1_mahal": round(float(rr.get("mahalanobis_per_dof", 999.0)), 3),
                     "best_alt": round(float(best_alt), 3)})
        flag = "OK" if correct else ("top1ok" if top1_zone in true_zones else "")
        print(f"  [{i+1:2d}] {lk['pipe']:5s} {lk['type']:9s} max|dp|={maxdp:5.2f}m true={sorted(true_zones)} "
              f"-> {res.outcome:9s} pred={pred} top1={top1_zone} {flag}")

    acted = [r for r in rows if r["outcome"] == "ACTED"]
    correct = [r for r in acted if r["correct"]]
    n = len(rows)
    summary = {
        "n_leaks_evaluated": n, "n_acted": len(acted),
        "coverage": len(acted) / n if n else 0.0,
        "decision_precision_on_acted": len(correct) / len(acted) if acted else 0.0,
        "forced_top1_zone_accuracy": sum(1 for r in rows if r["top1_zone"] in r["true_zones"]) / n if n else 0.0,
        "sensor_std_m": SENSOR_STD,
    }
    rc = risk_coverage(rows)
    cvs = contract_vs_scalar(rows)
    out = {"provenance": {"benchmark": "BattLeDIM L-Town", "model": "nominal L-TOWN.inp",
                          "data": "competition-provided SCADA pressures (simulated by the organizers) + labeled leaks",
                          "dp_method": "diurnal-matched onset step (+-24h)"},
           "summary": summary, "risk_coverage": rc,
           "contract_vs_scalar_threshold": cvs, "rows": rows}
    json.dump(out, open(os.path.join(ROOT, "artifacts", "results_ltown.json"), "w", encoding="utf-8"),
              indent=2)
    print("\n--- L-TOWN REAL-DATA RESULT ---")
    print(f"  leaks evaluated: {n}")
    print(f"  forced top-1 zone accuracy (act on all): {summary['forced_top1_zone_accuracy']*100:.0f}%")
    print(f"  ACCOUNTABLE: coverage {summary['coverage']*100:.0f}%  "
          f"decision-precision on acted {summary['decision_precision_on_acted']*100:.0f}%")
    print("\n--- REAL-DATA RISK-COVERAGE FRONTIER (rank by leak-existence, sweep accept threshold) ---")
    print(f"  {'exist>=':>8s} {'n_acted':>8s} {'coverage':>9s} {'precision':>10s}")
    seen = None
    for p in rc["frontier"]:
        key = (p["n_acted"], round(p["precision"], 3))
        if key == seen:
            continue          # collapse runs of equal (acted, precision) for readability
        seen = key
        print(f"  {p['existence_threshold']:>8.2f} {p['n_acted']:>8d} {p['coverage']*100:>8.0f}% {p['precision']*100:>9.0f}%")
    mc = rc["max_coverage_at_full_precision"]
    if mc:
        print(f"  >> max coverage at 100% precision: {mc['coverage']*100:.0f}% ({mc['n_acted']} leaks)")
    m9 = rc["max_coverage_at_90pct_precision"]
    if m9:
        print(f"  >> max coverage at >=90% precision: {m9['coverage']*100:.0f}% "
              f"({m9['n_acted']} leaks, precision {m9['precision']*100:.0f}%)")
    print("\n--- CONTRACT vs SCALAR THRESHOLD (matched coverage) ---")
    print(f"  at {cvs['matched_coverage_acted']} acted leaks: "
          f"contract precision {cvs['contract_precision']*100:.0f}%  vs  "
          f"scalar-existence precision {cvs['scalar_existence_precision']*100:.0f}%")
    print(f"  contract vetoes (high existence, fails margin/alt): {cvs['vetoed_by_contract']}")
    print("saved -> artifacts/results_ltown.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
