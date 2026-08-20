"""
The make-or-break comparison (audit blocker): does the structured goal contract
beat a well-tuned single-score selective baseline?

We build, on the SAME mixed 4-class test events:
  (1) a calibrated-confidence threshold baseline (logistic calibrator on
      [existence, margin, exp(-mahalanobis)], fit on a DEV seed) — and its ECE
      before/after calibration, addressing the ECE=0.41 blocker;
  (2) a split-conformal selective predictor on that confidence (distribution-free
      target-risk control), reported as coverage at target risk;
  (3) the full goal contract (multi-predicate), swept over its residual gate.

The decisive question is whether the contract's differential gate (existence +
alternative-exclusion) lets it dominate a single-scalar threshold on the MIXED
set, where confidently-wrong confounders are the failure mode a scalar misses.

Run:  python eval/conformal.py     (via the wds_rag interpreter)
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
from eval.run_experiments import run_once
import eval.metrics as M

DEV_SEED, TEST_SEEDS, NOISE = 17, [42, 101, 2025, 31337], 0.05


def correct_action(r):
    return int(r["true_class"] == "leak" and r["top1_kind"] == "leak"
              and r["pred_partition"] == r["true_region"])


def feats(rows):
    X = np.array([[r["existence"], r["margin"], np.exp(-r["top1_mahal"])] for r in rows])
    y = np.array([correct_action(r) for r in rows])
    return X, y


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf), np.asarray(correct)
    e, n = 0.0, len(conf)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (conf > lo) & (conf <= hi) if b else (conf >= lo) & (conf <= hi)
        if m.sum():
            e += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def selective_curve(conf, correct, grid):
    """act if conf>=t : (coverage, precision) over acted."""
    out = []
    for t in grid:
        acted = conf >= t
        k = int(acted.sum())
        out.append({"t": float(t), "coverage": k / len(conf),
                    "precision": float(correct[acted].mean()) if k else 1.0,
                    "risk": float(1 - correct[acted].mean()) if k else 0.0})
    return out


def main():
    print("=" * 72)
    print("CONFORMAL / CALIBRATED-THRESHOLD vs GOAL CONTRACT (mixed test, σ=0.05)")
    print("=" * 72)
    ctx = build_context(os.path.join(ROOT, "configs", "exa7.yaml"))
    gr = GraphRAGTool(ctx.cfg["sensor_fingerprints_file"], ctx.sensors)
    dt, asens = DigitalTwinTool(), ActiveSensingTool()

    # DEV rows (fit calibrator + conformal threshold) and pooled TEST rows
    _, dev_rows = run_once(ctx, gr, dt, asens, NOISE, DEV_SEED, active=True)
    test_rows = []
    for sd in TEST_SEEDS:
        _, rw = run_once(ctx, gr, dt, asens, NOISE, sd, active=True)
        test_rows += rw

    from sklearn.linear_model import LogisticRegression
    Xd, yd = feats(dev_rows)
    cal = LogisticRegression(max_iter=1000).fit(Xd, yd)
    Xt, yt = feats(test_rows)
    conf_cal = cal.predict_proba(Xt)[:, 1]
    conf_raw = np.array([1.0 / (1.0 + r["top1_mahal"]) for r in test_rows])

    ece_raw, ece_cal = ece(conf_raw, yt), ece(conf_cal, yt)
    print(f"\nECE on mixed test:  raw 1/(1+mahal) = {ece_raw:.3f}   ->  calibrated = {ece_cal:.3f}")

    grid = np.linspace(0, 1, 51)
    cal_curve = selective_curve(conf_cal, yt, grid)
    # full goal contract swept over residual gate (existence/margin/alt fixed)
    mahal_grid = [round(x, 3) for x in np.linspace(0.3, 8.0, 40)]
    contract = M.risk_coverage_mahal(test_rows, mahal_grid, min_margin=0.12, alt_max=0.20)
    exist_only = M.risk_coverage(test_rows, [round(x, 3) for x in np.linspace(0, 0.99, 34)],
                                 mode="existence_only")

    def prec_at(curve, cov, pk="precision", ck="coverage"):
        e = [c for c in curve if c[ck] >= cov]
        return max((c[pk] for c in e), default=0.0) if e else 0.0

    print("\nDecision precision at coverage>=0.30 (mixed test, higher=better):")
    print(f"   calibrated-threshold baseline : {prec_at(cal_curve, .30)*100:5.1f}%")
    print(f"   existence-threshold baseline  : {prec_at([{ 'precision':c['decision_precision'],'coverage':c['coverage']} for c in exist_only], .30)*100:5.1f}%")
    print(f"   goal contract (full)          : {prec_at([{ 'precision':c['decision_precision'],'coverage':c['coverage']} for c in contract], .30)*100:5.1f}%")

    # split-conformal selective: DEV threshold for target risk -> TEST coverage/risk
    Xv, yv = feats(dev_rows)
    conf_dev = cal.predict_proba(Xv)[:, 1]
    conformal = {}
    for eps in (0.05, 0.10):
        # smallest threshold whose DEV acted-risk <= eps
        thr = 1.0
        for t in sorted(conf_dev):
            acted = conf_dev >= t
            if acted.sum() and (1 - yv[acted].mean()) <= eps:
                thr = t; break
        acted = conf_cal >= thr
        conformal[f"target_risk_{eps}"] = {
            "threshold": float(thr),
            "test_coverage": float(acted.mean()),
            "test_risk": float(1 - yt[acted].mean()) if acted.sum() else 0.0}
        print(f"   split-conformal @target risk {eps:.2f}: coverage {acted.mean()*100:4.1f}%  "
              f"realized risk {(1-yt[acted].mean())*100 if acted.sum() else 0:4.1f}%")

    out = {"provenance": {"dev_seed": DEV_SEED, "test_seeds": TEST_SEEDS, "noise_m": NOISE,
                          "n_test": len(test_rows)},
           "ece": {"raw": ece_raw, "calibrated": ece_cal},
           "precision_at_cov0.30": {
               "calibrated_threshold": prec_at(cal_curve, .30),
               "existence_only": prec_at([{ 'precision':c['decision_precision'],'coverage':c['coverage']} for c in exist_only], .30),
               "goal_contract": prec_at([{ 'precision':c['decision_precision'],'coverage':c['coverage']} for c in contract], .30)},
           "split_conformal": conformal,
           "calibrated_curve": cal_curve, "contract_curve": contract}
    json.dump(out, open(os.path.join(ROOT, "artifacts", "results_conformal.json"), "w",
                        encoding="utf-8"), indent=2)
    print("\nsaved -> artifacts/results_conformal.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
