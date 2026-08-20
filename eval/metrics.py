"""
Selective-prediction metrics. The accept/abstain decision is re-derived
analytically from per-case stats (no re-simulation), so the risk-coverage curve
is produced by sweeping the contract's existence threshold.

A case is an ACT if the gate says so; it is a CORRECT act iff the true class is
'leak' AND the predicted partition equals the true region. Anything else acted
on is a FALSE DISPATCH (wrong partition, or a non-leak confounder mis-acted).
"""
from __future__ import annotations


def gate_act(row, t_exist, min_margin=0.0, max_mahal=1e9, alt_max=1.0,
             max_region=10 ** 9, mode="full"):
    """Return True if the system would ACT (dispatch) on this case."""
    if row["top1_kind"] != "leak":
        return False
    if row["existence"] < t_exist:
        return False
    if mode == "existence_only":
        return True
    # full physics-and-safety contract
    if row["margin"] < min_margin:
        return False
    if row["top1_mahal"] > max_mahal:
        return False
    if row["best_alt"] > alt_max:
        return False
    if row["region"] > max_region:
        return False
    return True


def _is_correct(row):
    return row["true_class"] == "leak" and row["pred_partition"] == row["true_region"]


def risk_coverage(rows, ts, mode="full", min_margin=0.12, max_mahal=3.0,
                  alt_max=0.20, max_region=60):
    out = []
    n = len(rows)
    for t in ts:
        acted = [r for r in rows if gate_act(r, t, min_margin, max_mahal, alt_max,
                                             max_region, mode=mode)]
        k = len(acted)
        correct = sum(1 for r in acted if _is_correct(r))
        false_disp = k - correct
        # false dispatch caused specifically by a non-leak confounder mis-acted
        false_conf = sum(1 for r in acted if r["true_class"] != "leak")
        out.append({
            "t_exist": round(t, 3),
            "coverage": k / n if n else 0.0,
            "n_acted": k,
            "decision_precision": (correct / k) if k else 1.0,
            "false_dispatch_rate": (false_disp / k) if k else 0.0,
            "false_confounder_acts": false_conf,
            "selective_risk": (false_disp / k) if k else 0.0,
        })
    return out


def risk_coverage_mahal(rows, mahal_grid, t_exist=0.5, min_margin=0.12,
                        alt_max=0.20, max_region=60):
    """Risk-coverage by sweeping the residual-fit gate (the continuous selector):
    a tighter Mahalanobis bound acts on fewer, better-fit leaks -> higher precision."""
    out = []
    n = len(rows)
    for mh in mahal_grid:
        acted = [r for r in rows if gate_act(r, t_exist, min_margin, mh, alt_max,
                                             max_region, mode="full")]
        k = len(acted)
        correct = sum(1 for r in acted if _is_correct(r))
        fd = k - correct
        out.append({"max_mahal": round(float(mh), 3),
                    "coverage": k / n if n else 0.0, "n_acted": k,
                    "decision_precision": (correct / k) if k else 1.0,
                    "false_dispatch_rate": (fd / k) if k else 0.0,
                    "selective_risk": (fd / k) if k else 0.0})
    return out


def aurc(curve):
    """Area under the risk-coverage curve (lower is better), trapezoid over coverage."""
    pts = sorted([(c["coverage"], c["selective_risk"]) for c in curve])
    area = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        area += (x1 - x0) * (y0 + y1) / 2.0
    return area


def precision_at_coverage(curve, target_cov):
    """Highest decision-precision achievable at coverage >= target_cov."""
    elig = [c for c in curve if c["coverage"] >= target_cov]
    if not elig:
        return None
    best = max(elig, key=lambda c: c["decision_precision"])
    return {"coverage": best["coverage"], "decision_precision": best["decision_precision"],
            "false_dispatch_rate": best["false_dispatch_rate"],
            "threshold": best.get("t_exist", best.get("max_mahal"))}


def differential_confusion(rows, classes=("leak", "demand_anomaly", "sensor_fault",
                                          "valve_misstate", "none")):
    """Confusion of the Executor's top-1 hypothesis kind vs the true class."""
    idx = {c: i for i, c in enumerate(classes)}
    mat = [[0] * len(classes) for _ in classes]
    for r in rows:
        ti = idx.get(r["true_class"])
        pi = idx.get(r["top1_kind"])
        if ti is not None and pi is not None:
            mat[ti][pi] += 1
    return {"classes": list(classes), "matrix": mat}


def forced_top1_accuracy(rows):
    """Coverage=1.0 baseline: always trust the retrieval top-1 partition on the
    leak cases (the honest 'old localizer forced to answer' reference)."""
    leaks = [r for r in rows if r["true_class"] == "leak"]
    if not leaks:
        return None
    correct = sum(1 for r in leaks if r["retrieval_top1"] == r["true_region"])
    return {"n_leaks": len(leaks), "top1_accuracy": correct / len(leaks)}
