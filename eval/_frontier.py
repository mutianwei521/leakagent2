"""Shared selective-prediction frontier helpers, used by BOTH the L-Town
(eval/run_ltown.py) and City H (eval/run_city_h.py) real/real-anchored legs so
the two report identical, drift-free risk-coverage logic.

Each `row` must carry: existence (leak-existence posterior), top1_zone (argmax leak
zone, or None if the top hypothesis is not a leak), true_zones (iterable of correct
zone ids), and outcome ("ACTED"/...). A dispatch is 'correct' when top1_zone is in
true_zones. Both functions only re-gate stored statistics — no hydraulic sims."""
from __future__ import annotations


def risk_coverage(rows):
    """Trace the risk-coverage frontier the standard selective-prediction way: rank
    every event by its leak-EXISTENCE posterior (the confidence score the contract
    gates on) and sweep the acceptance threshold tau. At each tau the system 'acts'
    on events with existence >= tau, predicting the top-1 leak zone; a dispatch is
    'correct' when that zone contains the true location (top1_zone None counts as an
    incorrect dispatch). Coverage = acted / N, decision-precision = correct / acted."""
    n = len(rows)
    taus = sorted({round(r["existence"], 3) for r in rows} | {0.0, 1.01}, reverse=True)
    curve = []
    for tau in taus:
        acted = [r for r in rows if r["existence"] >= tau]
        k = len(acted)
        corr = sum(1 for r in acted if r["top1_zone"] in r["true_zones"])
        curve.append({"existence_threshold": float(tau), "n_acted": k,
                      "coverage": k / n if n else 0.0,
                      "precision": (corr / k) if k else 1.0})
    full = [p for p in curve if p["n_acted"] > 0 and p["precision"] >= 0.999]
    max_cov_full_prec = max(full, key=lambda p: p["coverage"], default=None)
    at90 = max((p for p in curve if p["precision"] >= 0.90 and p["n_acted"] > 0),
               key=lambda p: p["coverage"], default=None)
    return {"frontier": curve, "max_coverage_at_full_precision": max_cov_full_prec,
            "max_coverage_at_90pct_precision": at90}


def contract_vs_scalar(rows):
    """At the contract's own coverage (k acted events), compare its decision-precision
    against a SCALAR existence-threshold rule that accepts the k highest-existence
    events. If the multi-predicate contract scores higher at matched coverage, its
    margin/alternative/residual predicates are doing real work beyond a confidence
    threshold (the 'not a threshold with extra steps' test)."""
    acted = [r for r in rows if r["outcome"] == "ACTED"]
    k = len(acted)
    contract_correct = sum(1 for r in acted if r["top1_zone"] in r["true_zones"])
    topk = sorted(rows, key=lambda r: -r["existence"])[:k]
    scalar_correct = sum(1 for r in topk if r["top1_zone"] in r["true_zones"])
    return {"matched_coverage_acted": k,
            "contract_precision": contract_correct / k if k else 1.0,
            "scalar_existence_precision": scalar_correct / k if k else 1.0,
            "contract_acted": sorted(str(r.get("pipe", r.get("id", "?"))) for r in acted),
            "scalar_top_k": sorted(str(r.get("pipe", r.get("id", "?"))) for r in topk),
            "vetoed_by_contract": sorted(set(str(r.get("pipe", r.get("id", "?"))) for r in topk)
                                         - set(str(r.get("pipe", r.get("id", "?"))) for r in acted))}
