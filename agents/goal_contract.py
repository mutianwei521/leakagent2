"""
Goal-contract evaluation: the Supervisor's deterministic acceptance predicates.

ACCEPT iff every hard predicate passes. The checks are pure arithmetic on the
evidence package — an LLM auditor (added later) may only ADD rejections, never
flip a failed hard check to ACCEPT. This structurally prevents the kind of
metric-overwriting documented in the prior work.
"""
from __future__ import annotations

from schemas.evidence import GoalVerdict

HARD = ["G1_existence", "G2_region", "G3_margin", "G4_alt", "G6_residual", "G7_safety"]


def evaluate_contract(ev, contract) -> GoalVerdict:
    checks, values, req = {}, {}, []
    top1 = ev.top1

    # G1 — leak existence
    values["existence"] = round(ev.existence_mass, 4)
    checks["G1_existence"] = bool(
        top1 is not None and top1.kind == "leak"
        and ev.existence_mass >= contract.min_leak_probability)

    # G3 — margin over the best alternative of ANY kind
    values["margin"] = round(ev.margin, 4)
    checks["G3_margin"] = bool(ev.margin >= contract.min_top1_margin)

    # G2 — candidate repair region small enough to act on
    values["region_nodes"] = len(ev.candidate_region)
    checks["G2_region"] = bool(len(ev.candidate_region) <= contract.max_candidate_region_nodes)

    # G6 — top-1 must physically reproduce the observation (residual gate)
    rr = (top1.params.get("residual") if top1 else {}) or {}
    values["top1_mahalanobis"] = round(rr.get("mahalanobis_per_dof", 999.0), 3)
    checks["G6_residual"] = bool(
        (not contract.require_residual_consistency)
        or rr.get("mahalanobis_per_dof", 999.0) <= contract.max_residual_mahalanobis)

    # G4 — alternative-hypothesis elimination
    others = [h for h in ev.hypotheses if h is not top1]
    best_alt = max((h.posterior for h in others), default=0.0)
    values["best_alt_posterior"] = round(best_alt, 4)
    checks["G4_alt"] = bool(
        (not contract.require_alt_falsification) or best_alt <= contract.alt_falsify_prob)

    # G7 — safety boundary
    unsafe = ev.safety.get("unsafe_flags", [])
    iso = ev.safety.get("max_isolation_customers", 0)
    checks["G7_safety"] = bool(len(unsafe) <= contract.max_unsafe_actions
                               and iso <= contract.max_isolation_customers)

    failed = [k for k in HARD if not checks.get(k, False)]
    decision = "ACCEPT" if not failed else "REJECT"

    # structured补证 requests for the failed predicates (drive active sensing in P4)
    if "G1_existence" in failed:
        req.append({"type": "raise_existence",
                    "suggested_tool": "active_sensing|digital_twin"})
    if "G3_margin" in failed and top1 is not None and ev.top2 is not None:
        req.append({"type": "separate",
                    "targets": [getattr(top1, "partition", None), getattr(ev.top2, "partition", None)],
                    "suggested_tool": "active_sensing"})
    if "G4_alt" in failed:
        worst = max(others, key=lambda h: h.posterior, default=None)
        if worst is not None:
            req.append({"type": "falsify", "target": worst.key(),
                        "suggested_tool": "digital_twin|acoustic"})

    rationale = ("ACCEPT: all contract predicates satisfied." if decision == "ACCEPT"
                 else "REJECT: " + ", ".join(failed))
    return GoalVerdict(decision=decision, checks=checks, values=values,
                       failed_checks=failed, required_evidence=req,
                       rationale=rationale, contract_version=contract.version)
