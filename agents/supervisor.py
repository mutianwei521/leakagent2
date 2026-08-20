"""
SupervisorAgent: the independent goal-review agent.

It never localizes. It evaluates the deterministic goal contract against the
evidence package and returns ACCEPT / REJECT(+required evidence). An optional
LLM auditor (different model family, sees ONLY the structured package, not the
Executor's reasoning) may be attached later; by construction it can only ADD
rejections, never override a failed hard check -> non-collusive by design.
"""
from __future__ import annotations

from .goal_contract import evaluate_contract


class SupervisorAgent:
    def __init__(self, contract, llm_auditor=None):
        self.contract = contract
        self.llm_auditor = llm_auditor   # optional; deferred (P-LLM)

    def review(self, ev, round_idx=1):
        verdict = evaluate_contract(ev, self.contract)
        if verdict.decision == "ACCEPT" and self.llm_auditor is not None:
            # The auditor may only DOWNGRADE accept -> reject (anti-collusion).
            audit = self.llm_auditor.audit(ev, self.contract)
            if audit is not None and audit.get("reject"):
                verdict.decision = "REJECT"
                verdict.failed_checks.append("LLM_audit")
                verdict.required_evidence += audit.get("required_evidence", [])
                verdict.rationale += " | LLM auditor: " + audit.get("reason", "")
        return verdict
