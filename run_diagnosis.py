"""
Orchestration: the Executor <-> Supervisor diagnostic loop over one observation.

ABSTAIN is a first-class, honest outcome. decision-precision is measured ONLY
over ACTED cases; coverage = fraction acted. The risk-coverage curve is produced
by sweeping the goal-contract thresholds (see eval/selective_prediction.py).
"""
from __future__ import annotations
from dataclasses import dataclass, field

from schemas.evidence import EvidencePackage
from schemas.certificate import make_certificate


@dataclass
class DiagnosisResult:
    outcome: str                 # 'ACTED' | 'ABSTAINED'
    accepted_partition: object   # int | None
    evidence: object
    verdict: object
    history: list = field(default_factory=list)
    certificate: object = None   # AcceptanceCertificate on ACCEPT

    @property
    def acted(self):
        return self.outcome == "ACTED"


def diagnose(observation, ctx, executor, supervisor, demand_period="day_normal",
             case_id="", max_rounds=1):
    ev = EvidencePackage(case_id=case_id, network=ctx.name, observation=observation,
                         demand_period=demand_period)
    history = []
    verdict = None
    for rnd in range(1, max_rounds + 1):
        ev = executor.step(ev, rnd)
        verdict = supervisor.review(ev, rnd)
        history.append(verdict)
        if verdict.decision == "ACCEPT":
            sup_id = getattr(getattr(supervisor, "llm_auditor", None), "client", None)
            cert = make_certificate(
                ev, verdict, getattr(supervisor, "contract", None),
                supervisor_model_id=(getattr(sup_id, "model", None) or "deterministic-goal-contract"))
            ev.recommendation = {
                "leak_partition": ev.top1.partition,
                "fit_node": ev.top1.params.get("fit_node"),
                "fit_rate": ev.top1.params.get("fit_rate"),
                "confidence": round(ev.top1.posterior, 4),
                "candidate_region_size": len(ev.candidate_region),
                "certificate": cert.to_dict(),
            }
            result = DiagnosisResult("ACTED", ev.top1.partition, ev, verdict, history)
            result.certificate = cert
            return result
        # REJECT -> in the full system, active sensing supplies new evidence (P4).
        if rnd < max_rounds:
            executor.incorporate_feedback(verdict.required_evidence)
    return DiagnosisResult("ABSTAINED", None, ev, verdict, history)
