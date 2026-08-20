"""
Acceptance certificate: the tamper-evident audit artefact emitted on ACCEPT.

It binds the supervisor's decision (predicate values, thresholds, accepted
hypothesis) to a SHA-256 digest of a canonical serialisation of the evidence
package. Re-hashing the (possibly tampered) evidence and comparing to the
certificate's digest detects any subsequent modification — the artefact a
utility or regulator needs to trust an automated action. This makes the
Methods' "cryptographic digest renders tampering detectable" claim concrete.
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, field, asdict


def canonical_bytes(ev) -> bytes:
    """Deterministic, sorted-key JSON of the evidence package. Excludes the
    post-decision `recommendation` field (which carries the certificate itself),
    so the digest attests to the diagnostic evidence, not to itself."""
    d = ev.to_dict()
    d.pop("recommendation", None)
    return json.dumps(d, sort_keys=True, ensure_ascii=False,
                      default=str, separators=(",", ":")).encode("utf-8")


def evidence_digest(ev) -> str:
    return hashlib.sha256(canonical_bytes(ev)).hexdigest()


@dataclass
class AcceptanceCertificate:
    decision: str
    contract_version: str
    thresholds: dict = field(default_factory=dict)
    predicate_checks: dict = field(default_factory=dict)   # {Gn: bool}
    predicate_values: dict = field(default_factory=dict)   # {name: measured value}
    accepted_hypothesis: dict = field(default_factory=dict)
    candidate_region_size: int = 0
    rounds: int = 0
    supervisor_model_id: str = "deterministic-goal-contract"
    evidence_sha256: str = ""
    timestamp: str = ""        # caller may stamp; left blank for reproducibility

    def to_dict(self) -> dict:
        return asdict(self)

    def verify(self, ev) -> bool:
        """True iff the certificate's digest still matches the evidence package."""
        return self.evidence_sha256 == evidence_digest(ev)


def make_certificate(ev, verdict, contract, supervisor_model_id=None,
                     timestamp="") -> AcceptanceCertificate:
    top = ev.top1
    return AcceptanceCertificate(
        decision=verdict.decision,
        contract_version=getattr(contract, "version", "G.v1"),
        thresholds=contract.to_dict() if hasattr(contract, "to_dict") else {},
        predicate_checks=dict(verdict.checks),
        predicate_values=dict(verdict.values),
        accepted_hypothesis={
            "type": getattr(top, "kind", None),
            "partition": getattr(top, "partition", None),
            "node": (top.params.get("fit_node") if top else None),
            "rate_Ls": (top.params.get("fit_rate") if top else None),
            "posterior": round(getattr(top, "posterior", 0.0), 4) if top else 0.0,
        },
        candidate_region_size=len(ev.candidate_region),
        rounds=ev.rounds,
        supervisor_model_id=supervisor_model_id or "deterministic-goal-contract",
        evidence_sha256=evidence_digest(ev),
        timestamp=timestamp,
    )
