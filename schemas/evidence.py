"""
Structured evidence schema for the Executor-Supervisor diagnosis loop.

Design intent (see roadmap 3.2 / 4.3):
  - Every change to the hypothesis posterior is traceable to a specific tool
    output via `EvidencePackage.evidence_log`. This makes the diagnosis auditable
    and structurally prevents the kind of hard-coded / overwritten numbers found
    in the prior work: a number with no tool reference cannot survive an audit.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

# The four competing anomaly classes + "no actionable anomaly".
HYPOTHESIS_KINDS = ("leak", "demand_anomaly", "sensor_fault", "valve_misstate", "none")


@dataclass
class Hypothesis:
    """A single competing explanation of an observed pressure anomaly."""
    kind: str                                  # one of HYPOTHESIS_KINDS
    partition: Optional[int] = None            # leak partition (kind == 'leak')
    node: Optional[str] = None                 # leak node / faulty sensor / mis-stated asset
    params: dict = field(default_factory=dict) # e.g. {'rate_Ls': 20.0} or {'sensor': '14852','bias': 0.3}
    prior: float = 0.0
    posterior: float = 0.0
    status: str = "open"                       # 'open' | 'falsified' | 'supported'
    falsified_by: list = field(default_factory=list)  # evidence references that falsified it

    def key(self) -> str:
        """Stable, UNIQUE identity used as a dict key in likelihoods / posteriors.
        Must distinguish hypotheses of the same kind (e.g. a demand anomaly in
        each partition) or their likelihoods collide and the posterior inverts."""
        if self.kind == "leak":
            return f"leak:P{self.partition}"
        if self.kind == "demand_anomaly":
            return f"demand:P{self.partition}"
        if self.node is not None:
            return f"{self.kind}:{self.node}"
        return self.kind

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ToolResult:
    """Output of one tool invocation, with the likelihoods it contributes."""
    tool: str
    cost: float = 0.0                           # budget units consumed (e.g. # WNTR solves)
    payload: dict = field(default_factory=dict) # tool-specific structured output
    likelihoods: dict = field(default_factory=dict)  # {hypothesis_key: likelihood}
    loglik: dict = field(default_factory=dict)       # {hypothesis_key: log-likelihood}
    passed: Optional[bool] = None               # pass/fail for check-style tools
    round_idx: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GoalVerdict:
    """The Supervisor's audit result for one round."""
    decision: str                               # 'ACCEPT' | 'REJECT'
    checks: dict = field(default_factory=dict)  # {predicate_name: bool}
    values: dict = field(default_factory=dict)  # {predicate_name: measured value}
    failed_checks: list = field(default_factory=list)
    required_evidence: list = field(default_factory=list)  # structured补证 requests
    rationale: str = ""
    contract_version: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvidencePackage:
    """The structured dossier the Executor builds and the Supervisor audits."""
    case_id: str = ""
    network: str = ""
    observation: dict = field(default_factory=dict)   # {sensor_node: delta_pressure}
    demand_period: Optional[str] = None
    hypotheses: list = field(default_factory=list)     # list[Hypothesis]
    tool_results: list = field(default_factory=list)   # list[ToolResult]
    candidate_region: list = field(default_factory=list)  # node ids forming the repair region
    existence_mass: float = 0.0     # P(leak exists) = sum of leak-hypothesis posteriors
    top1: Optional[Hypothesis] = None
    top2: Optional[Hypothesis] = None
    margin: float = 0.0             # P(top1) - P(top2)
    entropy_bits: float = 0.0
    rounds: int = 0
    safety: dict = field(default_factory=lambda: {"implied_actions": [], "unsafe_flags": [],
                                                  "max_isolation_customers": 0})
    evidence_log: list = field(default_factory=list)   # auditable per-tool record
    recommendation: Optional[dict] = None              # filled only after ACCEPT

    # ---- mutation helpers -------------------------------------------------
    def add_tool_result(self, tr: ToolResult) -> None:
        self.tool_results.append(tr)
        self.evidence_log.append({
            "round": tr.round_idx, "tool": tr.tool, "cost": tr.cost,
            "loglik_contrib": tr.loglik, "passed": tr.passed,
        })

    def get_hypothesis(self, key: str) -> Optional[Hypothesis]:
        for h in self.hypotheses:
            if h.key() == key:
                return h
        return None

    def leak_hypotheses(self) -> list:
        return [h for h in self.hypotheses if h.kind == "leak"]

    def to_dict(self) -> dict:
        d = asdict(self)
        # dataclasses inside lists are already converted by asdict
        return d
