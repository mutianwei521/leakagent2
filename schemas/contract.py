"""
The Goal Contract: the Supervisor's formal, verifiable acceptance predicates.

The contract thresholds are the paper's tunable knobs that generate the
risk-coverage curve (sweep `min_leak_probability` / `min_top1_margin` / etc.).
Every predicate the supervisor evaluates is deterministic arithmetic over the
evidence package that an LLM cannot override; the language-model auditor may
only add a rejection on top of them.

Predicate numbering below is the one `agents/goal_contract.py` evaluates and the
paper reports: G1 existence, G2 region, G3 margin, G4 alternatives, G6 residual,
G7 safety. G5 and the human-review flag are declared but NOT evaluated (see the
RESERVED block); nothing in the decision path reads them.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field


@dataclass
class GoalContract:
    version: str = "G.v1"
    # G1 existence: P(leak exists) must clear this
    min_leak_probability: float = 0.70
    # G2 region: candidate repair region must be small enough to act on
    max_candidate_region_nodes: int = 25
    # G3 margin: top hypothesis must beat the best alternative (any kind) by this
    min_top1_margin: float = 0.15
    # G4 alternative elimination: every plausible alternative must be 'falsified'
    alt_falsify_prob: float = 0.10            # alternatives below this prob count as excluded
    require_alt_falsification: bool = True
    # G6 residual: the top hypothesis must physically reproduce the observation
    require_residual_consistency: bool = True
    max_residual_mahalanobis: float = 3.0
    # G7 safety
    max_unsafe_actions: int = 0
    max_isolation_customers: int = 0

    # ---- RESERVED: declared for deployment, NOT evaluated by ------------
    # ---- agents/goal_contract.py and never read anywhere in this study --
    # G5 calibration validity (deployment would require the domain-matched
    # calibrator; every leg here is configured with its own network's
    # calibration, but the contract does not check it)
    require_calibrated: bool = True
    # human-review gate (deployment hand-off; no predicate reads this)
    require_human_review_flag: bool = True
    # region extent in metres; G2 evaluates the node count only
    max_candidate_region_extent_m: float = 600.0

    @classmethod
    def from_dict(cls, d: dict) -> "GoalContract":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (d or {}).items() if k in fields})

    def to_dict(self) -> dict:
        return asdict(self)
