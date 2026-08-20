"""Core data schemas for the Executor-Supervisor diagnosis system."""
from .evidence import Hypothesis, ToolResult, EvidencePackage, GoalVerdict
from .contract import GoalContract

__all__ = [
    "Hypothesis", "ToolResult", "EvidencePackage", "GoalVerdict", "GoalContract",
]
