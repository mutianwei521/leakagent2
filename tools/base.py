"""
Tool abstraction + budget-tracking registry.

The LLM Executor ORCHESTRATES tools (decides which to call) but does not compute
their numeric outputs — every number entering the evidence package comes from a
deterministic tool, which is the structural guard against fabricated metrics.
"""
from __future__ import annotations
from typing import Any


class Tool:
    """Base class for executor tools. Tools may expose a generic `run` or
    specialized methods (e.g. DigitalTwinTool.predict_leak); the LLM only
    orchestrates which to call, never computes their numeric outputs."""
    name: str = "tool"
    cost: float = 1.0  # nominal budget units (e.g. WNTR solves) per call

    def run(self, ev, ctx, **kwargs):
        raise NotImplementedError(f"{self.name} exposes specialized methods, not run()")


class BudgetError(RuntimeError):
    pass


class ToolRegistry:
    """Holds the tool instances and tracks a shared resource budget."""

    def __init__(self, tools: dict, budgets: dict | None = None):
        self.tools = dict(tools)                       # name -> Tool
        budgets = budgets or {}
        self.max_twin_sims = budgets.get("max_twin_sims", 100)
        self.max_active_sensors = budgets.get("max_active_sensors", 3)
        self.max_rounds = budgets.get("max_rounds", 4)
        self.spent = {"twin_sims": 0.0, "active_sensors": 0, "total_cost": 0.0}

    def get(self, name: str) -> Tool:
        return self.tools[name]

    def charge(self, kind: str, amount: float = 1.0) -> None:
        self.spent.setdefault(kind, 0)
        self.spent[kind] += amount
        self.spent["total_cost"] += amount

    def twin_budget_left(self) -> float:
        return self.max_twin_sims - self.spent.get("twin_sims", 0)

    def budget_exhausted(self) -> bool:
        return self.spent.get("twin_sims", 0) >= self.max_twin_sims
