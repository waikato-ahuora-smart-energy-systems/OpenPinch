"""Parent-owned runtime state for multiperiod HPR preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ....contracts.hpr import HPRPeriodCase
from ....domain.problem_table import ProblemTable


@dataclass(slots=True)
class _PreparedHPRPeriodCase:
    """Runtime data for one period in a shared HPR design solve."""

    period_id: str
    period_idx: int
    weight: float
    solver_case: HPRPeriodCase
    base_target: Any
    optimizer_pt: ProblemTable


__all__ = ["_PreparedHPRPeriodCase"]
