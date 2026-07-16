"""Private stagewise network-evolution records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class _EvolutionCandidateSpec:
    kind: Literal["minus", "plus"]
    unit: int
    branch_index: int
    rank: int
    prev_case: Any
    position: tuple[int, int, int]
    z_allowed: list
    signature: tuple[tuple[int, int, int], ...]


@dataclass
class _EvolutionBranchState:
    model: Any
    best_tac: float
    stale_depths: int = 0
