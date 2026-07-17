"""Lightweight base records for process components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...application.problem import PinchProblem
    from ...domain.zone import Zone


@dataclass
class ProcessComponent:
    """Base class for memory-only process components."""

    id: str
    problem: "PinchProblem"
    component_type: str
    active: bool = True

    def activate(self):
        """Activate the component."""
        self.active = True
        self._invalidate_problem_targets()
        return self

    def deactivate(self):
        """Deactivate the component."""
        self.active = False
        self._invalidate_problem_targets()
        return self

    def _invalidate_problem_targets(self) -> None:
        root = self.problem.master_zone
        if root is not None:
            _clear_zone_targets(root)
        self.problem._results = None


def _clear_zone_targets(zone: "Zone") -> None:
    zone.targets.clear()
    zone.graphs.clear()
    for subzone in zone.subzones.values():
        _clear_zone_targets(subzone)
