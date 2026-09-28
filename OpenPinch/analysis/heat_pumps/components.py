"""Lightweight base records for process components."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ...domain.zone import Zone


class ProcessComponentHost(Protocol):
    """The ``PinchProblem`` members process components use.

    Defined here so the analysis layer does not import the application layer.
    """

    _process_components: dict[str, ProcessComponent]

    @property
    def process_components(self) -> Mapping[str, ProcessComponent]: ...

    def _invalidate_analysis(self) -> None: ...

    def _require_prepared_root_zone(self) -> Zone: ...


@dataclass
class ProcessComponent:
    """Base class for memory-only process components."""

    id: str
    problem: ProcessComponentHost
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
        self.problem._invalidate_analysis()


def _clear_zone_targets(zone: "Zone") -> None:
    zone.targets.clear()
    zone.graphs.clear()
    for subzone in zone.subzones.values():
        _clear_zone_targets(subzone)
