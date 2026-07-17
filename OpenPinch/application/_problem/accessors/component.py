from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ....services.components.process_mvr import ProcessMVRComponent
    from ...problem import PinchProblem


class _ComponentAccessor:
    """Helper namespace for mutating prepared process models with components."""

    def __init__(self, problem: "PinchProblem") -> None:
        self._problem = problem

    def process_mvr(
        self,
        source_streams,
        *,
        mvr_id: Optional[str] = None,
        n_stages: int = 1,
        liquid_injection: bool = True,
        mvr_stage_t_lift: Optional[float] = None,
        mvr_stage_pressure_ratio: Optional[float] = None,
        max_stage_t_sat_lift: Optional[float] = None,
        options: Optional[dict[str, Any]] = None,
        period_id: Optional[str] = None,
    ) -> "ProcessMVRComponent":
        """Add a direct process MVR component to selected hot gas stream(s)."""
        from ....services.components.process_mvr import (
            create_process_mvr_component,
        )

        return create_process_mvr_component(
            self._problem,
            source_streams=source_streams,
            mvr_id=mvr_id,
            n_stages=n_stages,
            liquid_injection=liquid_injection,
            mvr_stage_t_lift=mvr_stage_t_lift,
            mvr_stage_pressure_ratio=mvr_stage_pressure_ratio,
            max_stage_t_sat_lift=max_stage_t_sat_lift,
            options=options,
            period_id=period_id,
        )


class _ComponentAccessorDescriptor:
    """Descriptor returning a problem-bound component accessor."""

    def __get__(self, obj: "PinchProblem" | None, owner=None):
        if obj is None:
            return self
        return _ComponentAccessor(obj)
