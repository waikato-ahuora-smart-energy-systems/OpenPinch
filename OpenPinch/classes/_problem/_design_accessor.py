"""Design-workflow accessor for problem-owned HEN synthesis operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ...lib.schemas.synthesis import HeatExchangerNetworkSynthesisResult
    from ..pinch_problem import PinchProblem


class _DesignAccessor:
    """Callable design helper for advanced workflows rooted in one problem."""

    def __init__(self, problem: "PinchProblem") -> None:
        self._problem = problem

    def heat_exchanger_network_synthesis(
        self,
        *,
        options: Optional[dict[str, Any]] = None,
        state_id: Optional[str] = None,
        workspace_variant: Optional[str] = None,
    ) -> "HeatExchangerNetworkSynthesisResult":
        """Run HEN synthesis and cache the result on ``problem.results.design``."""
        from ...services.heat_exchanger_network_synthesis.service import (
            heat_exchanger_network_synthesis_service,
        )

        runtime_options = dict(options or {})
        if state_id is not None:
            runtime_options["state_id"] = state_id
        return heat_exchanger_network_synthesis_service(
            self._problem,
            options=runtime_options,
            workspace_variant=workspace_variant,
        )


class _DesignAccessorDescriptor:
    """Non-data descriptor exposing design workflows on problem instances."""

    def __get__(self, obj: Optional["PinchProblem"], owner=None):
        if obj is None:
            return self
        return _DesignAccessor(obj)
