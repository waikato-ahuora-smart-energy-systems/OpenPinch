"""Application descriptor for optional graph presentation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...problem import PinchProblem


class _PlotAccessorDescriptor:
    """Bind the presentation-owned graph accessor to one problem instance."""

    def __get__(self, obj: "PinchProblem | None", owner=None):
        if obj is None:
            return self
        from ....presentation.graphs.problem import _PlotAccessor
        from ... import hpr_selection

        # Looked up per call so the selector stays patchable in tests.
        return _PlotAccessor(
            obj,
            select_hpr_graphs=lambda *args, **kwargs: hpr_selection.select_hpr_graphs(
                *args, **kwargs
            ),
        )
