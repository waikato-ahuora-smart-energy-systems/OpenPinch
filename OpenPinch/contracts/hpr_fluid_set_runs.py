"""Public values for vapour compression targets run on screened fluid sets."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .hpr import HPRTopologyIdentifier
from .hpr_refrigerant_screen import RefrigerantScreenLayout


class HprFluidSetRun(BaseModel):
    """One vapour compression target run on one screened fluid set.

    ``fluids`` holds one fluid per vapour compression stage, hottest first;
    ``layouts`` are the screened layouts that proposed this run. ``target``
    is the solved HPR target, or ``None`` when the run raised ``error``.
    ``total_annualized_cost`` is in $/y.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    method: HPRTopologyIdentifier
    condensers: int = Field(ge=1)
    evaporators: int = Field(ge=1)
    mvr_stages: int | None = None
    fluids: tuple[str, ...] = Field(min_length=1)
    layouts: tuple[RefrigerantScreenLayout, ...] = Field(min_length=1)
    success: bool
    total_annualized_cost: float | None = None
    error: str | None = None
    target: Any = Field(default=None, exclude=True, repr=False)


class HprFluidSetRunsResult(BaseModel):
    """Every fluid set run, in screening order, and the cheapest success."""

    model_config = ConfigDict(frozen=True)

    runs: tuple[HprFluidSetRun, ...]

    @property
    def best(self) -> HprFluidSetRun | None:
        """The successful run with the lowest total annualized cost."""
        solved = [
            run
            for run in self.runs
            if run.success and run.total_annualized_cost is not None
        ]
        return min(solved, key=lambda run: run.total_annualized_cost, default=None)

    @property
    def failures(self) -> tuple[HprFluidSetRun, ...]:
        """The runs that raised or did not succeed."""
        return tuple(run for run in self.runs if not run.success)


__all__ = ["HprFluidSetRun", "HprFluidSetRunsResult"]
