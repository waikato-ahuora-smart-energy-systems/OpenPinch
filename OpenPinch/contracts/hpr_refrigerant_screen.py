"""Frozen public values for refrigerant screening of HPR layouts."""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..domain._validation import non_empty_str
from ..domain.analysis import AnalysisProvenance
from .hpr import HPRTopologyIdentifier

_Text = Annotated[str, non_empty_str("value must not be empty")]


class _FrozenContract(BaseModel):
    """Strict immutable base for refrigerant screening values."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class RefrigerantScreenVerdict(StrEnum):
    """Screening verdicts, in the order they are judged."""

    ABOVE_CRITICAL = "above critical"
    BELOW_TRIPLE_POINT = "below triple point"
    OVER_RATIO_LIMIT = "over ratio limit"
    DEEP_VACUUM_SUCTION = "deep vacuum suction"
    NEAR_CRITICAL = "near critical"
    PRACTICAL = "practical"


_FEASIBLE = frozenset(
    {RefrigerantScreenVerdict.PRACTICAL, RefrigerantScreenVerdict.NEAR_CRITICAL}
)
_UNSATURATED = frozenset(
    {
        RefrigerantScreenVerdict.ABOVE_CRITICAL,
        RefrigerantScreenVerdict.BELOW_TRIPLE_POINT,
    }
)


# Per stage, hottest first: the feasible (fluid, critical temperature) pairs.
FluidColumns = Sequence[Sequence[tuple[str, float]]]


def count_ordered_fluid_sets(columns: FluidColumns) -> int:
    """Count the sets, one fluid per stage, whose critical temperature never rises.

    Targeting sorts a refrigerant list hottest stage first by critical
    temperature, so a hotter stage never runs a lower-Tcrit fluid; the same
    fluid may repeat. Counted coldest stage first without listing the sets.
    """
    if not columns:
        return 1
    below = [1] * len(columns[-1])
    for upper, lower in zip(columns[-2::-1], columns[:0:-1]):
        below = [
            sum(n for (_, t), n in zip(lower, below) if t <= T_crit)
            for _, T_crit in upper
        ]
    return sum(below)


def ordered_fluid_sets(columns: FluidColumns) -> tuple[tuple[str, ...], ...]:
    """List the sets that :func:`count_ordered_fluid_sets` counts, in stage order."""

    def extend(stage: int, bound: float, prefix: tuple[str, ...]):
        if stage == len(columns):
            yield prefix
            return
        for fluid, T_crit in columns[stage]:
            if T_crit <= bound:
                yield from extend(stage + 1, T_crit, (*prefix, fluid))

    return tuple(extend(0, math.inf, ()))


class RefrigerantScreenQuantity(_FrozenContract):
    """One finite magnitude with explicit unit metadata."""

    value: float
    unit: _Text


class RefrigerantScreenRow(_FrozenContract):
    """One fluid screened on one stage.

    Saturation pressures, the pressure ratio and the suction volumetric
    heating capacity are ``None`` for a fluid above critical or below its
    triple point. ``rank`` orders the feasible (practical or near critical)
    fluids of a stage.
    """

    fluid: _Text
    T_crit: RefrigerantScreenQuantity
    p_evap: RefrigerantScreenQuantity | None = None
    p_cond: RefrigerantScreenQuantity | None = None
    pressure_ratio: float | None = None
    suction_vhc: RefrigerantScreenQuantity | None = None
    verdict: RefrigerantScreenVerdict
    rank: int | None = None

    @model_validator(mode="after")
    def _validate_verdict_consistency(self) -> Self:
        feasible = self.verdict in _FEASIBLE
        if feasible and self.rank is None:
            raise ValueError("feasible rows need a rank")
        if not feasible and self.rank is not None:
            raise ValueError("only feasible rows carry a rank")
        if self.rank is not None and self.rank < 1:
            raise ValueError("rank must be positive")
        properties = (self.p_evap, self.p_cond, self.pressure_ratio, self.suction_vhc)
        if self.verdict in _UNSATURATED:
            if any(value is not None for value in properties):
                raise ValueError(f"a fluid {self.verdict} has no saturation properties")
        elif any(value is None for value in properties):
            raise ValueError(
                f"a fluid judged {self.verdict} needs saturation properties"
            )
        return self


class RefrigerantScreenStage(_FrozenContract):
    """One compression stage of a layout, with every candidate screened.

    The feasible rows come first, ranked from 1; an MVR stage screens water
    only, since an MVR is an open cycle.
    """

    kind: Literal["vapour_compression", "mvr"]
    T_evap: RefrigerantScreenQuantity
    T_cond: RefrigerantScreenQuantity
    rows: tuple[RefrigerantScreenRow, ...] = Field(min_length=1)

    @property
    def feasible(self) -> tuple[str, ...]:
        """The feasible fluids, best ranked first."""
        return tuple(row.fluid for row in self.rows if row.rank is not None)

    @model_validator(mode="after")
    def _validate_rows(self) -> Self:
        ranks = [row.rank for row in self.rows]
        n_feasible = len(self.feasible)
        if ranks[:n_feasible] != list(range(1, n_feasible + 1)):
            raise ValueError("feasible rows come first, ranked 1, 2, ...")
        if self.kind == "mvr" and any(row.fluid != "Water" for row in self.rows):
            raise ValueError("an MVR stage screens water only")
        return self


class RefrigerantScreenLayout(_FrozenContract):
    """One targeting method on one choice of Carnot levels, hottest first.

    ``n_fluid_sets`` counts the fluid sets that can run every stage with
    the critical temperature never rising from the hottest stage down.
    """

    method: HPRTopologyIdentifier
    T_cond_levels: tuple[RefrigerantScreenQuantity, ...] = Field(min_length=1)
    T_evap_levels: tuple[RefrigerantScreenQuantity, ...] = Field(min_length=1)
    stages: tuple[RefrigerantScreenStage, ...] = Field(min_length=1)
    n_fluid_sets: int

    def _columns(self) -> list[list[tuple[str, float]]]:
        return [
            [
                (row.fluid, row.T_crit.value)
                for row in stage.rows
                if row.rank is not None
            ]
            for stage in self.stages
            if stage.kind == "vapour_compression"
        ]

    def _mvr_runs(self) -> bool:
        return all(stage.feasible for stage in self.stages if stage.kind == "mvr")

    @model_validator(mode="after")
    def _validate_count(self) -> Self:
        count = count_ordered_fluid_sets(self._columns()) if self._mvr_runs() else 0
        if self.n_fluid_sets != count:
            raise ValueError(
                f"n_fluid_sets must be {count}, the sets with Tcrit never rising"
            )
        return self

    def fluid_sets(self) -> tuple[tuple[str, ...], ...]:
        """Every vapour compression fluid set, one fluid per stage, hottest first.

        A set's critical temperature never rises down the stages. MVR stages
        always run on water and are left out of the sets.
        """
        return ordered_fluid_sets(self._columns()) if self._mvr_runs() else ()


class HprRefrigerantScreenResult(_FrozenContract):
    """Every layout a parallel or cascade Carnot target's levels allow."""

    provenance: AnalysisProvenance | None = None
    source: Literal["carnot_target", "explicit_levels"]
    topology: Literal["parallel", "cascade"]
    is_heat_pumping: bool
    T_evap_levels: tuple[RefrigerantScreenQuantity, ...]
    T_cond_levels: tuple[RefrigerantScreenQuantity, ...]
    dt_cont: RefrigerantScreenQuantity
    dt_cascade_hx: RefrigerantScreenQuantity
    max_pressure_ratio: float
    min_suction_pressure: RefrigerantScreenQuantity
    critical_margin: RefrigerantScreenQuantity
    layouts: tuple[RefrigerantScreenLayout, ...]

    @property
    def feasible_layouts(self) -> tuple[RefrigerantScreenLayout, ...]:
        """The layouts with at least one fluid set, in screening order."""
        return tuple(layout for layout in self.layouts if layout.n_fluid_sets)


__all__ = [
    "FluidColumns",
    "HprRefrigerantScreenResult",
    "RefrigerantScreenLayout",
    "RefrigerantScreenQuantity",
    "RefrigerantScreenRow",
    "RefrigerantScreenStage",
    "RefrigerantScreenVerdict",
    "count_ordered_fluid_sets",
    "ordered_fluid_sets",
]
