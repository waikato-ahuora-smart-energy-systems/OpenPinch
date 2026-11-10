"""Frozen public values for single-stage refrigerant screening."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from ..domain._validation import non_empty_str
from ..domain.analysis import AnalysisProvenance

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


_UNSATURATED = frozenset(
    {
        RefrigerantScreenVerdict.ABOVE_CRITICAL,
        RefrigerantScreenVerdict.BELOW_TRIPLE_POINT,
    }
)


class RefrigerantScreenQuantity(_FrozenContract):
    """One finite magnitude with explicit unit metadata."""

    value: float
    unit: _Text


class RefrigerantScreenRow(_FrozenContract):
    """One fluid screened between one evaporating and one condensing level.

    Saturation pressures, the pressure ratio and the suction volumetric
    heating capacity are ``None`` for a fluid above critical or below its
    triple point. ``rank`` orders the practical fluids of a level pair.
    """

    fluid: _Text
    T_evap: RefrigerantScreenQuantity
    T_cond: RefrigerantScreenQuantity
    T_crit: RefrigerantScreenQuantity
    p_evap: RefrigerantScreenQuantity | None = None
    p_cond: RefrigerantScreenQuantity | None = None
    pressure_ratio: float | None = None
    suction_vhc: RefrigerantScreenQuantity | None = None
    verdict: RefrigerantScreenVerdict
    rank: int | None = None

    @model_validator(mode="after")
    def _validate_verdict_consistency(self) -> Self:
        practical = self.verdict is RefrigerantScreenVerdict.PRACTICAL
        if practical and self.rank is None:
            raise ValueError("practical rows need a rank")
        if not practical and self.rank is not None:
            raise ValueError("only practical rows carry a rank")
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


class HprRefrigerantScreenResult(_FrozenContract):
    """Candidate fluids screened between a target's saturation levels."""

    provenance: AnalysisProvenance | None = None
    source: Literal["carnot_target", "explicit_levels"]
    T_evap_levels: tuple[RefrigerantScreenQuantity, ...]
    T_cond_levels: tuple[RefrigerantScreenQuantity, ...]
    dt_cont: RefrigerantScreenQuantity
    max_pressure_ratio: float
    min_suction_pressure: RefrigerantScreenQuantity
    critical_margin: RefrigerantScreenQuantity
    rows: tuple[RefrigerantScreenRow, ...]

    @property
    def practical(self) -> tuple[RefrigerantScreenRow, ...]:
        """The practical rows, in screening order."""
        return tuple(
            row
            for row in self.rows
            if row.verdict is RefrigerantScreenVerdict.PRACTICAL
        )


__all__ = [
    "HprRefrigerantScreenResult",
    "RefrigerantScreenQuantity",
    "RefrigerantScreenRow",
    "RefrigerantScreenVerdict",
]
