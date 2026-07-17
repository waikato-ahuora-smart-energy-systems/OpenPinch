"""Internal heat-exchanger area-slice model and calculations."""

from __future__ import annotations

import math
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, field_validator

_AREA_ABS_TOL = 1e-3
_AREA_REL_TOL = 1e-4


class HeatExchangerAreaSlice(BaseModel):
    """One duty-aligned local area slice within a parent-level exchanger."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    period: str
    hot_segment_identity: str
    cold_segment_identity: str
    duty: float
    hot_inlet_temperature: float
    hot_outlet_temperature: float
    cold_inlet_temperature: float
    cold_outlet_temperature: float
    hot_htc: float
    cold_htc: float
    overall_htc: float
    lmtd: float
    area: float

    @field_validator("period", "hot_segment_identity", "cold_segment_identity")
    @classmethod
    def _validate_identity(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("segment area identities must not be empty")
        return text

    @field_validator("duty", "hot_htc", "cold_htc", "overall_htc", "lmtd", "area")
    @classmethod
    def _validate_positive_finite(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("segment area values must be finite and positive")
        return float(value)

    @field_validator(
        "hot_inlet_temperature",
        "hot_outlet_temperature",
        "cold_inlet_temperature",
        "cold_outlet_temperature",
    )
    @classmethod
    def _validate_slice_temperature(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("segment area temperatures must be finite")
        return float(value)


def segment_duty_by_period(
    contributions: Iterable[HeatExchangerAreaSlice],
) -> dict[str, float]:
    """Return local slice duty totals grouped by operating period."""
    return _sum_by_period(contributions, attribute="duty")


def segment_area_by_period(
    contributions: Iterable[HeatExchangerAreaSlice],
) -> dict[str, float]:
    """Return local slice area totals grouped by operating period."""
    return _sum_by_period(contributions, attribute="area")


def segment_design_area(
    contributions: Iterable[HeatExchangerAreaSlice],
) -> float | None:
    """Return the maximum period-total slice area when slices are available."""
    period_totals = segment_area_by_period(contributions)
    return max(period_totals.values()) if period_totals else None


def validate_segment_design_area(
    area: float | None,
    contributions: Iterable[HeatExchangerAreaSlice],
) -> float | None:
    """Return the authoritative design area or reject an inconsistent value."""
    design_area = segment_design_area(contributions)
    if design_area is None:
        return area
    if area is not None and not math.isclose(
        area,
        design_area,
        rel_tol=_AREA_REL_TOL,
        abs_tol=_AREA_ABS_TOL,
    ):
        raise ValueError(
            f"area must match the maximum period-total segment area {design_area:.12g}"
        )
    return design_area


def _sum_by_period(
    contributions: Iterable[HeatExchangerAreaSlice],
    *,
    attribute: str,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for contribution in contributions:
        totals[contribution.period] = totals.get(contribution.period, 0.0) + float(
            getattr(contribution, attribute)
        )
    return totals


__all__: list[str] = []
