"""Frozen public values for heat pump and refrigeration load kinks."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ..domain._validation import non_empty_str
from ..domain.analysis import AnalysisProvenance
from ..domain.value import Value
from .heat_recovery_dt_min import HeatRecoveryQuantity

_NonEmptyText = Annotated[str, non_empty_str("value must not be empty")]


class _FrozenContract(BaseModel):
    """Strict immutable base for load-kink values."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def _canonical(quantity: HeatRecoveryQuantity, unit: str, field: str) -> float:
    try:
        return float(Value(quantity.value, unit=quantity.unit).to(unit))
    except Exception as exc:
        raise ValueError(f"{field} must use units compatible with {unit}") from exc


class HprLoadKinkKind(StrEnum):
    """How the net load profile bends at one vertex.

    ``STEP`` is a temperature change with no duty after the vertex.
    ``STEEPENS`` and ``FLATTENS`` mark a slope that changes by at least the
    steepening factor; ``MINOR`` is any smaller change.
    """

    STEP = "step"
    STEEPENS = "steepens"
    FLATTENS = "flattens"
    MINOR = "minor"


LOGICAL_LOAD_KINDS = frozenset({HprLoadKinkKind.STEP, HprLoadKinkKind.STEEPENS})


class HprLoadKink(_FrozenContract):
    """One interior vertex of the sink or source net load profile.

    Slopes are shifted-temperature changes per unit load fraction, so a slope
    of 20 K means the profile spans 20 K over its whole duty at that rate.
    """

    load_fraction: float
    duty: HeatRecoveryQuantity
    temperature: HeatRecoveryQuantity
    slope_before: HeatRecoveryQuantity
    slope_after: HeatRecoveryQuantity | None
    kind: HprLoadKinkKind

    @field_validator("load_fraction")
    @classmethod
    def _validate_load_fraction(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError("load_fraction must be between 0 and 1")
        return value

    @model_validator(mode="after")
    def _validate_vertex(self) -> "HprLoadKink":
        _canonical(self.duty, "kW", "duty")
        _canonical(self.temperature, "degC", "temperature")
        _canonical(self.slope_before, "delta_degC", "slope_before")
        if self.kind is HprLoadKinkKind.STEP:
            if self.slope_after is not None:
                raise ValueError("a step has no slope after the vertex")
        elif self.slope_after is None:
            raise ValueError("only a step has no slope after the vertex")
        else:
            _canonical(self.slope_after, "delta_degC", "slope_after")
        return self


class HprLoadKinksResult(_FrozenContract):
    """Vertices of a net load profile and the logical HPR load fractions.

    ``logical_load_fractions`` are the step and steepens vertices in profile
    order followed by 1.0. Each can be passed as ``load_fraction`` to any heat
    pump (sink profile) or refrigeration (source profile) target.
    """

    provenance: AnalysisProvenance | None = None
    scope: _NonEmptyText
    period_id: _NonEmptyText
    is_refrigeration: bool
    steepening: float
    maximum_duty: HeatRecoveryQuantity
    vertices: tuple[HprLoadKink, ...]
    logical_load_fractions: tuple[float, ...]

    @field_validator("steepening")
    @classmethod
    def _validate_steepening(cls, value: float) -> float:
        if value <= 1.0:
            raise ValueError("steepening must be greater than 1")
        return value

    @model_validator(mode="after")
    def _validate_profile(self) -> "HprLoadKinksResult":
        if _canonical(self.maximum_duty, "kW", "maximum_duty") <= 0.0:
            raise ValueError("maximum_duty must be positive")
        fractions = [vertex.load_fraction for vertex in self.vertices]
        if any(later <= earlier for earlier, later in zip(fractions, fractions[1:])):
            raise ValueError("vertex load fractions must be strictly increasing")
        expected = tuple(
            vertex.load_fraction
            for vertex in self.vertices
            if vertex.kind in LOGICAL_LOAD_KINDS
        ) + (1.0,)
        if self.logical_load_fractions != expected:
            raise ValueError(
                "logical_load_fractions must list the step and steepens "
                "vertices followed by 1.0"
            )
        return self


__all__ = [
    "LOGICAL_LOAD_KINDS",
    "HprLoadKink",
    "HprLoadKinkKind",
    "HprLoadKinksResult",
]
