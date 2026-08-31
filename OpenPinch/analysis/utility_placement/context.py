"""Immutable numerical context for utility-placement evaluation."""

from __future__ import annotations

import math
from typing import Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from OpenPinch.contracts.utility_placement import (
    CoordinateKey,
    DecisionField,
    PhysicalCoordinateBound,
    PlacementFeasibilityEnvelope,
    PlacementPeriodEnvelope,
    PlacementUnitSystem,
    QuantityValue,
    TemplateBlueprintSet,
    UtilityLevelKind,
    UtilityPlacementBaseTarget,
    UtilityPlacementRequest,
    UtilitySide,
)

from .errors import PlacementContextError
from .normalization import convert_placement_value


class _FrozenContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def _finite(value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("value must be finite")
    return 0.0 if result == 0.0 else result


class ProcessEntropySlice(_FrozenContext):
    """One real-temperature process contribution used by entropy accounting."""

    interval_index: int
    side: UtilitySide
    temperature_in_kelvin: float
    temperature_out_kelvin: float
    available_duty: float
    heat_capacity_flow: float

    @field_validator("interval_index", mode="before")
    @classmethod
    def _validate_index_type(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("interval_index must be an integer")
        return value

    @field_validator("interval_index")
    @classmethod
    def _validate_index(cls, value: int) -> int:
        if value < 0:
            raise ValueError("interval_index must be non-negative")
        return value

    @field_validator(
        "temperature_in_kelvin",
        "temperature_out_kelvin",
        "available_duty",
        "heat_capacity_flow",
        mode="before",
    )
    @classmethod
    def _validate_number(cls, value: object) -> float:
        return _finite(value)  # type: ignore[arg-type]

    @model_validator(mode="after")
    def _validate_physical_values(self) -> Self:
        if self.temperature_in_kelvin <= 0.0 or self.temperature_out_kelvin <= 0.0:
            raise ValueError("entropy-slice temperatures must be positive kelvin")
        if self.available_duty < 0.0 or self.heat_capacity_flow < 0.0:
            raise ValueError(
                "entropy-slice duty and heat-capacity flow must be non-negative"
            )
        return self


class PlacementTargetSnapshot(_FrozenContext):
    """Detached targeting arrays needed to replay one process period."""

    shifted_temperatures: tuple[float, ...]
    real_temperatures: tuple[float, ...]
    hot_load_profile: tuple[float, ...]
    cold_load_profile: tuple[float, ...]
    real_hot_composite: tuple[float, ...]
    real_cold_composite: tuple[float, ...]
    hot_pinch_index: int
    cold_pinch_index: int
    entropy_slices: tuple[ProcessEntropySlice, ...]

    @field_validator(
        "shifted_temperatures",
        "real_temperatures",
        "hot_load_profile",
        "cold_load_profile",
        "real_hot_composite",
        "real_cold_composite",
        mode="before",
    )
    @classmethod
    def _validate_vector(cls, value: object) -> tuple[float, ...]:
        return tuple(_finite(item) for item in value)  # type: ignore[arg-type]

    @field_validator("hot_pinch_index", "cold_pinch_index", mode="before")
    @classmethod
    def _validate_pinch_type(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("pinch indices must be integers")
        return value

    @model_validator(mode="after")
    def _validate_alignment(self) -> Self:
        size = len(self.shifted_temperatures)
        if size < 2:
            raise ValueError("target snapshot must contain at least two temperatures")
        if any(
            len(vector) != size
            for vector in (self.hot_load_profile, self.cold_load_profile)
        ):
            raise ValueError("target snapshot arrays must align")
        if len(self.real_temperatures) < 2:
            raise ValueError(
                "real target snapshot must contain at least two temperatures"
            )
        if any(
            len(vector) != len(self.real_temperatures)
            for vector in (self.real_hot_composite, self.real_cold_composite)
        ):
            raise ValueError("real composite arrays must align")
        if (
            not 0 <= self.hot_pinch_index < size
            or not 0 <= self.cold_pinch_index < size
        ):
            raise ValueError("pinch indices must address the target snapshot")
        if any(value < 0.0 for value in self.hot_load_profile + self.cold_load_profile):
            raise ValueError("target load profiles must be non-negative")
        return self


class PlacementPeriodInput(_FrozenContext):
    """Detached process and physical-bound input for one weighted period."""

    period_id: str
    weight: float
    snapshot: PlacementTargetSnapshot
    residual_hot_duty: float
    residual_cold_duty: float
    ambient_temperature_kelvin: float
    coordinate_bounds: tuple[PhysicalCoordinateBound, ...]
    maximum_duties: tuple[tuple[str, float], ...] = ()
    fallback_temperature_span: float = 0.01
    fallback_hot_target_temperature: float | None = None
    fallback_cold_target_temperature: float | None = None

    @field_validator("period_id")
    @classmethod
    def _validate_period_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("period_id must not be empty")
        return normalized

    @field_validator(
        "weight",
        "residual_hot_duty",
        "residual_cold_duty",
        "ambient_temperature_kelvin",
        "fallback_temperature_span",
        mode="before",
    )
    @classmethod
    def _validate_scalar(cls, value: object) -> float:
        return _finite(value)  # type: ignore[arg-type]

    @model_validator(mode="after")
    def _validate_values(self) -> Self:
        if self.weight < 0.0:
            raise ValueError("period weight must be non-negative")
        if self.residual_hot_duty < 0.0 or self.residual_cold_duty < 0.0:
            raise ValueError("residual duties must be non-negative")
        if self.ambient_temperature_kelvin <= 0.0:
            raise ValueError("ambient temperature must be positive kelvin")
        names = tuple(name for name, _ in self.maximum_duties)
        if len(set(names)) != len(names):
            raise ValueError("maximum-duty utility names must be unique")
        if any(
            not name.strip() or not math.isfinite(value) or value < 0.0
            for name, value in self.maximum_duties
        ):
            raise ValueError("maximum duties must be named, finite, and non-negative")
        if self.fallback_temperature_span <= 0.0:
            raise ValueError("fallback temperature span must be positive")
        for temperature in (
            self.fallback_hot_target_temperature,
            self.fallback_cold_target_temperature,
        ):
            if temperature is not None and not math.isfinite(temperature):
                raise ValueError("fallback temperatures must be finite")
        return self


class UtilityPlacementContext(_FrozenContext):
    """Complete immutable numerical input to Unit 2 evaluation."""

    scope: UtilityPlacementBaseTarget
    base_target_id: str
    periods: tuple[PlacementPeriodInput, ...]
    envelope: PlacementFeasibilityEnvelope
    units: PlacementUnitSystem


def _expected_coordinates(blueprints: TemplateBlueprintSet) -> set[CoordinateKey]:
    expected: set[CoordinateKey] = set()
    for blueprint in blueprints.all:
        expected.add(
            CoordinateKey(
                template_key=blueprint.key,
                field=DecisionField.SUPPLY_TEMPERATURE,
            )
        )
        if blueprint.kind is UtilityLevelKind.SENSIBLE:
            expected.add(
                CoordinateKey(
                    template_key=blueprint.key,
                    field=DecisionField.TEMPERATURE_SPAN,
                )
            )
    return expected


def _raise_context(code: str, message: str, **context: object) -> None:
    raise PlacementContextError(code=code, message=message, **context)


def build_utility_placement_context(
    *,
    request: UtilityPlacementRequest,
    blueprints: TemplateBlueprintSet,
    scope: UtilityPlacementBaseTarget,
    base_target_id: str,
    periods: tuple[PlacementPeriodInput, ...],
) -> UtilityPlacementContext:
    """Validate and freeze resolved all-period numerical placement context."""
    if scope is UtilityPlacementBaseTarget.AUTO:
        _raise_context(
            "unresolved_scope",
            "Utility-placement context requires a resolved target scope.",
            scope=scope,
        )
    if not base_target_id.strip():
        _raise_context("missing_target_id", "Base target identity must not be empty.")
    if not periods:
        _raise_context("missing_periods", "At least one placement period is required.")

    period_ids = tuple(period.period_id for period in periods)
    if len(set(period_ids)) != len(period_ids):
        _raise_context(
            "duplicate_period", "Placement period identifiers must be unique."
        )
    if request.period_ids is not None and period_ids != request.period_ids:
        _raise_context(
            "period_selection_mismatch",
            "Resolved placement periods must match the requested period order.",
            details=(("requested", request.period_ids), ("resolved", period_ids)),
        )
    if not any(period.weight > 0.0 for period in periods):
        _raise_context(
            "missing_positive_period_weight",
            "At least one placement period weight must be positive.",
        )

    expected = _expected_coordinates(blueprints)
    envelope_periods: list[PlacementPeriodEnvelope] = []
    for period in periods:
        if period.ambient_temperature_kelvin <= 0.0:
            _raise_context(
                "invalid_ambient_temperature",
                "Each ambient temperature must be positive kelvin.",
                period_id=period.period_id,
            )
        if period.weight < 0.0:
            _raise_context(
                "negative_period_weight",
                "Placement period weights must be non-negative.",
                period_id=period.period_id,
            )
        observed = tuple(bound.coordinate for bound in period.coordinate_bounds)
        if len(observed) != len(set(observed)) or set(observed) != expected:
            _raise_context(
                "coordinate_coverage_mismatch",
                "Each period must provide every placement coordinate exactly once.",
                period_id=period.period_id,
                details=(
                    ("expected_count", len(expected)),
                    ("observed_count", len(observed)),
                ),
            )
        envelope_periods.append(
            PlacementPeriodEnvelope(
                period_id=period.period_id,
                weight=period.weight,
                coordinate_bounds=period.coordinate_bounds,
                residual_hot_duty=period.residual_hot_duty,
                residual_cold_duty=period.residual_cold_duty,
            )
        )

    minimum_separation = convert_placement_value(
        request.options.minimum_separation,
        canonical_unit=request.units.temperature_difference,
        default_unit=request.units.temperature_difference,
        field_path="options.minimum_separation",
    )
    envelope = PlacementFeasibilityEnvelope(
        periods=tuple(envelope_periods),
        minimum_separation=QuantityValue(
            value=minimum_separation,
            unit=request.units.temperature_difference,
        ),
        scope=scope,
        base_target_id=base_target_id,
        units=request.units,
    )
    return UtilityPlacementContext(
        scope=scope,
        base_target_id=base_target_id,
        periods=periods,
        envelope=envelope,
        units=request.units,
    )


__all__ = [
    "PlacementPeriodInput",
    "PlacementTargetSnapshot",
    "ProcessEntropySlice",
    "UtilityPlacementContext",
    "build_utility_placement_context",
]
