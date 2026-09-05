"""Strict plain-data contracts for HPR performance maps."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

type JsonValue = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)


class _FrozenContract(BaseModel):
    """Strict immutable base shared by performance-map values."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def _require_name(value: str, label: str) -> str:
    if not value.strip():
        raise ValueError(f"{label} must not be empty")
    return value


def _canonical_numeric_tuple(
    value: object,
    *,
    label: str,
    fraction: bool = False,
) -> tuple[float, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence")
    values = tuple(float(item) for item in value)
    if not values:
        raise ValueError(f"{label} must not be empty")
    if not all(math.isfinite(item) for item in values):
        raise ValueError(f"{label} values must be finite")
    if fraction and any(item <= 0.0 or item > 1.0 for item in values):
        raise ValueError(f"{label} values must be in the interval (0, 1]")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} values must be unique")
    return tuple(sorted(values))


def _validate_json_value(value: object, *, path: str) -> JsonValue:
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} numbers must be finite")
        return value
    if isinstance(value, list):
        return [
            _validate_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            normalized[key] = _validate_json_value(item, path=f"{path}.{key}")
        return normalized
    raise ValueError(f"{path} contains a non-JSON value of type {type(value).__name__}")


class HprPerformanceMapRequest(_FrozenContract):
    """Canonical grid request consumed by an HPR map generator."""

    map_id: str
    source_temperatures: tuple[float, ...]
    sink_temperatures: tuple[float, ...]
    load_fractions: tuple[float, ...]
    reference_capacity: float | None = Field(default=None, gt=0.0)

    @field_validator("map_id")
    @classmethod
    def _validate_map_id(cls, value: str) -> str:
        return _require_name(value, "map_id")

    @field_validator("source_temperatures", mode="before")
    @classmethod
    def _validate_source_temperatures(cls, value: object) -> tuple[float, ...]:
        return _canonical_numeric_tuple(value, label="source_temperatures")

    @field_validator("sink_temperatures", mode="before")
    @classmethod
    def _validate_sink_temperatures(cls, value: object) -> tuple[float, ...]:
        return _canonical_numeric_tuple(value, label="sink_temperatures")

    @field_validator("load_fractions", mode="before")
    @classmethod
    def _validate_load_fractions(cls, value: object) -> tuple[float, ...]:
        return _canonical_numeric_tuple(
            value,
            label="load_fractions",
            fraction=True,
        )


class HprPerformanceMapUnits(_FrozenContract):
    """Closed canonical units for performance-map schema 1.0."""

    source_temperature: Literal["degC"]
    sink_temperature: Literal["degC"]
    q_source: Literal["kW"]
    q_sink: Literal["kW"]
    electric_power: Literal["kW"]


class HprPerformancePoint(_FrozenContract):
    """One active point on a fixed-temperature part-load curve."""

    name: str
    curve_id: str
    source_temperature: float
    sink_temperature: float
    load_fraction: float = Field(gt=0.0, le=1.0)
    q_source: float = Field(ge=0.0)
    q_sink: float = Field(ge=0.0)
    electric_power: float = Field(gt=0.0)
    cop: float = Field(gt=0.0)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        return _require_name(value, "performance point name")

    @field_validator("curve_id")
    @classmethod
    def _validate_curve_id(cls, value: str) -> str:
        return _require_name(value, "curve_id")


class HprPerformanceMap(_FrozenContract):
    """Complete versioned physical HPR performance map."""

    schema_version: Literal["1.0"]
    map_id: str
    mode: Literal["heat_pump", "refrigeration"]
    units: HprPerformanceMapUnits
    reference_capacity: float = Field(gt=0.0)
    reference_capacity_basis: Literal["q_sink", "q_source"]
    interpolation_topology: Literal["ordered_part_load_curve"]
    thermodynamic_backend: str
    model_id: str
    provenance: dict[str, JsonValue]
    points: tuple[HprPerformancePoint, ...] = Field(min_length=1)
    cop_convention: Literal["heating", "cooling"]
    energy_balance_tolerance: float = Field(default=1e-6, ge=0.0)
    temperature_match_tolerance: float = Field(default=1e-6, ge=0.0)

    @field_validator("map_id")
    @classmethod
    def _validate_map_id(cls, value: str) -> str:
        return _require_name(value, "map_id")

    @field_validator("thermodynamic_backend")
    @classmethod
    def _validate_backend(cls, value: str) -> str:
        return _require_name(value, "thermodynamic_backend")

    @field_validator("model_id")
    @classmethod
    def _validate_model_id(cls, value: str) -> str:
        return _require_name(value, "model_id")

    @field_validator("provenance", mode="before")
    @classmethod
    def _validate_provenance(cls, value: object) -> dict[str, JsonValue]:
        if not isinstance(value, Mapping):
            raise ValueError("provenance must be a mapping")
        normalized = cast(
            dict[str, JsonValue],
            _validate_json_value(value, path="provenance"),
        )
        if not normalized:
            raise ValueError("provenance must not be empty")
        return normalized

    @model_validator(mode="after")
    def _validate_map_semantics(self) -> Self:
        expected_basis = "q_sink" if self.mode == "heat_pump" else "q_source"
        if self.reference_capacity_basis != expected_basis:
            raise ValueError(
                f"{self.mode} reference_capacity_basis must be {expected_basis!r}"
            )
        expected_convention = "heating" if self.mode == "heat_pump" else "cooling"
        if self.cop_convention != expected_convention:
            raise ValueError(
                f"{self.mode} cop_convention must be {expected_convention!r}"
            )

        names: set[str] = set()
        coordinates: set[tuple[str, float, float, float]] = set()
        points_by_curve: dict[str, list[HprPerformancePoint]] = {}
        for point in self.points:
            if point.name in names:
                raise ValueError("performance point names must be unique per map")
            names.add(point.name)
            coordinate = (
                point.curve_id,
                point.source_temperature,
                point.sink_temperature,
                point.load_fraction,
            )
            if coordinate in coordinates:
                raise ValueError("performance point coordinates must be unique")
            coordinates.add(coordinate)
            points_by_curve.setdefault(point.curve_id, []).append(point)
            self._validate_point_physics(point)

        for curve_id, curve_points in points_by_curve.items():
            source_temperatures = {point.source_temperature for point in curve_points}
            sink_temperatures = {point.sink_temperature for point in curve_points}
            if len(source_temperatures) != 1 or len(sink_temperatures) != 1:
                raise ValueError(
                    f"curve {curve_id!r} must use one source/sink temperature pair"
                )
            load_fractions = [point.load_fraction for point in curve_points]
            if any(
                current >= next_value
                for current, next_value in zip(
                    load_fractions,
                    load_fractions[1:],
                )
            ):
                raise ValueError(
                    f"curve {curve_id!r} load_fraction values must be strictly "
                    "increasing"
                )

        canonical_points = tuple(
            sorted(
                self.points,
                key=lambda point: (
                    point.source_temperature,
                    point.sink_temperature,
                    point.load_fraction,
                    point.curve_id,
                    point.name,
                ),
            )
        )
        if self.points != canonical_points:
            raise ValueError(
                "points must use canonical temperature order with strictly "
                "increasing load_fraction values"
            )
        return self

    def _validate_point_physics(self, point: HprPerformancePoint) -> None:
        residual = abs(point.q_sink - point.q_source - point.electric_power)
        if residual > self.energy_balance_tolerance:
            raise ValueError(
                f"performance point {point.name!r} violates "
                "q_sink = q_source + electric_power"
            )
        useful_duty = (
            point.q_sink
            if self.reference_capacity_basis == "q_sink"
            else point.q_source
        )
        expected_duty = point.load_fraction * self.reference_capacity
        if abs(useful_duty - expected_duty) > self.energy_balance_tolerance:
            raise ValueError(
                f"performance point {point.name!r} useful duty must equal "
                "load_fraction * reference_capacity"
            )
        expected_cop = useful_duty / point.electric_power
        if abs(point.cop - expected_cop) > self.energy_balance_tolerance:
            raise ValueError(
                f"performance point {point.name!r} violates {self.cop_convention} COP"
            )


__all__ = [
    "HprPerformanceMap",
    "HprPerformanceMapRequest",
    "HprPerformanceMapUnits",
    "HprPerformancePoint",
    "JsonValue",
]
