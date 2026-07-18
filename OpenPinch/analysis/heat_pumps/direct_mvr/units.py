"""Unit normalization for direct gas MVR inputs and outputs."""

from __future__ import annotations

import numpy as np

from ....domain.configuration import tol
from ....domain.stream import Stream
from ....domain.value import Value
from .models import (
    DEFAULT_ENTHALPY_UNIT,
    DEFAULT_HEAT_FLOW_UNIT,
    DEFAULT_PRESSURE_UNIT,
    DEFAULT_TEMPERATURE_UNIT,
    DirectGasMVROutputUnits,
    DirectGasMVRStageResult,
)


def stage_mass_flow(stage: DirectGasMVRStageResult) -> float:
    """Resolve a stage's hot-side mass flow from explicit or energy data."""
    if stage.hot_mass_flow > tol:
        return stage.hot_mass_flow
    enthalpy_delta = enthalpy_delta_to_j_per_kg(
        stage.h_hot_supply,
        stage.h_target,
        stage.enthalpy_unit,
    )
    if enthalpy_delta <= tol:
        return 0.0
    heat_flow_watts = Value(stage.heat_flow, stage.heat_flow_unit).to("W").value
    return heat_flow_watts / enthalpy_delta


def stage_pressure_to_pascal(value: float, pressure_unit: str) -> float:
    return float(Value(value, pressure_unit).to("Pa").value)


def value_at_index(
    value,
    index: int,
    *,
    unit: str | None = None,
) -> float | None:
    """Resolve one scalar or period-indexed Value-like magnitude."""
    if value is None:
        return None
    try:
        selected = value[index]
    except Exception:
        selected = value
    if isinstance(selected, Value):
        if unit is not None:
            selected = selected.to(unit)
        return float(selected.value)
    return float(selected)


def to_kelvin(temperature_celsius: float) -> float:
    return float(Value(temperature_celsius, "degC").to("K").value)


def to_degrees_celsius(temperature_kelvin: float) -> float:
    return float(Value(temperature_kelvin, "K").to("degC").value)


def to_pascal(pressure_kilopascal: float) -> float:
    return float(Value(pressure_kilopascal, "kPa").to("Pa").value)


def to_kilopascal(pressure_pascal: float) -> float:
    return float(Value(pressure_pascal, "Pa").to("kPa").value)


def to_watt(power_kilowatt: float) -> float:
    return float(Value(power_kilowatt, "kW").to("W").value)


def to_kilowatt(power_watt: float) -> float:
    return float(Value(power_watt, "W").to("kW").value)


def to_j_per_kg(enthalpy_kj_per_kg: float) -> float:
    return float(Value(enthalpy_kj_per_kg, "kJ/kg").to("J/kg").value)


def to_kj_per_kg(enthalpy_j_per_kg: float) -> float:
    return float(Value(enthalpy_j_per_kg, "J/kg").to("kJ/kg").value)


def profile_to_output_units(
    profile: np.ndarray,
    output_units: DirectGasMVROutputUnits,
) -> np.ndarray:
    """Convert enthalpy/temperature profile columns to public output units."""
    converted = np.asarray(profile, dtype=float).copy()
    converted[:, 0] = Value(converted[:, 0], "J/kg").to(output_units.enthalpy).value
    converted[:, 1] = Value(converted[:, 1], "degC").to(output_units.temperature).value
    return converted


def convert_value(value: float, source_unit: str, target_unit: str) -> float:
    return float(Value(value, source_unit).to(target_unit).value)


def source_output_units(stream: Stream) -> DirectGasMVROutputUnits:
    """Derive public result units from the source stream's configured Values."""
    return DirectGasMVROutputUnits(
        temperature=stream_value_unit(
            stream.supply_temperature, DEFAULT_TEMPERATURE_UNIT
        ),
        pressure=stream_value_unit(stream.supply_pressure, DEFAULT_PRESSURE_UNIT),
        enthalpy=stream_value_unit(
            stream.supply_enthalpy or stream.target_enthalpy,
            DEFAULT_ENTHALPY_UNIT,
        ),
        heat_flow=stream_value_unit(stream.heat_flow, DEFAULT_HEAT_FLOW_UNIT),
    )


def stream_value_unit(value, fallback: str) -> str:
    return str(getattr(value, "unit", None) or fallback)


def enthalpy_delta_to_j_per_kg(
    enthalpy_start: float,
    enthalpy_end: float,
    enthalpy_unit: str,
) -> float:
    delta = abs(float(enthalpy_start) - float(enthalpy_end))
    return float(Value(delta, enthalpy_unit).to("J/kg").value)


def is_finite_integer(value) -> bool:
    """Return whether a value can represent one finite integer."""
    try:
        numeric_value = float(value)
    except TypeError, ValueError:
        return False
    return bool(np.isfinite(numeric_value) and numeric_value.is_integer())


__all__: list[str] = []
