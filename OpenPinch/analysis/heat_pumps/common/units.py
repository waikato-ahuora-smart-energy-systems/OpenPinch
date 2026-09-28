"""Period-value and unit conversion helpers shared by heat-pump modules."""

from __future__ import annotations

from ....domain.value import Value

__all__ = [
    "to_degrees_celsius",
    "to_kelvin",
    "to_kilopascal",
    "value_at_index",
]


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


def to_kilopascal(pressure_pascal: float) -> float:
    return float(Value(pressure_pascal, "Pa").to("kPa").value)
