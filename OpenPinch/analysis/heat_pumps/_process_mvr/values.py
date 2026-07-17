"""Period-value and unit helpers for Process MVR orchestration."""

from __future__ import annotations

from ....domain.value import Value


def required_period_value(value, index: int, name: str, selector) -> float:
    """Resolve one required source-stream value for an operating period."""
    unit = "degC" if name.startswith("t_") else None
    selected = value_at_index(value, index, unit=unit)
    if selected is None:
        raise ValueError(f"MVR source stream {selector!r} requires {name}.")
    return selected


def period_values_or_scalar(values: list[float]):
    """Collapse a one-period list while retaining multiperiod values."""
    return values[0] if len(values) == 1 else values


def value_at_index(
    value,
    index: int,
    *,
    unit: str | None = None,
) -> float | None:
    """Resolve a scalar or indexed Value-like object to one float."""
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


def to_kilopascals(pressure_pascal: float) -> float:
    return float(Value(pressure_pascal, "Pa").to("kPa").value)


def enthalpy_delta_to_j_per_kg(
    enthalpy_start: float,
    enthalpy_end: float,
    enthalpy_unit: str,
) -> float:
    delta = abs(float(enthalpy_start) - float(enthalpy_end))
    return float(Value(delta, enthalpy_unit).to("J/kg").value)


__all__: list[str] = []
