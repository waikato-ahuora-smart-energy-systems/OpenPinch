"""CoolProp state calculations for direct gas MVR stages."""

from __future__ import annotations

import numpy as np
from CoolProp.CoolProp import PropsSI

from ....domain.stream import Stream
from .units import (
    to_degrees_celsius,
    to_j_per_kg,
    to_kelvin,
    to_pascal,
    value_at_index,
)


def source_enthalpies(
    stream: Stream,
    fluid: str,
    supply_pressure_kpa: float,
    supply_temperature_c: float,
    target_temperature_c: float,
    period_index: int,
) -> tuple[float, float]:
    """Resolve source enthalpies from input data or CoolProp states."""
    supply_enthalpy = value_at_index(stream.h_supply, period_index, unit="kJ/kg")
    target_enthalpy = value_at_index(stream.h_target, period_index, unit="kJ/kg")
    if supply_enthalpy is not None and target_enthalpy is not None:
        return to_j_per_kg(supply_enthalpy), to_j_per_kg(target_enthalpy)
    return (
        state_property_at_temperature_pressure(
            fluid=fluid,
            output="H",
            t_c=supply_temperature_c,
            p_kpa=supply_pressure_kpa,
            saturated_quality=1.0,
        ),
        state_property_at_temperature_pressure(
            fluid=fluid,
            output="H",
            t_c=target_temperature_c,
            p_kpa=supply_pressure_kpa,
            saturated_quality=0.0,
        ),
    )


def state_property_at_temperature_pressure(
    *,
    fluid: str,
    output: str,
    t_c: float,
    p_kpa: float,
    saturated_quality: float,
) -> float:
    """Evaluate a state, falling back to exact saturation-quality input."""
    temperature_kelvin = to_kelvin(t_c)
    pressure_pascal = to_pascal(p_kpa)
    try:
        return PropsSI(
            output,
            "T",
            temperature_kelvin,
            "P",
            pressure_pascal,
            fluid,
        )
    except ValueError:
        try:
            saturation_temperature = PropsSI(
                "T",
                "P",
                pressure_pascal,
                "Q",
                saturated_quality,
                fluid,
            )
        except Exception:
            raise
        if abs(temperature_kelvin - saturation_temperature) > 0.05:
            raise
        return PropsSI(
            output,
            "P",
            pressure_pascal,
            "Q",
            saturated_quality,
            fluid,
        )


def find_pressure_for_actual_discharge(
    *,
    fluid: str,
    inlet_pressure: float,
    inlet_enthalpy: float,
    inlet_entropy: float,
    target_discharge_kelvin: float,
    compressor_efficiency: float,
) -> tuple[float, float, float]:
    """Find discharge pressure for an actual-temperature target by bisection."""
    pressure_low = inlet_pressure * 1.0001
    pressure_high = inlet_pressure * 1.2
    _, temperature_high = actual_outlet_at_pressure(
        fluid,
        pressure_high,
        inlet_enthalpy,
        inlet_entropy,
        compressor_efficiency,
    )
    for _ in range(80):
        if temperature_high >= target_discharge_kelvin:
            break
        pressure_high *= 1.5
        _, temperature_high = actual_outlet_at_pressure(
            fluid,
            pressure_high,
            inlet_enthalpy,
            inlet_entropy,
            compressor_efficiency,
        )
    else:
        raise ValueError("Could not find a feasible MVR discharge pressure.")

    for _ in range(80):
        pressure_mid = 0.5 * (pressure_low + pressure_high)
        _, temperature_mid = actual_outlet_at_pressure(
            fluid,
            pressure_mid,
            inlet_enthalpy,
            inlet_entropy,
            compressor_efficiency,
        )
        if temperature_mid < target_discharge_kelvin:
            pressure_low = pressure_mid
        else:
            pressure_high = pressure_mid
    outlet_enthalpy, outlet_temperature = actual_outlet_at_pressure(
        fluid,
        pressure_high,
        inlet_enthalpy,
        inlet_entropy,
        compressor_efficiency,
    )
    return pressure_high, outlet_enthalpy, outlet_temperature


def actual_outlet_at_pressure(
    fluid: str,
    outlet_pressure: float,
    inlet_enthalpy: float,
    inlet_entropy: float,
    compressor_efficiency: float,
) -> tuple[float, float]:
    """Return actual outlet enthalpy and temperature at a pressure."""
    isentropic_enthalpy = PropsSI(
        "H",
        "P",
        outlet_pressure,
        "S",
        inlet_entropy,
        fluid,
    )
    outlet_enthalpy = (
        inlet_enthalpy + (isentropic_enthalpy - inlet_enthalpy) / compressor_efficiency
    )
    outlet_temperature = PropsSI(
        "T",
        "P",
        outlet_pressure,
        "H",
        outlet_enthalpy,
        fluid,
    )
    return outlet_enthalpy, outlet_temperature


def build_cooling_temperature_enthalpy_curve(
    *,
    fluid: str,
    outlet_pressure: float,
    hot_supply_enthalpy: float,
    target_enthalpy: float,
) -> np.ndarray:
    """Build the stage cooling curve in J/kg and degrees Celsius."""
    if hot_supply_enthalpy <= target_enthalpy:
        raise ValueError("Compressed MVR stage has no heat-release enthalpy range.")
    enthalpy_values = profile_enthalpy_values(
        fluid=fluid,
        outlet_pressure=outlet_pressure,
        hot_supply_enthalpy=hot_supply_enthalpy,
        target_enthalpy=target_enthalpy,
    )
    points = [
        [
            enthalpy,
            to_degrees_celsius(
                PropsSI("T", "P", outlet_pressure, "H", enthalpy, fluid)
            ),
        ]
        for enthalpy in enthalpy_values
    ]
    return np.asarray(points, dtype=float)


def profile_enthalpy_values(
    *,
    fluid: str,
    outlet_pressure: float,
    hot_supply_enthalpy: float,
    target_enthalpy: float,
) -> np.ndarray:
    """Return cooling-curve enthalpies including saturation breakpoints."""
    breakpoints = [float(hot_supply_enthalpy), float(target_enthalpy)]
    try:
        critical_pressure = PropsSI("PCRIT", fluid)
        if outlet_pressure < critical_pressure:
            saturated_vapour = PropsSI("H", "P", outlet_pressure, "Q", 1.0, fluid)
            saturated_liquid = PropsSI("H", "P", outlet_pressure, "Q", 0.0, fluid)
            for saturation_enthalpy in (saturated_vapour, saturated_liquid):
                if target_enthalpy < saturation_enthalpy < hot_supply_enthalpy:
                    breakpoints.append(float(saturation_enthalpy))
    except Exception:
        pass

    ordered = sorted(set(breakpoints), reverse=True)
    values: list[float] = []
    for upper, lower in zip(ordered[:-1], ordered[1:]):
        segment = np.linspace(upper, lower, 31)
        if values:
            segment = segment[1:]
        values.extend(float(enthalpy) for enthalpy in segment)
    return np.asarray(values, dtype=float)


__all__: list[str] = []
