"""Stable entropy and exergy kernels for utility placement."""

from __future__ import annotations

import math

import numpy as np

from OpenPinch.analysis.graphs.composite import clean_composite_curve_ends
from OpenPinch.analysis.targeting.area_cost import get_balanced_CC
from OpenPinch.analysis.targeting.cascade import get_utility_heat_cascade
from OpenPinch.analysis.targeting.temperature_driving_force import (
    get_temperature_driving_forces,
)
from OpenPinch.contracts.utility_placement import (
    QuantityValue,
    ThermodynamicCostBreakdown,
    UtilityPlacementRequest,
)
from OpenPinch.domain.enums import ProblemTableLabel
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.stream_collection import StreamCollection

from .allocation import PlacementPeriodAllocation
from .context import PlacementPeriodInput
from .errors import PlacementThermodynamicError


def stream_entropy_change(
    duty: float,
    inlet_temperature_kelvin: float,
    outlet_temperature_kelvin: float,
) -> float:
    """Return the stream entropy-rate change for a positive heat-duty magnitude."""
    values = (duty, inlet_temperature_kelvin, outlet_temperature_kelvin)
    if any(not math.isfinite(value) for value in values):
        raise PlacementThermodynamicError(
            code="nonfinite_entropy_input",
            message="Entropy inputs must be finite.",
        )
    if duty < 0.0:
        raise PlacementThermodynamicError(
            code="negative_entropy_duty",
            message="Entropy duty must be non-negative.",
        )
    if inlet_temperature_kelvin <= 0.0 or outlet_temperature_kelvin <= 0.0:
        raise PlacementThermodynamicError(
            code="nonpositive_kelvin",
            message="Entropy temperatures must be positive kelvin.",
        )
    if duty == 0.0:
        return 0.0
    delta = outlet_temperature_kelvin - inlet_temperature_kelvin
    scale = max(abs(inlet_temperature_kelvin), abs(outlet_temperature_kelvin))
    if abs(delta) <= 1e-12 * scale:
        limit = duty / (
            (inlet_temperature_kelvin + outlet_temperature_kelvin) / 2.0
        )
        return math.copysign(limit, delta) if delta != 0.0 else limit
    heat_capacity_flow = duty / abs(delta)
    return heat_capacity_flow * math.log1p(delta / inlet_temperature_kelvin)


def _entropy_magnitude(
    duty: float,
    first_temperature_kelvin: float,
    second_temperature_kelvin: float,
) -> float:
    low, high = sorted((first_temperature_kelvin, second_temperature_kelvin))
    return abs(stream_entropy_change(duty, low, high))


def _close_numerical_heat_balance(
    hot_heat_loads: np.ndarray,
    cold_heat_loads: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove only table-rounding drift from an otherwise balanced curve pair."""
    hot_extent = float(np.ptp(hot_heat_loads))
    cold_extent = float(np.ptp(cold_heat_loads))
    reference = max(hot_extent, cold_extent, 1.0)
    if abs(hot_extent - cold_extent) > 1e-4 * reference:
        return hot_heat_loads, cold_heat_loads
    if hot_extent == 0.0 or cold_extent == 0.0:
        return hot_heat_loads, cold_heat_loads

    common_extent = (hot_extent + cold_extent) / 2.0

    def normalize(values: np.ndarray, extent: float) -> np.ndarray:
        return (values - float(np.min(values))) * (common_extent / extent)

    return normalize(hot_heat_loads, hot_extent), normalize(
        cold_heat_loads, cold_extent
    )


def balanced_composite_entropy_generation(
    *,
    hot_temperatures_celsius,
    hot_heat_loads,
    cold_temperatures_celsius,
    cold_heat_loads,
) -> tuple[float, float, float]:
    """Return hot, cold, and generated entropy rates from balanced composites."""
    try:
        hot_temperature = np.asarray(hot_temperatures_celsius, dtype=float)
        hot_heat = np.asarray(hot_heat_loads, dtype=float)
        cold_temperature = np.asarray(cold_temperatures_celsius, dtype=float)
        cold_heat = np.asarray(cold_heat_loads, dtype=float)
        arrays = (hot_temperature, hot_heat, cold_temperature, cold_heat)
        if any(values.ndim != 1 for values in arrays):
            raise ValueError("Balanced composite arrays must be one-dimensional.")
        if any(not np.all(np.isfinite(values)) for values in arrays):
            raise ValueError("Balanced composite arrays must be finite.")
        if hot_heat.size > 0 and cold_heat.size > 0:
            hot_heat, cold_heat = _close_numerical_heat_balance(hot_heat, cold_heat)
        intervals = get_temperature_driving_forces(
            hot_temperature,
            hot_heat,
            cold_temperature,
            cold_heat,
        )
    except (TypeError, ValueError) as exc:
        raise PlacementThermodynamicError(
            code="invalid_balanced_composite",
            message=str(exc) or "Balanced composite curves are invalid.",
        ) from exc

    hot_terms: list[float] = []
    cold_terms: list[float] = []
    for duty, hot_first, hot_second, cold_first, cold_second in zip(
        intervals["dh_vals"],
        intervals["t_h1"],
        intervals["t_h2"],
        intervals["t_c1"],
        intervals["t_c2"],
        strict=True,
    ):
        hot_terms.append(
            -_entropy_magnitude(
                float(duty),
                float(hot_first) + 273.15,
                float(hot_second) + 273.15,
            )
        )
        cold_terms.append(
            _entropy_magnitude(
                float(duty),
                float(cold_first) + 273.15,
                float(cold_second) + 273.15,
            )
        )
    hot_entropy = math.fsum(hot_terms)
    cold_entropy = math.fsum(cold_terms)
    generation = math.fsum((hot_entropy, cold_entropy))
    noise_limit = 1e-12 * max(abs(hot_entropy), abs(cold_entropy), 1.0)
    if generation < -noise_limit:
        raise PlacementThermodynamicError(
            code="negative_entropy_generation",
            message="Balanced composite curves produce negative entropy generation.",
            details=(("generation", generation), ("noise_limit", noise_limit)),
        )
    return hot_entropy, cold_entropy, max(generation, 0.0)


def _level_stream(level) -> Stream:
    return Stream(
        name=level.template_key.name,
        supply_temperature=level.supply_temperature,
        target_temperature=level.target_temperature,
        heat_flow=level.allocated_duty,
        is_process_stream=False,
    )


def _interpolate_profile(
    source_temperatures: tuple[float, ...],
    source_profile: tuple[float, ...],
    target_temperatures: tuple[float, ...],
) -> tuple[float, ...]:
    source_t = np.asarray(source_temperatures, dtype=float)
    source_h = np.asarray(source_profile, dtype=float)
    target_t = np.asarray(target_temperatures, dtype=float)
    return tuple(
        float(value)
        for value in np.interp(target_t[::-1], source_t[::-1], source_h[::-1])[::-1]
    )


def _balanced_composite_curves(period, allocation):
    active_levels = tuple(
        level
        for level in allocation.hot_levels + allocation.cold_levels
        if level.allocated_duty > 0.0
    )
    utility_breakpoints = {
        temperature
        for level in active_levels
        for temperature in (level.supply_temperature, level.target_temperature)
    }
    temperatures = tuple(
        sorted(
            set(period.snapshot.real_temperatures) | utility_breakpoints,
            reverse=True,
        )
    )
    process_hot = _interpolate_profile(
        period.snapshot.real_temperatures,
        period.snapshot.real_hot_composite,
        temperatures,
    )
    process_cold = _interpolate_profile(
        period.snapshot.real_temperatures,
        period.snapshot.real_cold_composite,
        temperatures,
    )
    hot_streams = StreamCollection(
        [_level_stream(level) for level in allocation.hot_levels]
    )
    cold_streams = StreamCollection(
        [_level_stream(level) for level in allocation.cold_levels]
    )
    utility_updates = get_utility_heat_cascade(
        np.asarray(temperatures, dtype=float),
        hot_streams,
        cold_streams,
        is_shifted=False,
    )["updates"]
    balanced = get_balanced_CC(
        T_col=np.asarray(temperatures, dtype=float),
        H_hot=np.asarray(process_hot, dtype=float),
        H_cold=np.asarray(process_cold, dtype=float),
        H_hot_ut=utility_updates[ProblemTableLabel.H_HOT_UT],
        H_cold_ut=utility_updates[ProblemTableLabel.H_COLD_UT],
    )["updates"]
    hot_temperature, hot_heat = clean_composite_curve_ends(
        np.asarray(temperatures, dtype=float),
        np.asarray(balanced[ProblemTableLabel.H_HOT_BAL], dtype=float),
    )
    cold_temperature, cold_heat = clean_composite_curve_ends(
        np.asarray(temperatures, dtype=float),
        np.asarray(balanced[ProblemTableLabel.H_COLD_BAL], dtype=float),
    )
    return hot_temperature, hot_heat, cold_temperature, cold_heat


def _process_entropy(period: PlacementPeriodInput) -> float:
    terms = []
    for item in period.snapshot.entropy_slices:
        if item.temperature_in_kelvin == item.temperature_out_kelvin:
            magnitude = item.available_duty / item.temperature_in_kelvin
            terms.append(-magnitude if item.side.value == "hot" else magnitude)
        else:
            terms.append(
                stream_entropy_change(
                    item.available_duty,
                    item.temperature_in_kelvin,
                    item.temperature_out_kelvin,
                )
            )
    return math.fsum(terms)


def evaluate_thermodynamic_cost(
    *,
    request: UtilityPlacementRequest,
    period: PlacementPeriodInput,
    allocation: PlacementPeriodAllocation,
) -> ThermodynamicCostBreakdown:
    """Evaluate physical entropy generation from balanced composite curves."""
    hot_temperature, hot_heat, cold_temperature, cold_heat = (
        _balanced_composite_curves(period, allocation)
    )
    _, _, generation = balanced_composite_entropy_generation(
        hot_temperatures_celsius=hot_temperature,
        hot_heat_loads=hot_heat,
        cold_temperatures_celsius=cold_temperature,
        cold_heat_loads=cold_heat,
    )
    process_entropy = _process_entropy(period)
    utility_entropy = generation - process_entropy
    exergy = period.ambient_temperature_kelvin * generation
    units = request.units
    return ThermodynamicCostBreakdown(
        utility_entropy=QuantityValue(value=utility_entropy, unit=units.entropy),
        process_entropy=QuantityValue(value=process_entropy, unit=units.entropy),
        total_entropy_generation=QuantityValue(value=generation, unit=units.entropy),
        ambient_temperature=QuantityValue(
            value=period.ambient_temperature_kelvin,
            unit="K",
        ),
        exergy_destruction=QuantityValue(value=exergy, unit=units.exergy),
    )


__all__ = [
    "evaluate_thermodynamic_cost",
    "balanced_composite_entropy_generation",
    "stream_entropy_change",
]
