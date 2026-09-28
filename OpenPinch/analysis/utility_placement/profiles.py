"""Numerical target-profile helpers for utility placement.

These functions turn solved problem tables and physical process streams into
the load profiles, entropy slices and coordinate bounds consumed by the
utility-placement model. They depend only on domain, contract and analysis
types, so the application layer can reuse them without the reverse import.
"""

from __future__ import annotations

import math

from OpenPinch.analysis.targeting.grand_composite import (
    get_seperated_gcc_heat_load_profiles,
)
from OpenPinch.contracts.utility_placement import (
    CoordinateKey,
    DecisionField,
    PhysicalCoordinateBound,
    QuantityInterval,
    TemplateBlueprintSet,
    UtilityLevelKind,
    UtilityPlacementRequest,
    UtilitySide,
)
from OpenPinch.domain._value.resolution import get_scalar_value
from OpenPinch.domain.configuration import C_to_K
from OpenPinch.domain.enums import ProblemTableLabel

from .context import ProcessEntropySlice
from .errors import PlacementContextError


def _finite_tuple(values) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _load_profiles(
    problem_table,
    *,
    net_label: ProblemTableLabel | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if net_label is None:
        hot = tuple(
            abs(value)
            for value in _finite_tuple(problem_table[ProblemTableLabel.H_NET_COLD])
        )
        cold = tuple(
            abs(value)
            for value in _finite_tuple(problem_table[ProblemTableLabel.H_NET_HOT])
        )
        if all(math.isfinite(value) for value in hot + cold):
            return hot, cold
        net = problem_table[ProblemTableLabel.H_NET_A]
        if not all(math.isfinite(float(value)) for value in net):
            net = problem_table[ProblemTableLabel.H_NET]
    else:
        net = problem_table[net_label]
    updates = get_seperated_gcc_heat_load_profiles(
        T_col=problem_table[ProblemTableLabel.T],
        H_net=net,
    )["updates"]
    return (
        tuple(
            abs(value) for value in _finite_tuple(updates[ProblemTableLabel.H_NET_COLD])
        ),
        tuple(
            abs(value) for value in _finite_tuple(updates[ProblemTableLabel.H_NET_HOT])
        ),
    )


def _calibrate_profile(
    profile: tuple[float, ...],
    *,
    residual_duty: float,
) -> tuple[float, ...]:
    """Match rounded problem-table coordinates to the exact target duty."""
    peak = max(profile, default=0.0)
    if peak == 0.0:
        if residual_duty == 0.0:
            return profile
        raise PlacementContextError(
            code="incomplete_load_profile",
            message="Target load profile cannot represent the residual duty.",
        )
    factor = residual_duty / peak
    return tuple(value * factor for value in profile)


def _process_entropy_slices(zone, period_idx: int) -> tuple[ProcessEntropySlice, ...]:
    """Extract real-temperature entropy inputs from physical process streams."""
    slices: list[ProcessEntropySlice] = []
    for side, streams in (
        (UtilitySide.HOT, zone.hot_streams),
        (UtilitySide.COLD, zone.cold_streams),
    ):
        for stream in streams:
            if not stream.is_active:
                continue
            parts = stream.segments or (stream,)
            for part in parts:
                supply = get_scalar_value(
                    part.supply_temperature,
                    period_idx=period_idx,
                )
                target = get_scalar_value(
                    part.target_temperature,
                    period_idx=period_idx,
                )
                raw_duty = get_scalar_value(part.heat_flow, period_idx=period_idx)
                if supply is None or target is None or raw_duty is None:
                    raise PlacementContextError(
                        code="incomplete_process_entropy_input",
                        message=(
                            "Process stream entropy input requires temperatures "
                            "and duty."
                        ),
                        details=(("stream", stream.name),),
                    )
                duty = abs(float(raw_duty))
                if duty == 0.0:
                    continue
                temperature_in = float(supply) + C_to_K
                temperature_out = float(target) + C_to_K
                span = abs(temperature_out - temperature_in)
                slices.append(
                    ProcessEntropySlice(
                        interval_index=len(slices),
                        side=side,
                        temperature_in_kelvin=temperature_in,
                        temperature_out_kelvin=temperature_out,
                        available_duty=duty,
                        heat_capacity_flow=duty / span if span > 0.0 else 0.0,
                    )
                )
    return tuple(slices)


def _coordinate_bounds(
    request: UtilityPlacementRequest,
    blueprints: TemplateBlueprintSet,
    temperatures: tuple[float, ...],
    *,
    hot_profile: tuple[float, ...] | None = None,
    cold_profile: tuple[float, ...] | None = None,
) -> tuple[PhysicalCoordinateBound, ...]:
    def support(profile: tuple[float, ...] | None) -> tuple[float, ...]:
        if profile is None:
            return temperatures
        if len(profile) != len(temperatures):
            raise PlacementContextError(
                code="profile_temperature_mismatch",
                message="Residual profile and temperature coordinates must align.",
            )
        active = tuple(
            temperature
            for index, temperature in enumerate(temperatures)
            if (index > 0 and abs(profile[index] - profile[index - 1]) > 1e-12)
            or (
                index < len(profile) - 1
                and abs(profile[index] - profile[index + 1]) > 1e-12
            )
        )
        return active or temperatures

    separation = request.options.minimum_separation.value
    level_count = request.isothermal_level_count + request.sensible_level_count
    hottest = max(temperatures)
    coldest = min(temperatures)
    outward_margin = request.options.default_isothermal_span.value + separation * max(
        level_count - 1, 0
    )
    hot_support = support(hot_profile)
    cold_support = support(cold_profile)
    hot_lower = min(hot_support)
    hot_upper = max(hot_support) + outward_margin
    # Keep cold supply bounds 0.01 K above absolute zero.
    cold_lower = max(-(C_to_K - 0.01), min(cold_support) - outward_margin)
    cold_upper = max(cold_support)
    paired_lower = min(hot_lower, cold_lower)
    paired_upper = max(hot_upper, cold_upper)
    maximum_span = max(
        request.options.minimum_sensible_span.value,
        hottest - coldest + outward_margin,
    )
    bounds: list[PhysicalCoordinateBound] = []
    for blueprint in blueprints.all:
        if request.uses_generated_pairs:
            supply = QuantityInterval(
                lower=paired_lower,
                upper=paired_upper,
                unit=request.units.absolute_temperature,
            )
        elif blueprint.key.side is UtilitySide.HOT:
            supply = QuantityInterval(
                lower=hot_lower,
                upper=hot_upper,
                unit=request.units.absolute_temperature,
            )
        else:
            supply = QuantityInterval(
                lower=cold_lower,
                upper=cold_upper,
                unit=request.units.absolute_temperature,
            )
        bounds.append(
            PhysicalCoordinateBound(
                coordinate=CoordinateKey(
                    template_key=blueprint.key,
                    field=DecisionField.SUPPLY_TEMPERATURE,
                ),
                bounds=supply,
                reason="residual-profile temperature support",
            )
        )
        if blueprint.kind is UtilityLevelKind.SENSIBLE:
            bounds.append(
                PhysicalCoordinateBound(
                    coordinate=CoordinateKey(
                        template_key=blueprint.key,
                        field=DecisionField.TEMPERATURE_SPAN,
                    ),
                    bounds=QuantityInterval(
                        lower=request.options.minimum_sensible_span.value,
                        upper=maximum_span,
                        unit=request.units.temperature_difference,
                    ),
                    reason="residual-profile sensible-span support",
                )
            )
    return tuple(bounds)
