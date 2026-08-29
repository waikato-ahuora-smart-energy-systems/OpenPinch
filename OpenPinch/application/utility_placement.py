"""Application-owned extraction for detached utility-placement analysis."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ..analysis.targeting.grand_composite import get_seperated_gcc_heat_load_profiles
from ..analysis.utility_placement.context import (
    PlacementPeriodInput,
    PlacementTargetSnapshot,
    ProcessEntropySlice,
    UtilityPlacementContext,
    build_utility_placement_context,
)
from ..analysis.utility_placement.errors import (
    PlacementContextError,
    PlacementRequestValidationError,
)
from ..analysis.utility_placement.normalization import (
    normalize_utility_placement_request,
    prepare_template_blueprints,
)
from ..analysis.utility_placement.service import optimise_utility_placement
from ..contracts.units import standardise_input_value
from ..contracts.utility_placement import (
    CoordinateKey,
    DecisionField,
    PhysicalCoordinateBound,
    QuantityInterval,
    TemplateBlueprintSet,
    UtilityLevelKind,
    UtilityLevelTemplate,
    UtilityPlacementBaseTarget,
    UtilityPlacementOptions,
    UtilityPlacementRequest,
    UtilitySide,
)
from ..domain._value.resolution import get_scalar_value
from ..domain.enums import ProblemTableLabel, StreamType, ZoneType
from ..domain.zone import Zone

if TYPE_CHECKING:
    from .problem import PinchProblem


def _resolved_scope(request: UtilityPlacementRequest) -> UtilityPlacementBaseTarget:
    if request.base_target is UtilityPlacementBaseTarget.AUTO:
        return UtilityPlacementBaseTarget.DIRECT
    return request.base_target


def _walk_zones(zone: Zone):
    yield zone
    for subzone in zone.subzones.values():
        yield from _walk_zones(subzone)


def _resolve_placement_zone(problem: "PinchProblem", zone: str | Zone | None) -> Zone:
    """Resolve one owned hierarchy node without relying on ambient zone state."""
    root = problem._build_execution_master_zone()
    zones = tuple(_walk_zones(root))
    if zone is None:
        return root
    if isinstance(zone, Zone):
        if any(candidate is zone for candidate in zones):
            return zone
        raise PlacementContextError(
            code="foreign_zone",
            message="The selected Zone does not belong to this PinchProblem.",
            details=(("zone", zone.address),),
        )
    if not isinstance(zone, str) or not zone.strip():
        raise PlacementContextError(
            code="invalid_zone",
            message="zone must be a non-empty name, address, or Zone object.",
        )
    selector = zone.strip()
    address_matches = tuple(
        candidate for candidate in zones if candidate.address == selector
    )
    if len(address_matches) == 1:
        return address_matches[0]
    name_matches = tuple(candidate for candidate in zones if candidate.name == selector)
    if len(name_matches) == 1:
        return name_matches[0]
    if len(name_matches) > 1:
        addresses = ", ".join(candidate.address for candidate in name_matches)
        raise PlacementContextError(
            code="ambiguous_zone",
            message=f"Zone name '{selector}' is ambiguous; use one of: {addresses}.",
            details=(("zone", selector),),
        )
    raise PlacementContextError(
        code="unknown_zone",
        message=f"Zone '{selector}' was not found in this PinchProblem.",
        details=(("zone", selector),),
    )


def _scope_for_zone(zone: Zone) -> UtilityPlacementBaseTarget:
    """Map a hierarchy node to its physically compatible target profile."""
    if zone.type in {ZoneType.P.value, ZoneType.O.value}:
        return UtilityPlacementBaseTarget.DIRECT
    if zone.type == ZoneType.S.value:
        return UtilityPlacementBaseTarget.TOTAL_SITE
    if zone.type in {ZoneType.C.value, ZoneType.R.value}:
        return UtilityPlacementBaseTarget.INDIRECT
    if zone.type == ZoneType.U.value:
        raise PlacementContextError(
            code="unsupported_zone_type",
            message="Utility Placement does not apply to a Utility Zone.",
            details=(("zone", zone.address), ("zone_type", zone.type)),
        )
    raise PlacementContextError(
        code="unsupported_zone_type",
        message=f"Utility Placement does not support zone type '{zone.type}'.",
        details=(("zone", zone.address), ("zone_type", zone.type)),
    )


def _missing_scalar(value) -> bool:
    if value is None:
        return True
    if hasattr(value, "value"):
        return value.value is None
    return False


def _utility_kind(
    utility,
    *,
    options: UtilityPlacementOptions,
    config,
) -> UtilityLevelKind:
    if utility.segments is not None or utility.profile is not None:
        return UtilityLevelKind.SENSIBLE
    if _missing_scalar(utility.t_target):
        return UtilityLevelKind.ISOTHERMAL
    supply = standardise_input_value(
        utility.t_supply,
        field_name="t_supply",
        config=config,
    )
    target = standardise_input_value(
        utility.t_target,
        field_name="t_target",
        config=config,
    )
    if supply is None or target is None:
        raise PlacementRequestValidationError(
            code="incomplete_utility_temperature",
            message=f"Existing utility '{utility.name}' has incomplete temperatures.",
            field_path=f"utilities.{utility.name}",
        )
    if len(supply.period_values) != len(target.period_values):
        raise PlacementRequestValidationError(
            code="utility_period_mismatch",
            message=(
                f"Existing utility '{utility.name}' temperatures do not align "
                "by period."
            ),
            field_path=f"utilities.{utility.name}",
        )
    maximum_span = max(
        abs(float(supply_value) - float(target_value))
        for supply_value, target_value in zip(
            supply.period_values,
            target.period_values,
            strict=True,
        )
    )
    return (
        UtilityLevelKind.ISOTHERMAL
        if maximum_span <= options.default_isothermal_span.value
        else UtilityLevelKind.SENSIBLE
    )


def _template_for_existing_utility(
    utility,
    *,
    side: UtilitySide,
    kind: UtilityLevelKind,
    paired: bool,
    options: UtilityPlacementOptions,
) -> UtilityLevelTemplate:
    name = f"{utility.name} ({side.value})" if paired else utility.name
    return UtilityLevelTemplate(
        name=name,
        side=side,
        kind=kind,
        fixed_span=(
            options.default_isothermal_span
            if kind is UtilityLevelKind.ISOTHERMAL
            else None
        ),
        fluid=utility.fluid_name,
    )


def _infer_problem_utility_templates(
    problem: "PinchProblem",
    *,
    selected_zone: Zone,
    options: UtilityPlacementOptions,
) -> tuple[
    int,
    int,
    tuple[UtilityLevelTemplate, ...],
    tuple[UtilityLevelTemplate, ...],
]:
    utilities = tuple(
        utility for utility in problem.validate().utilities if utility.active
    )
    if not utilities:
        raise PlacementRequestValidationError(
            code="missing_existing_utilities",
            message=(
                "Omitted counts require existing utilities on the PinchProblem; "
                "supply isothermal and optional sensible counts instead."
            ),
            field_path="utilities",
        )
    by_side: dict[UtilitySide, list[UtilityLevelTemplate]] = {
        UtilitySide.HOT: [],
        UtilitySide.COLD: [],
    }
    for utility in sorted(utilities, key=lambda item: item.name.casefold()):
        kind = _utility_kind(utility, options=options, config=selected_zone.config)
        sides = {
            StreamType.Hot.value: (UtilitySide.HOT,),
            StreamType.Cold.value: (UtilitySide.COLD,),
            StreamType.Both.value: (UtilitySide.HOT, UtilitySide.COLD),
        }.get(utility.type)
        if sides is None:
            raise PlacementRequestValidationError(
                code="invalid_utility_side",
                message=(
                    f"Existing utility '{utility.name}' has unsupported type "
                    f"'{utility.type}'."
                ),
                field_path=f"utilities.{utility.name}.type",
            )
        for side in sides:
            by_side[side].append(
                _template_for_existing_utility(
                    utility,
                    side=side,
                    kind=kind,
                    paired=len(sides) == 2,
                    options=options,
                )
            )

    counts = {
        kind: max(
            sum(template.kind is kind for template in by_side[UtilitySide.HOT]),
            sum(template.kind is kind for template in by_side[UtilitySide.COLD]),
        )
        for kind in UtilityLevelKind
    }
    if counts[UtilityLevelKind.ISOTHERMAL] < 2:
        raise PlacementRequestValidationError(
            code="insufficient_inferred_isothermal_levels",
            message=(
                "Existing utilities must infer at least 2 isothermal levels per side."
            ),
            field_path="utilities",
        )
    for side, templates in by_side.items():
        for kind in UtilityLevelKind:
            observed = sum(template.kind is kind for template in templates)
            for ordinal in range(observed + 1, counts[kind] + 1):
                kind_label = (
                    "iso" if kind is UtilityLevelKind.ISOTHERMAL else "sensible"
                )
                templates.append(
                    UtilityLevelTemplate(
                        name=f"inferred_{side.value}_{kind_label}_{ordinal}",
                        side=side,
                        kind=kind,
                        fixed_span=(
                            options.default_isothermal_span
                            if kind is UtilityLevelKind.ISOTHERMAL
                            else None
                        ),
                    )
                )
        templates.sort(key=lambda item: (item.kind.value, item.name.casefold()))
    return (
        counts[UtilityLevelKind.ISOTHERMAL],
        counts[UtilityLevelKind.SENSIBLE],
        tuple(by_side[UtilitySide.HOT]),
        tuple(by_side[UtilitySide.COLD]),
    )


def _build_problem_placement_request(
    problem: "PinchProblem",
    *,
    selected_zone: Zone,
    isothermal: int | None,
    sensible: int | None,
    period_ids,
    options,
) -> UtilityPlacementRequest:
    placement_options = (
        options
        if isinstance(options, UtilityPlacementOptions)
        else UtilityPlacementOptions.model_validate(options or {})
    )
    scope = _scope_for_zone(selected_zone)
    generated = isothermal is not None or sensible is not None
    if generated:
        if isothermal is None:
            raise PlacementRequestValidationError(
                code="missing_isothermal_count",
                message=(
                    "isothermal is required when generated level counts are supplied."
                ),
                field_path="isothermal",
            )
        inferred = (isothermal, 0 if sensible is None else sensible, None, None)
    else:
        inferred = _infer_problem_utility_templates(
            problem,
            selected_zone=selected_zone,
            options=placement_options,
        )
    iso_count, sensible_count, hot_templates, cold_templates = inferred
    return normalize_utility_placement_request(
        isothermal_level_count=iso_count,
        sensible_level_count=sensible_count,
        hot_templates=hot_templates,
        cold_templates=cold_templates,
        base_target=scope,
        zone=selected_zone.address,
        period_ids=None if period_ids is None else tuple(period_ids),
        options=placement_options,
    )


def _period_selection(
    problem: "PinchProblem", request: UtilityPlacementRequest
) -> tuple[str, ...]:
    available = tuple(problem.period_ids)
    selected = request.period_ids or available
    missing = tuple(
        period_id for period_id in selected if period_id not in problem.period_ids
    )
    if missing:
        raise PlacementContextError(
            code="unknown_period",
            message="Requested utility-placement period is not available.",
            period_id=missing[0],
        )
    return selected


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
                temperature_in = float(supply) + 273.15
                temperature_out = float(target) + 273.15
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
    cold_lower = max(-273.14, min(cold_support) - outward_margin)
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


def _period_weight(problem: "PinchProblem", period_id: str) -> float:
    index = problem.period_ids[period_id]
    weights = problem._master_zone.weights
    return float(weights[index]) if weights is not None else 1.0


def _extract_period(
    problem: "PinchProblem",
    request: UtilityPlacementRequest,
    blueprints: TemplateBlueprintSet,
    scope: UtilityPlacementBaseTarget,
    period_id: str,
) -> PlacementPeriodInput:
    isolated = type(problem)(
        source=problem.to_problem_json(),
        project_name=problem.project_name,
    )
    kwargs = {"period_id": period_id}
    if request.zone is not None:
        kwargs["zone"] = request.zone
    if scope is UtilityPlacementBaseTarget.DIRECT:
        target = isolated.target.direct_heat_integration(**kwargs)
    elif scope is UtilityPlacementBaseTarget.TOTAL_SITE:
        target = isolated.target.total_site_heat_integration(**kwargs)
    else:
        target = isolated.target.indirect_heat_integration(**kwargs)
    analysis_zone = isolated.master_zone.get_subzone(request.zone)
    if analysis_zone is None:
        raise PlacementContextError(
            code="unknown_zone",
            message="Requested utility-placement zone is not available.",
            details=(("zone", request.zone),),
        )
    shifted = target.pt
    real = getattr(target, "pt_real", None) or shifted
    shifted_temperatures = _finite_tuple(shifted[ProblemTableLabel.T])
    real_temperatures = _finite_tuple(real[ProblemTableLabel.T])
    pinch_label = ProblemTableLabel.H_NET_A
    if not all(
        math.isfinite(float(value)) for value in shifted[ProblemTableLabel.H_NET_A]
    ):
        pinch_label = ProblemTableLabel.H_NET
    hot_pinch, cold_pinch, *_ = shifted.pinch_idx(pinch_label)
    residual_hot_duty = float(get_scalar_value(target.hot_utility_target))
    residual_cold_duty = float(get_scalar_value(target.cold_utility_target))
    hot_profile, cold_profile = _load_profiles(
        shifted,
        net_label=(
            ProblemTableLabel.H_NET_UT
            if scope is not UtilityPlacementBaseTarget.DIRECT
            else None
        ),
    )
    hot_profile = _calibrate_profile(
        hot_profile,
        residual_duty=residual_hot_duty,
    )
    cold_profile = _calibrate_profile(
        cold_profile,
        residual_duty=residual_cold_duty,
    )
    return PlacementPeriodInput(
        period_id=period_id,
        weight=_period_weight(problem, period_id),
        snapshot=PlacementTargetSnapshot(
            shifted_temperatures=shifted_temperatures,
            real_temperatures=real_temperatures,
            hot_load_profile=hot_profile,
            cold_load_profile=cold_profile,
            real_hot_composite=_finite_tuple(real[ProblemTableLabel.H_HOT]),
            real_cold_composite=_finite_tuple(real[ProblemTableLabel.H_COLD]),
            hot_pinch_index=int(hot_pinch),
            cold_pinch_index=int(cold_pinch),
            entropy_slices=_process_entropy_slices(
                analysis_zone,
                isolated.period_ids[period_id],
            ),
        ),
        residual_hot_duty=residual_hot_duty,
        residual_cold_duty=residual_cold_duty,
        ambient_temperature_kelvin=298.15,
        coordinate_bounds=_coordinate_bounds(
            request,
            blueprints,
            shifted_temperatures,
            hot_profile=hot_profile,
            cold_profile=cold_profile,
        ),
    )


def build_problem_placement_context(
    problem: "PinchProblem",
    request: UtilityPlacementRequest,
) -> tuple[TemplateBlueprintSet, UtilityPlacementContext]:
    """Build a detached context by targeting only isolated problem copies."""
    scope = _resolved_scope(request)
    period_ids = _period_selection(problem, request)
    normalized = request.model_copy(
        update={"base_target": scope, "period_ids": period_ids}
    )
    blueprints = prepare_template_blueprints(normalized)
    periods = tuple(
        _extract_period(problem, normalized, blueprints, scope, period_id)
        for period_id in period_ids
    )
    zone_name = normalized.zone or problem.project_name or "Site"
    context = build_utility_placement_context(
        request=normalized,
        blueprints=blueprints,
        scope=scope,
        base_target_id=f"{scope.value}:{zone_name}",
        periods=periods,
    )
    return blueprints, context


def run_problem_utility_placement(
    problem: "PinchProblem",
    *,
    isothermal: int | None = None,
    sensible: int | None = None,
    zone=None,
    period_ids=None,
    options=None,
) -> "PinchProblem":
    """Optimize one placement and return its best utilities as a normal case."""
    selected_zone = _resolve_placement_zone(problem, zone)
    request = _build_problem_placement_request(
        problem,
        selected_zone=selected_zone,
        isothermal=isothermal,
        sensible=sensible,
        period_ids=period_ids,
        options=options,
    )
    blueprints, context = build_problem_placement_context(problem, request)
    resolved_request = request.model_copy(
        update={
            "base_target": context.scope,
            "period_ids": tuple(period.period_id for period in context.periods),
        }
    )
    result = optimise_utility_placement(
        request=resolved_request,
        blueprints=blueprints,
        context=context,
    )
    period = result.best.period_results[0]
    optimized_input = problem.to_problem_json()
    optimized_input["utilities"] = [
        {
            "name": level.template_key.name,
            "type": utility_type,
            "t_supply": level.supply_temperature.model_dump(),
            "t_target": level.target_temperature.model_dump(),
            "heat_flow": {"value": 0.0, "unit": result.units.heat_flow},
            "dt_cont": {"value": 0.0, "unit": result.units.temperature_difference},
            "htc": {"value": 1.0, "unit": "kW/m^2/delta_degC"},
        }
        for utility_type, levels in (
            ("Hot", period.hot_levels),
            ("Cold", period.cold_levels),
        )
        for level in levels
    ]
    from .problem import PinchProblem

    optimized_case = PinchProblem(optimized_input, project_name=problem.project_name)
    optimized_case._utility_placement_result = result
    return optimized_case


__all__ = ["build_problem_placement_context", "run_problem_utility_placement"]
