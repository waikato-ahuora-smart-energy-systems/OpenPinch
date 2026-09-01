"""Application-owned extraction for detached utility-placement analysis."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..analysis.targeting.cascade import get_process_heat_cascade
from ..analysis.targeting.direct import (
    _create_net_hot_and_cold_stream_collections_for_site_analysis,
    _prepare_direct_integration_profile,
    _prepare_utility_load_profile,
    _PreparedDirectIntegrationProfile,
    _PreparedUtilityLoadProfile,
    _target_prepared_utility_load_profile_duties,
)
from ..analysis.targeting.grand_composite import get_seperated_gcc_heat_load_profiles
from ..analysis.targeting.indirect import (
    _build_site_utility_profile,
    _match_utility_gen_and_use_at_same_level,
    _shift_site_process_profiles,
)
from ..analysis.utility_placement.allocation import (
    AllocationAdapterResult,
    _build_stream,
    _fallback_stream,
)
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
from ..contracts.input import UtilitySchema
from ..contracts.units import standardise_input_value
from ..contracts.utility_placement import (
    CandidateDiagnostic,
    CoordinateKey,
    DecisionField,
    DecodedPlacement,
    PhysicalCoordinateBound,
    QuantityInterval,
    QuantityValue,
    TemplateBlueprintSet,
    UtilityDutyLimit,
    UtilityLevelKind,
    UtilityLevelTemplate,
    UtilityPlacementBaseTarget,
    UtilityPlacementOptions,
    UtilityPlacementRequest,
    UtilitySide,
)
from ..domain._value.resolution import get_scalar_value
from ..domain.enums import ProblemTableLabel, StreamType, TargetType, ZoneType
from ..domain.problem_table import ProblemTable
from ..domain.stream import Stream
from ..domain.stream_collection import StreamCollection
from ..domain.zone import Zone
from ._problem.input.construction import _find_extreme_process_temperatures
from ._problem.input.utilities import (
    _get_hot_and_cold_utilities,
)
from .targeting import (
    direct_heat_integration_service,
    indirect_heat_integration_service,
)

if TYPE_CHECKING:
    from .problem import PinchProblem

_FALLBACK_TEMPERATURE_MARGIN = 50.0


@dataclass(frozen=True)
class _PreparedAggregateComposite:
    """Invariant Total Site process composite for one selected period."""

    temperatures: tuple[float, ...]
    hot_composite: tuple[float, ...]
    cold_composite: tuple[float, ...]


def _clone_zone_tree_for_target_replay(
    source: Zone,
    *,
    parent: Zone | None = None,
) -> Zone:
    """Clone mutable target state while sharing read-only process collections."""
    cloned = copy(source)
    cloned._parent_zone = parent
    cloned._subzones = {
        name: _clone_zone_tree_for_target_replay(child, parent=cloned)
        for name, child in source.subzones.items()
    }
    cloned._targets = {}
    cloned._graphs = {}
    cloned._net_hot_streams = cloned._new_stream_collection()
    cloned._net_cold_streams = cloned._new_stream_collection()
    cloned._subzone_net_hot_streams = cloned._new_stream_collection()
    cloned._subzone_net_cold_streams = cloned._new_stream_collection()
    cloned._hot_utilities = cloned._new_stream_collection()
    cloned._cold_utilities = cloned._new_stream_collection()
    return cloned


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
        utility
        for utility in problem.validate().utilities
        if utility.active and utility.name.strip().casefold() not in {"hu", "cu"}
    )
    if not utilities:
        raise PlacementRequestValidationError(
            code="missing_existing_utilities",
            message=(
                "Omitted counts require existing utilities other than HU/CU "
                "defaults on the PinchProblem; supply isothermal and optional "
                "sensible counts instead."
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
    maximum_duties=None,
    options=None,
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
    request = normalize_utility_placement_request(
        isothermal_level_count=iso_count,
        sensible_level_count=sensible_count,
        hot_templates=hot_templates,
        cold_templates=cold_templates,
        base_target=scope,
        zone=selected_zone.address,
        period_ids=(
            tuple(problem.period_ids) if period_ids is None else tuple(period_ids)
        ),
        options=placement_options,
    )
    selected_period_ids = _period_selection(problem, request)
    blueprints = prepare_template_blueprints(request)
    known_names = {blueprint.key.name for blueprint in blueprints.all}
    limits = _normalize_maximum_duties(
        maximum_duties,
        known_names=known_names,
        selected_period_ids=selected_period_ids,
        available_period_ids=tuple(problem.period_ids),
        config=selected_zone.config,
        heat_flow_unit=request.units.heat_flow,
    )
    return request.model_copy(update={"maximum_duties": limits})


def _normalize_maximum_duties(
    maximum_duties,
    *,
    known_names: set[str],
    selected_period_ids: tuple[str, ...],
    available_period_ids: tuple[str, ...],
    config,
    heat_flow_unit: str,
) -> tuple[UtilityDutyLimit, ...]:
    """Normalize one public name/value mapping after template resolution."""
    if maximum_duties is None:
        return ()
    if not isinstance(maximum_duties, Mapping):
        raise PlacementRequestValidationError(
            code="invalid_maximum_duties",
            message="maximum_duties must map utility names to duty limits.",
            field_path="maximum_duties",
        )

    normalized: list[UtilityDutyLimit] = []
    observed: set[str] = set()
    for raw_name, raw_value in maximum_duties.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise PlacementRequestValidationError(
                code="invalid_maximum_duty_name",
                message="Maximum-duty utility names must be non-empty strings.",
                field_path="maximum_duties",
            )
        name = raw_name.strip()
        if name in observed:
            raise PlacementRequestValidationError(
                code="duplicate_maximum_duty",
                message="Maximum-duty utility names must be unique.",
                field_path=f"maximum_duties.{name}",
            )
        observed.add(name)
        if name not in known_names:
            raise PlacementRequestValidationError(
                code="unknown_maximum_duty_utility",
                message=f"maximum_duties contains unknown utility '{name}'.",
                field_path=f"maximum_duties.{name}",
            )
        try:
            value = standardise_input_value(
                raw_value,
                field_name="heat_flow",
                config=config,
            )
        except Exception as exc:
            raise PlacementRequestValidationError(
                code="invalid_maximum_duty_unit",
                message=f"Maximum duty for '{name}' has an incompatible unit.",
                field_path=f"maximum_duties.{name}",
            ) from exc
        if value is None:
            raise PlacementRequestValidationError(
                code="missing_maximum_duty",
                message=f"Maximum duty for '{name}' requires a value.",
                field_path=f"maximum_duties.{name}",
            )

        magnitudes = tuple(float(item) for item in value.period_values)
        explicit_period_ids = None
        if isinstance(raw_value, Mapping) and raw_value.get("period_ids") is not None:
            explicit_period_ids = tuple(str(item) for item in raw_value["period_ids"])
            if len(explicit_period_ids) != len(magnitudes) or len(
                set(explicit_period_ids)
            ) != len(explicit_period_ids):
                raise PlacementRequestValidationError(
                    code="invalid_maximum_duty_periods",
                    message=(
                        f"Maximum-duty periods for '{name}' must align and be unique."
                    ),
                    field_path=f"maximum_duties.{name}.period_ids",
                )
            by_period = dict(zip(explicit_period_ids, magnitudes, strict=True))
            if any(period_id not in by_period for period_id in selected_period_ids):
                raise PlacementRequestValidationError(
                    code="missing_maximum_duty_period",
                    message=f"Maximum duty for '{name}' is missing a selected period.",
                    field_path=f"maximum_duties.{name}",
                )
            selected_values = tuple(
                by_period[period_id] for period_id in selected_period_ids
            )
        elif len(magnitudes) == 1:
            selected_values = magnitudes * len(selected_period_ids)
        elif len(magnitudes) == len(available_period_ids):
            by_period = dict(zip(available_period_ids, magnitudes, strict=True))
            selected_values = tuple(
                by_period[period_id] for period_id in selected_period_ids
            )
        elif len(magnitudes) == len(selected_period_ids):
            selected_values = magnitudes
        else:
            raise PlacementRequestValidationError(
                code="maximum_duty_period_mismatch",
                message=(
                    f"Maximum duty for '{name}' does not align with selected periods."
                ),
                field_path=f"maximum_duties.{name}",
            )
        if any(not math.isfinite(item) or item < 0.0 for item in selected_values):
            raise PlacementRequestValidationError(
                code="invalid_maximum_duty",
                message=f"Maximum duty for '{name}' must be finite and non-negative.",
                field_path=f"maximum_duties.{name}",
            )
        normalized.append(
            UtilityDutyLimit(
                name=name,
                period_ids=selected_period_ids,
                values=tuple(
                    QuantityValue(value=item, unit=heat_flow_unit)
                    for item in selected_values
                ),
            )
        )
    return tuple(sorted(normalized, key=lambda item: item.name.casefold()))


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


def _snapshot_from_target(
    isolated: "PinchProblem",
    request: UtilityPlacementRequest,
    scope: UtilityPlacementBaseTarget,
    period_id: str,
    target,
) -> tuple[PlacementTargetSnapshot, float, float]:
    """Extract the exact target arrays and candidate-local utility totals."""
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
    if scope is not UtilityPlacementBaseTarget.DIRECT:
        pinch_label = ProblemTableLabel.H_NET_UT
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
    snapshot = PlacementTargetSnapshot(
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
    )
    return snapshot, residual_hot_duty, residual_cold_duty


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
    snapshot, residual_hot_duty, residual_cold_duty = _snapshot_from_target(
        isolated,
        request,
        scope,
        period_id,
        target,
    )
    return PlacementPeriodInput(
        period_id=period_id,
        weight=_period_weight(problem, period_id),
        snapshot=snapshot,
        residual_hot_duty=residual_hot_duty,
        residual_cold_duty=residual_cold_duty,
        ambient_temperature_kelvin=298.15,
        maximum_duties=tuple(
            (limit.name, limit.for_period(period_id).value)
            for limit in request.maximum_duties
        ),
        fallback_temperature_span=request.options.default_isothermal_span.value,
        coordinate_bounds=_coordinate_bounds(
            request,
            blueprints,
            snapshot.shifted_temperatures,
            hot_profile=snapshot.hot_load_profile,
            cold_profile=snapshot.cold_load_profile,
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
    fallback_span = normalized.options.default_isothermal_span.value
    fallback_hot_target = (
        max(max(period.snapshot.real_temperatures) for period in periods)
        + _FALLBACK_TEMPERATURE_MARGIN
        - fallback_span
    )
    fallback_cold_target = (
        min(min(period.snapshot.real_temperatures) for period in periods)
        - _FALLBACK_TEMPERATURE_MARGIN
        + fallback_span
    )
    periods = tuple(
        period.model_copy(
            update={
                "fallback_hot_target_temperature": fallback_hot_target,
                "fallback_cold_target_temperature": fallback_cold_target,
            }
        )
        for period in periods
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


def _serialized_limit(
    request: UtilityPlacementRequest,
    name: str,
    *,
    period_id: str | None = None,
):
    limit = next((item for item in request.maximum_duties if item.name == name), None)
    if limit is None:
        return None
    if period_id is not None:
        selected = limit.for_period(period_id)
        return {"value": selected.value, "unit": selected.unit}
    values = tuple(value.value for value in limit.values)
    return {
        "values": list(values),
        "period_ids": list(limit.period_ids),
        "unit": request.units.heat_flow,
    }


def _candidate_utility_input(
    request: UtilityPlacementRequest,
    placement: DecodedPlacement,
    period: PlacementPeriodInput,
) -> list[dict[str, object]]:
    utilities: list[dict[str, object]] = []
    for utility_type, levels in (("Hot", placement.hot), ("Cold", placement.cold)):
        for level in levels:
            limit = _serialized_limit(
                request,
                level.template_key.name,
                period_id=period.period_id,
            )
            utilities.append(
                {
                    "name": level.template_key.name,
                    "type": utility_type,
                    "t_supply": level.supply_temperature.model_dump(),
                    "t_target": level.target_temperature.model_dump(),
                    "heat_flow": {"value": 0.0, "unit": request.units.heat_flow},
                    **({"maximum_heat_flow": limit} if limit is not None else {}),
                    "dt_cont": {
                        "value": 0.0,
                        "unit": request.units.temperature_difference,
                    },
                    "htc": {"value": 1.0, "unit": "kW/m^2/delta_degC"},
                }
            )
    fallback_targets = (
        (
            "HU",
            "Hot",
            period.fallback_hot_target_temperature,
            1.0,
        ),
        (
            "CU",
            "Cold",
            period.fallback_cold_target_temperature,
            -1.0,
        ),
    )
    for name, utility_type, target, direction in fallback_targets:
        if target is None:
            raise PlacementContextError(
                code="missing_fallback_temperature",
                message="Exact targeting requires context-wide fallback temperatures.",
                period_id=period.period_id,
            )
        supply = target + direction * period.fallback_temperature_span
        utilities.append(
            {
                "name": name,
                "type": utility_type,
                "t_supply": {
                    "value": supply,
                    "unit": request.units.absolute_temperature,
                },
                "t_target": {
                    "value": target,
                    "unit": request.units.absolute_temperature,
                },
                "heat_flow": {"value": 0.0, "unit": request.units.heat_flow},
                "dt_cont": {
                    "value": 0.0,
                    "unit": request.units.temperature_difference,
                },
                "htc": {"value": 1.0, "unit": "kW/m^2/delta_degC"},
            }
        )
    return utilities


class _ExactTargetReplayAdapter:
    """Replay Direct candidates exactly and cache aggregate process profiles."""

    def __init__(
        self,
        *,
        source: dict,
        project_name: str | None,
        request: UtilityPlacementRequest,
    ) -> None:
        self.source = source
        self.project_name = project_name
        self.request = request
        from .problem import PinchProblem

        self._prepared_problem = PinchProblem(
            source=self.source,
            project_name=self.project_name,
        )
        self._target_zone_addresses = tuple(
            zone.address for zone in self._target_zones()
        )
        self._prepared_profiles = self._prepare_direct_profiles()
        self._prepared_aggregate_profiles = self._prepare_aggregate_profiles()
        self._prepared_aggregate_composites = self._prepare_aggregate_composites()
        master = self._prepared_problem._master_zone
        self._utility_temperature_extremes = _find_extreme_process_temperatures(
            hot_streams=master.hot_streams,
            cold_streams=master.cold_streams,
        )

    def _target_zones(self) -> tuple[Zone, ...]:
        selected = _resolve_placement_zone(
            self._prepared_problem,
            self.request.zone,
        )
        if self.request.base_target is UtilityPlacementBaseTarget.DIRECT:
            return (selected,)
        return (selected, *selected.subzones.values())

    def _direct_profile_addresses(self) -> tuple[str, ...]:
        if self.request.base_target is UtilityPlacementBaseTarget.DIRECT:
            return self._target_zone_addresses
        child_addresses = self._target_zone_addresses[1:]
        return child_addresses or self._target_zone_addresses

    def _prepare_direct_profiles(
        self,
    ) -> dict[tuple[str, str], _PreparedDirectIntegrationProfile]:
        profiles = {}
        for period_id in self.request.period_ids:
            args = {"period_id": period_id}
            for address in self._direct_profile_addresses():
                zone = self._prepared_problem._master_zone.get_subzone(address)
                profiles[(period_id, zone.address)] = (
                    _prepare_direct_integration_profile(zone, args)
                )
        return profiles

    def _aggregate_profile_addresses(self) -> tuple[str, ...]:
        if self.request.base_target is UtilityPlacementBaseTarget.DIRECT:
            return ()
        selected_address = self._target_zone_addresses[0]
        child_addresses = self._target_zone_addresses[1:]
        return child_addresses or (selected_address,)

    def _prepare_aggregate_profiles(
        self,
    ) -> dict[tuple[str, str], _PreparedUtilityLoadProfile]:
        profiles = {}
        for period_id in self.request.period_ids:
            for address in self._aggregate_profile_addresses():
                zone = self._prepared_problem._master_zone.get_subzone(address)
                profiles[(period_id, address)] = _prepare_utility_load_profile(
                    zone,
                    self._prepared_profiles[(period_id, address)],
                )
        return profiles

    def _prepare_aggregate_composites(
        self,
    ) -> dict[str, _PreparedAggregateComposite]:
        if self.request.base_target is UtilityPlacementBaseTarget.DIRECT:
            return {}
        composites = {}
        for period_id in self.request.period_ids:
            net_hot_streams = StreamCollection()
            net_cold_streams = StreamCollection()
            for address in self._aggregate_profile_addresses():
                profile = self._prepared_aggregate_profiles[(period_id, address)]
                temperatures = profile.pt[ProblemTableLabel.T]
                maximum_temperature = float(np.max(temperatures))
                minimum_temperature = float(np.min(temperatures))
                hot_utility = StreamCollection(
                    [
                        Stream(
                            name="Cached hot coverage",
                            supply_temperature=maximum_temperature + 1.0,
                            target_temperature=maximum_temperature,
                            heat_flow=profile.hot_utility_target,
                            delta_t_contribution=0.0,
                            is_process_stream=False,
                        )
                    ]
                )
                cold_utility = StreamCollection(
                    [
                        Stream(
                            name="Cached cold coverage",
                            supply_temperature=minimum_temperature - 1.0,
                            target_temperature=minimum_temperature,
                            heat_flow=profile.cold_utility_target,
                            delta_t_contribution=0.0,
                            is_process_stream=False,
                        )
                    ]
                )
                child_hot, child_cold = (
                    _create_net_hot_and_cold_stream_collections_for_site_analysis(
                        T_vals=temperatures,
                        H_vals=profile.pt[ProblemTableLabel.H_NET_A],
                        hot_utilities=hot_utility,
                        cold_utilities=cold_utility,
                        idx=None,
                    )
                )
                for key, stream in child_hot.items():
                    net_hot_streams.add(
                        stream,
                        key=f"{address}.{key}",
                    )
                for key, stream in child_cold.items():
                    net_cold_streams.add(
                        stream,
                        key=f"{address}.{key}",
                    )

            pt = get_process_heat_cascade(
                hot_streams=net_hot_streams,
                cold_streams=net_cold_streams,
                is_shifted=True,
            )
            pt.update(
                **_shift_site_process_profiles(
                    T_col=pt[ProblemTableLabel.T],
                    H_hot=pt[ProblemTableLabel.H_HOT],
                    H_cold=pt[ProblemTableLabel.H_COLD],
                )
            )
            composites[period_id] = _PreparedAggregateComposite(
                temperatures=_finite_tuple(pt[ProblemTableLabel.T]),
                hot_composite=_finite_tuple(pt[ProblemTableLabel.H_HOT]),
                cold_composite=_finite_tuple(pt[ProblemTableLabel.H_COLD]),
            )
        return composites

    def _candidate_utility_collections(
        self,
        utilities: list[dict[str, object]],
    ):
        master = self._prepared_problem._master_zone
        utility_schemas = [UtilitySchema.model_validate(item) for item in utilities]
        hu_t_min, cu_t_max = self._utility_temperature_extremes
        prepared = _get_hot_and_cold_utilities(
            utilities=utility_schemas,
            hu_t_min=hu_t_min,
            cu_t_max=cu_t_max,
            config=master.config,
            dt_cont_multiplier=master.dt_cont_multiplier,
        )
        return (
            prepared.get_hot_utility_streams(),
            prepared.get_cold_utility_streams(),
        )

    @staticmethod
    def _placement_utility_collections(
        period: PlacementPeriodInput,
        placement: DecodedPlacement,
    ) -> tuple[StreamCollection, StreamCollection]:
        limits = dict(period.maximum_duties)
        hot = StreamCollection(
            [
                *(
                    _build_stream(
                        level,
                        maximum_duty=limits.get(level.template_key.name),
                    )
                    for level in placement.hot
                ),
                _fallback_stream(period, UtilitySide.HOT),
            ]
        )
        cold = StreamCollection(
            [
                *(
                    _build_stream(
                        level,
                        maximum_duty=limits.get(level.template_key.name),
                    )
                    for level in placement.cold
                ),
                _fallback_stream(period, UtilitySide.COLD),
            ]
        )
        return hot, cold

    def _candidate_problem(
        self,
        utilities: list[dict[str, object]],
    ) -> "PinchProblem":
        candidate = copy(self._prepared_problem)
        candidate._master_zone = _clone_zone_tree_for_target_replay(
            self._prepared_problem._master_zone
        )
        candidate._results = None
        candidate._last_target_run_spec = None
        candidate._period_results = {}
        candidate._utility_placement_result = None
        candidate._process_components = {}

        master = candidate._master_zone
        hot_utilities, cold_utilities = self._candidate_utility_collections(utilities)
        for address in self._target_zone_addresses:
            zone = master.get_subzone(address)
            zone.hot_utilities = hot_utilities.copy(deep=True)
            zone.cold_utilities = cold_utilities.copy(deep=True)
        return candidate

    def _target_candidate(
        self,
        isolated: "PinchProblem",
        period_id: str,
    ):
        prepared_profiles = {
            address: profile
            for (profile_period, address), profile in self._prepared_profiles.items()
            if profile_period == period_id
        }
        options = {
            "period_id": period_id,
            "_prepared_direct_profiles": prepared_profiles,
        }
        if self.request.base_target is UtilityPlacementBaseTarget.DIRECT:
            return isolated._execute_targeting(
                target_id=TargetType.DI.value,
                application_zone=self.request.zone,
                options=options,
                include_subzones=False,
                direct_service_func=direct_heat_integration_service,
            )
        return isolated._execute_targeting(
            target_id=TargetType.II.value,
            application_zone=self.request.zone,
            options=options,
            include_subzones=False,
            indirect_service_func=indirect_heat_integration_service,
        )

    def _allocation_result(
        self,
        *,
        hot_utilities,
        cold_utilities,
        period_idx: int | None,
        placement: DecodedPlacement,
        required_hot: float,
        required_cold: float,
        snapshot: PlacementTargetSnapshot,
    ) -> AllocationAdapterResult:
        def duty(utility) -> float:
            return float(get_scalar_value(utility.heat_flow, period_idx=period_idx))

        def temperature(utility, attribute: str) -> float:
            return float(
                get_scalar_value(
                    getattr(utility, attribute),
                    period_idx=period_idx,
                )
            )

        hot_by_name = {utility.name: utility for utility in hot_utilities}
        cold_by_name = {utility.name: utility for utility in cold_utilities}
        hot_names = {level.template_key.name for level in placement.hot}
        cold_names = {level.template_key.name for level in placement.cold}
        hot_fallbacks = tuple(
            utility
            for utility in hot_utilities
            if utility.name not in hot_names and duty(utility) > 0.0
        )
        cold_fallbacks = tuple(
            utility
            for utility in cold_utilities
            if utility.name not in cold_names and duty(utility) > 0.0
        )

        def fallback_values(utilities, default_name: str):
            if not utilities:
                return default_name, 0.0, None, None
            first = utilities[0]
            return (
                first.name,
                math.fsum(duty(utility) for utility in utilities),
                temperature(first, "supply_temperature"),
                temperature(first, "target_temperature"),
            )

        hot_fallback = fallback_values(hot_fallbacks, "HU")
        cold_fallback = fallback_values(cold_fallbacks, "CU")
        return AllocationAdapterResult(
            hot_duties=tuple(
                duty(hot_by_name[level.template_key.name])
                if level.template_key.name in hot_by_name
                else 0.0
                for level in placement.hot
            ),
            cold_duties=tuple(
                duty(cold_by_name[level.template_key.name])
                if level.template_key.name in cold_by_name
                else 0.0
                for level in placement.cold
            ),
            hot_fallback_name=hot_fallback[0],
            hot_fallback_duty=hot_fallback[1],
            hot_fallback_supply_temperature=hot_fallback[2],
            hot_fallback_target_temperature=hot_fallback[3],
            cold_fallback_name=cold_fallback[0],
            cold_fallback_duty=cold_fallback[1],
            cold_fallback_supply_temperature=cold_fallback[2],
            cold_fallback_target_temperature=cold_fallback[3],
            required_hot_duty=required_hot,
            required_cold_duty=required_cold,
            target_snapshot=snapshot,
        )

    def _allocate_exact_target_replay(
        self,
        period: PlacementPeriodInput,
        placement: DecodedPlacement,
    ) -> AllocationAdapterResult:
        candidate_utilities = _candidate_utility_input(
            self.request,
            placement,
            period,
        )
        isolated = self._candidate_problem(candidate_utilities)
        target = self._target_candidate(isolated, period.period_id)

        snapshot, required_hot, required_cold = _snapshot_from_target(
            isolated,
            self.request,
            self.request.base_target,
            period.period_id,
            target,
        )
        period_idx = isolated.period_ids[period.period_id]
        return self._allocation_result(
            hot_utilities=target.hot_utilities,
            cold_utilities=target.cold_utilities,
            period_idx=period_idx,
            placement=placement,
            required_hot=required_hot,
            required_cold=required_cold,
            snapshot=snapshot,
        )

    def _allocate_cached_aggregate(
        self,
        period: PlacementPeriodInput,
        placement: DecodedPlacement,
    ) -> AllocationAdapterResult:
        hot_base, cold_base = self._placement_utility_collections(period, placement)
        period_idx = None
        aggregate_hot = hot_base.copy(deep=True).set_common_stream_attribute(
            "heat_flow",
            0.0,
            idx=period_idx,
        )
        aggregate_cold = cold_base.copy(deep=True).set_common_stream_attribute(
            "heat_flow",
            0.0,
            idx=period_idx,
        )
        hot_totals = [0.0] * len(aggregate_hot)
        cold_totals = [0.0] * len(aggregate_cold)

        def duty(utility) -> float:
            return float(get_scalar_value(utility.heat_flow, period_idx=period_idx))

        try:
            for address in self._aggregate_profile_addresses():
                targeted_hot, targeted_cold = (
                    _target_prepared_utility_load_profile_duties(
                        self._prepared_aggregate_profiles[(period.period_id, address)],
                        hot_utilities=hot_base,
                        cold_utilities=cold_base,
                        period_idx=period_idx,
                    )
                )
                for totals, targeted in (
                    (hot_totals, targeted_hot),
                    (cold_totals, targeted_cold),
                ):
                    for index, utility_duty in enumerate(targeted):
                        totals[index] += utility_duty

            for aggregate, totals in (
                (aggregate_hot, hot_totals),
                (aggregate_cold, cold_totals),
            ):
                for utility, total in zip(aggregate, totals, strict=True):
                    if total > 0.0:
                        utility.set_value_attr_at_idx(
                            "heat_flow",
                            total,
                            idx=period_idx,
                        )

            profile = _build_site_utility_profile(
                hot_utilities=aggregate_hot,
                cold_utilities=aggregate_cold,
                is_shifted=False,
                idx=period_idx,
            )
            net_utility = np.asarray(
                profile["updates"][ProblemTableLabel.H_NET_UT],
                dtype=float,
            )
            sugcc = ProblemTable(
                {
                    ProblemTableLabel.T: np.asarray(profile["T_col"], dtype=float),
                    ProblemTableLabel.H_NET_UT: net_utility,
                }
            )
            temperatures = np.asarray(sugcc[ProblemTableLabel.T], dtype=float)
            net_utility = np.asarray(
                sugcc[ProblemTableLabel.H_NET_UT],
                dtype=float,
            )
            required_hot = float(net_utility[0])
            required_cold = float(net_utility[-1])
            hot_profile, cold_profile = _load_profiles(
                sugcc,
                net_label=ProblemTableLabel.H_NET_UT,
            )
            hot_profile = _calibrate_profile(
                hot_profile,
                residual_duty=required_hot,
            )
            cold_profile = _calibrate_profile(
                cold_profile,
                residual_duty=required_cold,
            )
            hot_pinch, cold_pinch, *_ = sugcc.pinch_idx(ProblemTableLabel.H_NET_UT)
        except (ArithmeticError, ValueError) as exc:
            return AllocationAdapterResult(
                hot_duties=(0.0,) * len(placement.hot),
                cold_duties=(0.0,) * len(placement.cold),
                diagnostics=(
                    CandidateDiagnostic(
                        code="targeting_infeasible",
                        constraint="utility_allocation",
                        message=str(exc) or "Utility allocation is infeasible.",
                        period_id=period.period_id,
                    ),
                ),
            )

        matched_hot, matched_cold = _match_utility_gen_and_use_at_same_level(
            hot_utilities=aggregate_hot,
            cold_utilities=aggregate_cold,
            period_idx=period_idx,
        )
        composite = self._prepared_aggregate_composites[period.period_id]
        snapshot = PlacementTargetSnapshot(
            shifted_temperatures=tuple(float(value) for value in temperatures),
            real_temperatures=composite.temperatures,
            hot_load_profile=hot_profile,
            cold_load_profile=cold_profile,
            real_hot_composite=composite.hot_composite,
            real_cold_composite=composite.cold_composite,
            hot_pinch_index=int(hot_pinch),
            cold_pinch_index=int(cold_pinch),
            entropy_slices=period.snapshot.entropy_slices,
        )
        return self._allocation_result(
            hot_utilities=matched_hot,
            cold_utilities=matched_cold,
            period_idx=period_idx,
            placement=placement,
            required_hot=required_hot,
            required_cold=required_cold,
            snapshot=snapshot,
        )

    def allocate(
        self,
        period: PlacementPeriodInput,
        placement: DecodedPlacement,
    ) -> AllocationAdapterResult:
        if self.request.base_target is UtilityPlacementBaseTarget.DIRECT:
            return self._allocate_exact_target_replay(period, placement)
        return self._allocate_cached_aggregate(period, placement)


def run_problem_utility_placement(
    problem: "PinchProblem",
    *,
    isothermal: int | None = None,
    sensible: int | None = None,
    zone=None,
    period_ids=None,
    maximum_duties=None,
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
        maximum_duties=maximum_duties,
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
        allocation_adapter=_ExactTargetReplayAdapter(
            source=problem.to_problem_json(),
            project_name=problem.project_name,
            request=resolved_request,
        ),
    )
    period = result.best.period_results[0]
    levels_by_side = {
        "Hot": list(period.hot_levels),
        "Cold": list(period.cold_levels),
    }
    for period_result in result.best.period_results:
        for utility_type, levels in (
            ("Hot", period_result.hot_levels),
            ("Cold", period_result.cold_levels),
        ):
            observed = {
                level.template_key.name for level in levels_by_side[utility_type]
            }
            levels_by_side[utility_type].extend(
                level
                for level in levels
                if level.is_fallback and level.template_key.name not in observed
            )

    optimized_input = problem.to_problem_json()
    optimized_input["utilities"] = [
        {
            "name": level.template_key.name,
            "type": utility_type,
            "t_supply": level.supply_temperature.model_dump(),
            "t_target": level.target_temperature.model_dump(),
            "heat_flow": {"value": 0.0, "unit": result.units.heat_flow},
            **(
                {
                    "maximum_heat_flow": _serialized_limit(
                        result.request,
                        level.template_key.name,
                    )
                }
                if _serialized_limit(result.request, level.template_key.name)
                is not None
                else {}
            ),
            "dt_cont": {"value": 0.0, "unit": result.units.temperature_difference},
            "htc": {"value": 1.0, "unit": "kW/m^2/delta_degC"},
        }
        for utility_type, levels in levels_by_side.items()
        for level in levels
    ]
    from .problem import PinchProblem

    optimized_case = PinchProblem(optimized_input, project_name=problem.project_name)
    optimized_case._utility_placement_result = result
    return optimized_case


__all__ = ["build_problem_placement_context", "run_problem_utility_placement"]
