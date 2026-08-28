"""Physical-bound reduction, ordering propagation, and deterministic starts."""

from __future__ import annotations

from collections.abc import Iterable

from OpenPinch.contracts.utility_placement import (
    CoordinateKey,
    DecisionField,
    EffectiveUtilityTemplate,
    PhysicalCoordinateBound,
    PlacementFeasibilityEnvelope,
    QuantityInterval,
    TemplateBlueprintSet,
    UtilityLevelKind,
    UtilityPlacementRequest,
    UtilitySide,
    UtilityTemplateBlueprint,
    UtilityTemplateSet,
)

from .errors import (
    EmptyPlacementFeasibleRegionError,
    PlacementModelValidationError,
    UtilityTemplateValidationError,
)

ABSOLUTE_ZERO_C = -273.15


def _expected_coordinates(
    blueprints: TemplateBlueprintSet,
) -> tuple[CoordinateKey, ...]:
    expected: list[CoordinateKey] = []
    for blueprint in blueprints.all:
        expected.append(
            CoordinateKey(
                template_key=blueprint.key,
                field=DecisionField.SUPPLY_TEMPERATURE,
            )
        )
        if blueprint.kind is UtilityLevelKind.SENSIBLE:
            expected.append(
                CoordinateKey(
                    template_key=blueprint.key,
                    field=DecisionField.TEMPERATURE_SPAN,
                )
            )
    return tuple(expected)


def _period_index(
    blueprints: TemplateBlueprintSet,
    envelope: PlacementFeasibilityEnvelope,
) -> dict[CoordinateKey, tuple[PhysicalCoordinateBound, ...]]:
    expected = set(_expected_coordinates(blueprints))
    collected = {coordinate: [] for coordinate in expected}
    if not envelope.periods:
        raise PlacementModelValidationError(
            code="missing_periods",
            message="The feasibility envelope must contain at least one period.",
            field_path="envelope.periods",
        )
    period_ids: set[str] = set()
    positive_weight = False
    for period in envelope.periods:
        if period.period_id in period_ids:
            raise PlacementModelValidationError(
                code="duplicate_period",
                message="Envelope period identifiers must be unique.",
                period_id=period.period_id,
            )
        period_ids.add(period.period_id)
        positive_weight = positive_weight or period.weight > 0.0
        observed = [bound.coordinate for bound in period.coordinate_bounds]
        if len(observed) != len(set(observed)) or set(observed) != expected:
            raise PlacementModelValidationError(
                code="coordinate_coverage_mismatch",
                message=(
                    "Each envelope period must contain every coordinate exactly once."
                ),
                period_id=period.period_id,
                details=(
                    ("expected_count", len(expected)),
                    ("observed_count", len(observed)),
                ),
            )
        for bound in period.coordinate_bounds:
            collected[bound.coordinate].append(bound)
    if not positive_weight:
        raise PlacementModelValidationError(
            code="missing_positive_period_weight",
            message="At least one envelope period must have positive weight.",
            field_path="envelope.periods",
        )
    return {coordinate: tuple(bounds) for coordinate, bounds in collected.items()}


def _intersection(
    coordinate: CoordinateKey,
    period_bounds: tuple[PhysicalCoordinateBound, ...],
    *,
    expected_unit: str,
    caller_bounds: QuantityInterval | None,
    tolerance: float,
) -> QuantityInterval:
    if any(bound.bounds.unit != expected_unit for bound in period_bounds):
        raise PlacementModelValidationError(
            code="noncanonical_envelope_unit",
            message="Envelope coordinate bounds must use canonical units.",
            template_key=coordinate.template_key,
            field_path=coordinate.field.value,
        )
    physical_lower = max(bound.bounds.lower for bound in period_bounds)
    physical_upper = min(bound.bounds.upper for bound in period_bounds)
    if physical_lower > physical_upper + tolerance:
        raise EmptyPlacementFeasibleRegionError(
            code="empty_period_intersection",
            message="Period coordinate bounds have no common intersection.",
            template_key=coordinate.template_key,
            field_path=coordinate.field.value,
            details=(
                ("physical_lower", physical_lower),
                ("physical_upper", physical_upper),
            ),
        )
    if abs(physical_lower - physical_upper) <= tolerance:
        physical_lower = physical_upper = (physical_lower + physical_upper) / 2.0
    if caller_bounds is None:
        lower, upper = physical_lower, physical_upper
    else:
        if caller_bounds.unit != expected_unit:
            raise UtilityTemplateValidationError(
                code="noncanonical_caller_bounds",
                message="Caller bounds must be normalized before model construction.",
                template_key=coordinate.template_key,
                field_path=coordinate.field.value,
            )
        if (
            caller_bounds.lower < physical_lower - tolerance
            or caller_bounds.upper > physical_upper + tolerance
        ):
            raise UtilityTemplateValidationError(
                code="caller_bounds_expand_physical_region",
                message="Caller bounds may narrow but cannot expand physical bounds.",
                template_key=coordinate.template_key,
                field_path=coordinate.field.value,
            )
        lower = max(caller_bounds.lower, physical_lower)
        upper = min(caller_bounds.upper, physical_upper)
    if lower > upper + tolerance:
        raise EmptyPlacementFeasibleRegionError(
            code="empty_caller_intersection",
            message="Caller and physical bounds have no common intersection.",
            template_key=coordinate.template_key,
            field_path=coordinate.field.value,
        )
    if abs(lower - upper) <= tolerance:
        lower = upper = (lower + upper) / 2.0
    return QuantityInterval(lower=lower, upper=upper, unit=expected_unit)


def _build_effective_template(
    blueprint: UtilityTemplateBlueprint,
    indexed_bounds: dict[CoordinateKey, tuple[PhysicalCoordinateBound, ...]],
    request: UtilityPlacementRequest,
) -> EffectiveUtilityTemplate:
    supply_key = CoordinateKey(
        template_key=blueprint.key,
        field=DecisionField.SUPPLY_TEMPERATURE,
    )
    supply_period_bounds = indexed_bounds[supply_key]
    supply_bounds = _intersection(
        supply_key,
        supply_period_bounds,
        expected_unit=request.units.absolute_temperature,
        caller_bounds=blueprint.supply_bounds,
        tolerance=request.tolerances.bounds,
    )
    span_bounds = None
    span_period_bounds: tuple[PhysicalCoordinateBound, ...] = ()
    if blueprint.kind is UtilityLevelKind.SENSIBLE:
        span_key = CoordinateKey(
            template_key=blueprint.key,
            field=DecisionField.TEMPERATURE_SPAN,
        )
        span_period_bounds = indexed_bounds[span_key]
        span_bounds = _intersection(
            span_key,
            span_period_bounds,
            expected_unit=request.units.temperature_difference,
            caller_bounds=blueprint.span_bounds,
            tolerance=request.tolerances.bounds,
        )
        if span_bounds.lower <= 0.0:
            raise EmptyPlacementFeasibleRegionError(
                code="nonpositive_sensible_span",
                message="Sensible temperature spans must remain positive.",
                template_key=blueprint.key,
                field_path=DecisionField.TEMPERATURE_SPAN.value,
            )
    maximum_span = (
        blueprint.fixed_span.value
        if blueprint.fixed_span is not None
        else span_bounds.upper
        if span_bounds is not None
        else 0.0
    )
    minimum_supply = (
        ABSOLUTE_ZERO_C + maximum_span
        if blueprint.key.side is UtilitySide.HOT
        else ABSOLUTE_ZERO_C
    )
    if supply_bounds.lower <= minimum_supply:
        raise EmptyPlacementFeasibleRegionError(
            code="nonpositive_kelvin_region",
            message="Accepted temperature bounds must remain above absolute zero.",
            template_key=blueprint.key,
            field_path=DecisionField.SUPPLY_TEMPERATURE.value,
        )
    return EffectiveUtilityTemplate(
        key=blueprint.key,
        kind=blueprint.kind,
        placement_rank=blueprint.placement_rank,
        supply_bounds=supply_bounds,
        fixed_span=blueprint.fixed_span,
        span_bounds=span_bounds,
        fluid=blueprint.fluid,
        supply_period_bounds=supply_period_bounds,
        span_period_bounds=span_period_bounds,
        caller_supply_bounds=blueprint.supply_bounds,
        caller_span_bounds=blueprint.span_bounds,
    )


def _propagate_order(
    templates: tuple[EffectiveUtilityTemplate, ...],
    *,
    side: UtilitySide,
    separation: float,
    tolerance: float,
) -> tuple[EffectiveUtilityTemplate, ...]:
    lower = [template.supply_bounds.lower for template in templates]
    upper = [template.supply_bounds.upper for template in templates]
    if side is UtilitySide.HOT:
        for index in range(len(templates) - 2, -1, -1):
            lower[index] = max(lower[index], lower[index + 1] + separation)
        for index in range(1, len(templates)):
            upper[index] = min(upper[index], upper[index - 1] - separation)
    else:
        for index in range(1, len(templates)):
            lower[index] = max(lower[index], lower[index - 1] + separation)
        for index in range(len(templates) - 2, -1, -1):
            upper[index] = min(upper[index], upper[index + 1] - separation)
    updated: list[EffectiveUtilityTemplate] = []
    for index, template in enumerate(templates):
        if lower[index] > upper[index] + tolerance:
            raise EmptyPlacementFeasibleRegionError(
                code="empty_ordered_bounds",
                message="Adjacent utility ordering leaves no feasible supply bounds.",
                template_key=template.key,
                field_path=DecisionField.SUPPLY_TEMPERATURE.value,
                details=(
                    ("lower", lower[index]),
                    ("upper", upper[index]),
                    ("separation", separation),
                ),
            )
        if abs(lower[index] - upper[index]) <= tolerance:
            lower[index] = upper[index] = (lower[index] + upper[index]) / 2.0
        updated.append(
            template.model_copy(
                update={
                    "supply_bounds": QuantityInterval(
                        lower=lower[index],
                        upper=upper[index],
                        unit=template.supply_bounds.unit,
                    )
                }
            )
        )
    return tuple(updated)


def derive_effective_templates(
    request: UtilityPlacementRequest,
    blueprints: TemplateBlueprintSet,
    envelope: PlacementFeasibilityEnvelope,
) -> UtilityTemplateSet:
    """Intersect all period bounds and propagate physical side ordering."""
    if envelope.minimum_separation.unit != request.units.temperature_difference:
        raise PlacementModelValidationError(
            code="noncanonical_separation_unit",
            message="Minimum separation must use the canonical difference unit.",
            field_path="envelope.minimum_separation",
        )
    indexed_bounds = _period_index(blueprints, envelope)
    hot = tuple(
        _build_effective_template(blueprint, indexed_bounds, request)
        for blueprint in blueprints.hot
    )
    cold = tuple(
        _build_effective_template(blueprint, indexed_bounds, request)
        for blueprint in blueprints.cold
    )
    separation = envelope.minimum_separation.value
    return UtilityTemplateSet(
        hot=_propagate_order(
            hot,
            side=UtilitySide.HOT,
            separation=separation,
            tolerance=request.tolerances.ordering,
        ),
        cold=_propagate_order(
            cold,
            side=UtilitySide.COLD,
            separation=separation,
            tolerance=request.tolerances.ordering,
        ),
    )


def _side_supply_values(
    templates: Iterable[EffectiveUtilityTemplate],
    *,
    side: UtilitySide,
    separation: float,
) -> dict[CoordinateKey, float]:
    ordered_templates = tuple(templates)
    result: dict[CoordinateKey, float] = {}
    adjacent: float | None = None
    for template in reversed(ordered_templates):
        bounds = template.supply_bounds
        if adjacent is None:
            value = bounds.lower if side is UtilitySide.HOT else bounds.upper
        elif side is UtilitySide.HOT:
            value = max(bounds.lower, adjacent + separation)
        else:
            value = min(bounds.upper, adjacent - separation)
        if value < bounds.lower or value > bounds.upper:
            raise PlacementModelValidationError(
                code="invalid_initial_supply",
                message="A feasible model failed deterministic start construction.",
                template_key=template.key,
            )
        result[
            CoordinateKey(
                template_key=template.key,
                field=DecisionField.SUPPLY_TEMPERATURE,
            )
        ] = 0.0 if value == 0.0 else value
        adjacent = value
    return result


def build_initial_values(
    templates: UtilityTemplateSet,
    minimum_separation: float,
) -> dict[CoordinateKey, float]:
    """Build a deterministic feasible start nearest the process-temperature envelope."""
    values = _side_supply_values(
        templates.hot,
        side=UtilitySide.HOT,
        separation=minimum_separation,
    )
    values.update(
        _side_supply_values(
            templates.cold,
            side=UtilitySide.COLD,
            separation=minimum_separation,
        )
    )
    for template in templates.all:
        if template.kind is UtilityLevelKind.SENSIBLE:
            if template.span_bounds is None:
                raise PlacementModelValidationError(
                    code="missing_sensible_span_bounds",
                    message="Sensible templates require effective span bounds.",
                    template_key=template.key,
                )
            midpoint = (template.span_bounds.lower + template.span_bounds.upper) / 2.0
            values[
                CoordinateKey(
                    template_key=template.key,
                    field=DecisionField.TEMPERATURE_SPAN,
                )
            ] = 0.0 if midpoint == 0.0 else midpoint
    return values


__all__ = [
    "build_initial_values",
    "derive_effective_templates",
]
