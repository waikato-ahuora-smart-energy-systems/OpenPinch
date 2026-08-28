"""Pure request, template, and unit normalization."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from pint.errors import DimensionalityError, OffsetUnitCalculusError, UndefinedUnitError
from pydantic import ValidationError

from OpenPinch.contracts.common import ValueWithUnit
from OpenPinch.contracts.utility_placement import (
    QuantityInterval,
    QuantityValue,
    TemplateBlueprintSet,
    TemplateKey,
    UtilityLevelKind,
    UtilityLevelTemplate,
    UtilityPlacementRequest,
    UtilitySide,
    UtilityTemplateBlueprint,
)
from OpenPinch.domain.value import Value

from .errors import (
    PlacementRequestValidationError,
    UtilityPlacementUnitError,
    UtilityTemplateValidationError,
)

ConversionAdapter = Callable[..., float]


def convert_placement_value(
    value: float | int | ValueWithUnit | QuantityValue | Value,
    *,
    canonical_unit: str,
    default_unit: str,
    field_path: str,
) -> float:
    """Convert one scalar through the existing OpenPinch ``Value`` owner."""
    if isinstance(value, bool):
        raise UtilityPlacementUnitError(
            code="invalid_quantity",
            message="Boolean values are not valid physical quantities.",
            field_path=field_path,
        )
    if isinstance(value, Value):
        source = value
        source_unit = value.unit
    elif isinstance(value, QuantityValue | ValueWithUnit):
        if value.value is None:
            raise UtilityPlacementUnitError(
                code="missing_quantity",
                message="A quantity magnitude is required.",
                field_path=field_path,
            )
        source_unit = value.unit or default_unit
        source = Value(value.value, source_unit)
    elif isinstance(value, int | float):
        source_unit = default_unit
        source = Value(value, source_unit)
    else:
        raise UtilityPlacementUnitError(
            code="invalid_quantity",
            message="Unsupported utility-placement quantity type.",
            field_path=field_path,
            details=(("type", type(value).__name__),),
        )

    try:
        if source.num_periods != 1:
            raise ValueError("multiperiod values are not scalar quantities")
        result = float(source.to(canonical_unit))
    except (
        DimensionalityError,
        OffsetUnitCalculusError,
        UndefinedUnitError,
        TypeError,
        ValueError,
    ) as exc:
        raise UtilityPlacementUnitError(
            code="incompatible_unit",
            message="Quantity unit is not compatible with the required dimension.",
            field_path=field_path,
            details=(
                ("source_unit", str(source_unit)),
                ("canonical_unit", canonical_unit),
            ),
        ) from exc
    if not math.isfinite(result):
        raise UtilityPlacementUnitError(
            code="non_finite_quantity",
            message="Converted quantity must be finite.",
            field_path=field_path,
        )
    return 0.0 if result == 0.0 else result


def normalize_utility_placement_request(
    request: UtilityPlacementRequest | None = None,
    *,
    isothermal_level_count: int | None = None,
    sensible_level_count: int = 0,
    **values: Any,
) -> UtilityPlacementRequest:
    """Normalize public arguments or detach an already normalized request."""
    if request is not None:
        if isothermal_level_count is not None or values or sensible_level_count != 0:
            raise PlacementRequestValidationError(
                code="ambiguous_request",
                message="Supply either a request or public arguments, not both.",
            )
        parsed = UtilityPlacementRequest.model_validate(request.model_dump())
        return _normalize_option_quantities(parsed)
    if isothermal_level_count is None:
        raise PlacementRequestValidationError(
            code="missing_isothermal_count",
            message="isothermal_level_count is required.",
            field_path="isothermal_level_count",
        )
    try:
        parsed = UtilityPlacementRequest(
            isothermal_level_count=isothermal_level_count,
            sensible_level_count=sensible_level_count,
            **values,
        )
        return _normalize_option_quantities(parsed)
    except ValidationError as exc:
        first = exc.errors(include_url=False)[0]
        raise PlacementRequestValidationError(
            code="invalid_request",
            message=f"Utility-placement request validation failed: {first['msg']}.",
            field_path=".".join(str(part) for part in first["loc"]),
            details=(("reason", first["msg"]),),
        ) from exc


def _normalize_option_quantities(
    request: UtilityPlacementRequest,
) -> UtilityPlacementRequest:
    options = request.options
    canonical_unit = request.units.temperature_difference
    updates = {
        field_name: QuantityValue(
            value=convert_placement_value(
                getattr(options, field_name),
                canonical_unit=canonical_unit,
                default_unit=canonical_unit,
                field_path=f"options.{field_name}",
            ),
            unit=canonical_unit,
        )
        for field_name in (
            "minimum_separation",
            "minimum_sensible_span",
            "default_isothermal_span",
        )
    }
    return request.model_copy(update={"options": options.model_copy(update=updates)})


def _generated_templates(
    request: UtilityPlacementRequest,
    side: UtilitySide,
) -> tuple[UtilityLevelTemplate, ...]:
    templates: list[UtilityLevelTemplate] = []
    for ordinal in range(1, request.isothermal_level_count + 1):
        templates.append(
            UtilityLevelTemplate(
                name=f"{side.value}_iso_{ordinal}",
                side=side,
                kind=UtilityLevelKind.ISOTHERMAL,
                fixed_span=request.options.default_isothermal_span,
            )
        )
    for ordinal in range(1, request.sensible_level_count + 1):
        templates.append(
            UtilityLevelTemplate(
                name=f"{side.value}_sensible_{ordinal}",
                side=side,
                kind=UtilityLevelKind.SENSIBLE,
            )
        )
    return tuple(templates)


def _normalize_interval(
    interval: QuantityInterval | None,
    *,
    canonical_unit: str,
    field_path: str,
    convert_value: ConversionAdapter,
) -> QuantityInterval | None:
    if interval is None:
        return None
    lower = convert_value(
        QuantityValue(value=interval.lower, unit=interval.unit),
        canonical_unit=canonical_unit,
        default_unit=canonical_unit,
        field_path=f"{field_path}.lower",
    )
    upper = convert_value(
        QuantityValue(value=interval.upper, unit=interval.unit),
        canonical_unit=canonical_unit,
        default_unit=canonical_unit,
        field_path=f"{field_path}.upper",
    )
    return QuantityInterval(lower=lower, upper=upper, unit=canonical_unit)


def _normalize_quantity(
    quantity: QuantityValue | None,
    *,
    canonical_unit: str,
    field_path: str,
    convert_value: ConversionAdapter,
) -> QuantityValue | None:
    if quantity is None:
        return None
    return QuantityValue(
        value=convert_value(
            quantity,
            canonical_unit=canonical_unit,
            default_unit=canonical_unit,
            field_path=field_path,
        ),
        unit=canonical_unit,
    )


def _validate_inventory(
    request: UtilityPlacementRequest,
    side: UtilitySide,
    templates: tuple[UtilityLevelTemplate, ...],
) -> None:
    observed_iso = sum(
        template.kind is UtilityLevelKind.ISOTHERMAL for template in templates
    )
    observed_sensible = sum(
        template.kind is UtilityLevelKind.SENSIBLE for template in templates
    )
    if (
        observed_iso != request.isothermal_level_count
        or observed_sensible != request.sensible_level_count
    ):
        raise UtilityTemplateValidationError(
            code="template_inventory_mismatch",
            message="Explicit template inventory must match the requested counts.",
            field_path=f"{side.value}_templates",
            details=(
                ("expected_isothermal", request.isothermal_level_count),
                ("observed_isothermal", observed_iso),
                ("expected_sensible", request.sensible_level_count),
                ("observed_sensible", observed_sensible),
            ),
        )
    if any(template.side is not side for template in templates):
        raise UtilityTemplateValidationError(
            code="template_side_mismatch",
            message="Every template must agree with its declared side collection.",
            field_path=f"{side.value}_templates",
        )


def _normalize_template(
    template: UtilityLevelTemplate,
    placement_rank: int,
    request: UtilityPlacementRequest,
    *,
    convert_value: ConversionAdapter,
) -> UtilityTemplateBlueprint:
    prefix = f"{template.side.value}_templates.{placement_rank}"
    supply_bounds = _normalize_interval(
        template.supply_bounds,
        canonical_unit=request.units.absolute_temperature,
        field_path=f"{prefix}.supply_bounds",
        convert_value=convert_value,
    )
    fixed_span = _normalize_quantity(
        template.fixed_span,
        canonical_unit=request.units.temperature_difference,
        field_path=f"{prefix}.fixed_span",
        convert_value=convert_value,
    )
    span_bounds = _normalize_interval(
        template.span_bounds,
        canonical_unit=request.units.temperature_difference,
        field_path=f"{prefix}.span_bounds",
        convert_value=convert_value,
    )
    if template.kind is UtilityLevelKind.ISOTHERMAL:
        if span_bounds is not None:
            raise UtilityTemplateValidationError(
                code="isothermal_span_bounds",
                message="Isothermal templates use a fixed span, not span bounds.",
                field_path=f"{prefix}.span_bounds",
            )
        fixed_span = fixed_span or QuantityValue(
            value=request.options.default_isothermal_span.value,
            unit=request.units.temperature_difference,
        )
        if fixed_span.value <= 0.0:
            raise UtilityTemplateValidationError(
                code="invalid_isothermal_span",
                message="Isothermal fixed span must be positive.",
                field_path=f"{prefix}.fixed_span",
            )
    elif fixed_span is not None:
        raise UtilityTemplateValidationError(
            code="sensible_fixed_span",
            message="Sensible templates use span bounds, not a fixed span.",
            field_path=f"{prefix}.fixed_span",
        )
    if span_bounds is not None and span_bounds.lower <= 0.0:
        raise UtilityTemplateValidationError(
            code="invalid_sensible_span",
            message="Sensible span lower bound must be positive.",
            field_path=f"{prefix}.span_bounds",
        )
    return UtilityTemplateBlueprint(
        key=TemplateKey(side=template.side, name=template.name),
        kind=template.kind,
        placement_rank=placement_rank,
        supply_bounds=supply_bounds,
        fixed_span=fixed_span,
        span_bounds=span_bounds,
        fluid=template.fluid,
    )


def prepare_template_blueprints(
    request: UtilityPlacementRequest,
    *,
    convert_value: ConversionAdapter = convert_placement_value,
) -> TemplateBlueprintSet:
    """Return a complete canonical blueprint set without process dependencies."""
    normalized_request = normalize_utility_placement_request(request)
    hot_templates = normalized_request.hot_templates or _generated_templates(
        normalized_request, UtilitySide.HOT
    )
    cold_templates = normalized_request.cold_templates or _generated_templates(
        normalized_request, UtilitySide.COLD
    )
    _validate_inventory(normalized_request, UtilitySide.HOT, hot_templates)
    _validate_inventory(normalized_request, UtilitySide.COLD, cold_templates)
    hot = tuple(
        _normalize_template(
            template,
            rank,
            normalized_request,
            convert_value=convert_value,
        )
        for rank, template in enumerate(hot_templates)
    )
    cold = tuple(
        _normalize_template(
            template,
            rank,
            normalized_request,
            convert_value=convert_value,
        )
        for rank, template in enumerate(cold_templates)
    )
    names = [blueprint.key.name for blueprint in hot + cold]
    if len(set(names)) != len(names):
        raise UtilityTemplateValidationError(
            code="duplicate_template",
            message="Template names must be globally unique.",
            field_path="templates",
        )
    return TemplateBlueprintSet(hot=hot, cold=cold)


__all__ = [
    "convert_placement_value",
    "normalize_utility_placement_request",
    "prepare_template_blueprints",
]
