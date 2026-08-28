"""Stable vector schema, encode/decode, and independent verification."""

from __future__ import annotations

import math
from collections.abc import Sequence

from OpenPinch.contracts.utility_placement import (
    CandidateDiagnostic,
    CandidateVerification,
    CoordinateKey,
    DecisionCoordinate,
    DecisionField,
    DecodedPlacement,
    DecodedUtilityLevel,
    EffectiveUtilityTemplate,
    PlacementFeasibilityEnvelope,
    QuantityInterval,
    QuantityValue,
    TemplateBlueprintSet,
    UtilityLevelKind,
    UtilityPlacementModel,
    UtilityPlacementRequest,
    UtilitySide,
    UtilityTemplateSet,
)

from .bounds import build_initial_values, derive_effective_templates
from .errors import PlacementModelValidationError

ABSOLUTE_ZERO_C = -273.15
_SENSIBLE_START_FRACTIONS = (0.5, 0.2, 0.4, 0.6, 0.8)


def build_decision_coordinates(
    templates: UtilityTemplateSet,
) -> tuple[DecisionCoordinate, ...]:
    """Build the fixed coordinate-family sequence for one template set."""
    ordered: list[tuple[CoordinateKey, QuantityInterval]] = []
    for side_templates in (templates.hot, templates.cold):
        for template in side_templates:
            if template.kind is UtilityLevelKind.ISOTHERMAL:
                ordered.append(
                    (
                        CoordinateKey(
                            template_key=template.key,
                            field=DecisionField.SUPPLY_TEMPERATURE,
                        ),
                        template.supply_bounds,
                    )
                )
        for template in side_templates:
            if template.kind is UtilityLevelKind.SENSIBLE:
                ordered.append(
                    (
                        CoordinateKey(
                            template_key=template.key,
                            field=DecisionField.SUPPLY_TEMPERATURE,
                        ),
                        template.supply_bounds,
                    )
                )
                if template.span_bounds is None:
                    raise PlacementModelValidationError(
                        code="missing_sensible_span_bounds",
                        message="Sensible template is missing effective span bounds.",
                        template_key=template.key,
                    )
                ordered.append(
                    (
                        CoordinateKey(
                            template_key=template.key,
                            field=DecisionField.TEMPERATURE_SPAN,
                        ),
                        template.span_bounds,
                    )
                )
    return tuple(
        DecisionCoordinate(index=index, coordinate=key, bounds=bounds)
        for index, (key, bounds) in enumerate(ordered)
    )


def _diagnostic(
    code: str,
    constraint: str,
    message: str,
    *,
    coordinate: DecisionCoordinate | None = None,
    measured: float | None = None,
    limit: float | None = None,
) -> CandidateDiagnostic:
    unit = coordinate.bounds.unit if coordinate is not None else "dimensionless"
    key = coordinate.coordinate.template_key if coordinate is not None else None
    return CandidateDiagnostic(
        code=code,
        constraint=constraint,
        message=message,
        side=key.side if key is not None else None,
        template_key=key,
        measured=(
            QuantityValue(value=measured, unit=unit) if measured is not None else None
        ),
        limit=QuantityValue(value=limit, unit=unit) if limit is not None else None,
    )


def verify_candidate(
    model: UtilityPlacementModel,
    point: Sequence[float],
) -> CandidateVerification:
    """Return structured diagnostics without throwing for an ordinary point."""
    if len(point) != len(model.coordinates):
        return CandidateVerification(
            feasible=False,
            diagnostics=(
                CandidateDiagnostic(
                    code="dimension_mismatch",
                    constraint="vector_dimension",
                    message="Candidate length does not match the decision schema.",
                    details=(
                        ("expected", len(model.coordinates)),
                        ("observed", len(point)),
                    ),
                ),
            ),
        )
    diagnostics: list[CandidateDiagnostic] = []
    values: dict[CoordinateKey, float] = {}
    for coordinate, raw_value in zip(model.coordinates, point, strict=True):
        try:
            value = float(raw_value)
        except TypeError, ValueError:
            diagnostics.append(
                _diagnostic(
                    "non_finite_coordinate",
                    "coordinate_finiteness",
                    "Candidate coordinate must be a finite number.",
                    coordinate=coordinate,
                )
            )
            continue
        if not math.isfinite(value):
            diagnostics.append(
                _diagnostic(
                    "non_finite_coordinate",
                    "coordinate_finiteness",
                    "Candidate coordinate must be a finite number.",
                    coordinate=coordinate,
                )
            )
            continue
        values[coordinate.coordinate] = 0.0 if value == 0.0 else value
        if (
            value < coordinate.bounds.lower - model.request.tolerances.bounds
            or value > coordinate.bounds.upper + model.request.tolerances.bounds
        ):
            limit = (
                coordinate.bounds.lower
                if value < coordinate.bounds.lower
                else coordinate.bounds.upper
            )
            diagnostics.append(
                _diagnostic(
                    "coordinate_out_of_bounds",
                    "coordinate_bounds",
                    "Candidate coordinate lies outside its effective bounds.",
                    coordinate=coordinate,
                    measured=value,
                    limit=limit,
                )
            )
    if diagnostics:
        return CandidateVerification(feasible=False, diagnostics=tuple(diagnostics))

    separation = model.envelope.minimum_separation.value
    for side, templates in (
        (UtilitySide.HOT, model.templates.hot),
        (UtilitySide.COLD, model.templates.cold),
    ):
        supplies = [
            values[
                CoordinateKey(
                    template_key=template.key,
                    field=DecisionField.SUPPLY_TEMPERATURE,
                )
            ]
            for template in templates
        ]
        for index in range(len(supplies) - 1):
            actual = (
                supplies[index] - supplies[index + 1]
                if side is UtilitySide.HOT
                else supplies[index + 1] - supplies[index]
            )
            if actual < separation - model.request.tolerances.ordering:
                template = templates[index + 1]
                coordinate = next(
                    item
                    for item in model.coordinates
                    if item.coordinate
                    == CoordinateKey(
                        template_key=template.key,
                        field=DecisionField.SUPPLY_TEMPERATURE,
                    )
                )
                diagnostics.append(
                    _diagnostic(
                        "ordering_violation",
                        "minimum_separation",
                        "Adjacent utility supplies violate physical ordering.",
                        coordinate=coordinate,
                        measured=actual,
                        limit=separation,
                    )
                )
    for template in model.templates.all:
        supply = values[
            CoordinateKey(
                template_key=template.key,
                field=DecisionField.SUPPLY_TEMPERATURE,
            )
        ]
        if template.kind is UtilityLevelKind.ISOTHERMAL:
            if template.fixed_span is None:
                raise PlacementModelValidationError(
                    code="missing_fixed_span",
                    message="Isothermal template is missing its fixed span.",
                    template_key=template.key,
                )
            span = template.fixed_span.value
        else:
            span = values[
                CoordinateKey(
                    template_key=template.key,
                    field=DecisionField.TEMPERATURE_SPAN,
                )
            ]
        target = (
            supply - span if template.key.side is UtilitySide.HOT else supply + span
        )
        if supply <= ABSOLUTE_ZERO_C or target <= ABSOLUTE_ZERO_C:
            coordinate = next(
                item
                for item in model.coordinates
                if item.coordinate.template_key == template.key
                and item.coordinate.field is DecisionField.SUPPLY_TEMPERATURE
            )
            diagnostics.append(
                _diagnostic(
                    "nonpositive_kelvin",
                    "absolute_temperature",
                    "Supply and target temperatures must remain above absolute zero.",
                    coordinate=coordinate,
                    measured=min(supply, target),
                    limit=ABSOLUTE_ZERO_C,
                )
            )
    return CandidateVerification(
        feasible=not diagnostics,
        diagnostics=tuple(diagnostics),
    )


def _raise_for_verification(verification: CandidateVerification) -> None:
    diagnostic = verification.diagnostics[0]
    raise PlacementModelValidationError(
        code=diagnostic.code,
        message=diagnostic.message,
        template_key=diagnostic.template_key,
        field_path=diagnostic.constraint,
        details=diagnostic.details,
    )


def decode_placement(
    model: UtilityPlacementModel,
    point: Sequence[float],
) -> DecodedPlacement:
    """Strictly decode one valid decision point into declaration-order levels."""
    verification = verify_candidate(model, point)
    if not verification.feasible:
        _raise_for_verification(verification)
    normalized_point = tuple(
        0.0 if float(value) == 0.0 else float(value) for value in point
    )
    values = {
        coordinate.coordinate: normalized_point[coordinate.index]
        for coordinate in model.coordinates
    }

    def decode_level(template: EffectiveUtilityTemplate) -> DecodedUtilityLevel:
        supply = values[
            CoordinateKey(
                template_key=template.key,
                field=DecisionField.SUPPLY_TEMPERATURE,
            )
        ]
        if template.kind is UtilityLevelKind.ISOTHERMAL:
            if template.fixed_span is None:
                raise PlacementModelValidationError(
                    code="missing_fixed_span",
                    message="Isothermal template is missing its fixed span.",
                    template_key=template.key,
                )
            span = template.fixed_span.value
        else:
            span = values[
                CoordinateKey(
                    template_key=template.key,
                    field=DecisionField.TEMPERATURE_SPAN,
                )
            ]
        target = (
            supply - span if template.key.side is UtilitySide.HOT else supply + span
        )
        return DecodedUtilityLevel(
            template_key=template.key,
            kind=template.kind,
            placement_rank=template.placement_rank,
            supply_temperature=QuantityValue(
                value=supply,
                unit=model.request.units.absolute_temperature,
            ),
            target_temperature=QuantityValue(
                value=target,
                unit=model.request.units.absolute_temperature,
            ),
            temperature_span=QuantityValue(
                value=span,
                unit=model.request.units.temperature_difference,
            ),
        )

    return DecodedPlacement(
        hot=tuple(decode_level(template) for template in model.templates.hot),
        cold=tuple(decode_level(template) for template in model.templates.cold),
        coordinates=normalized_point,
    )


def encode_placement(
    model: UtilityPlacementModel,
    placement: DecodedPlacement,
) -> tuple[float, ...]:
    """Encode one complete decoded placement in the model's stable schema."""
    levels = placement.hot + placement.cold
    by_key = {level.template_key: level for level in levels}
    expected = {template.key for template in model.templates.all}
    if len(by_key) != len(levels) or set(by_key) != expected:
        raise PlacementModelValidationError(
            code="placement_identity_mismatch",
            message="Decoded placement must contain every model template exactly once.",
            field_path="placement",
        )
    encoded: list[float] = []
    for coordinate in model.coordinates:
        level = by_key[coordinate.coordinate.template_key]
        value = (
            level.supply_temperature.value
            if coordinate.coordinate.field is DecisionField.SUPPLY_TEMPERATURE
            else level.temperature_span.value
        )
        encoded.append(0.0 if value == 0.0 else value)
    point = tuple(encoded)
    verification = verify_candidate(model, point)
    if not verification.feasible:
        _raise_for_verification(verification)
    return point


def verify_placement(
    model: UtilityPlacementModel,
    placement: DecodedPlacement,
) -> CandidateVerification:
    """Verify a decoded placement without exposing encoding exceptions."""
    try:
        point = encode_placement(model, placement)
    except PlacementModelValidationError as exc:
        return CandidateVerification(
            feasible=False,
            diagnostics=(
                CandidateDiagnostic(
                    code=exc.code,
                    constraint=str(exc.context.get("field_path", "placement")),
                    message=str(exc),
                ),
            ),
        )
    return verify_candidate(model, point)


def build_utility_placement_model(
    request: UtilityPlacementRequest,
    blueprints: TemplateBlueprintSet,
    envelope: PlacementFeasibilityEnvelope,
) -> UtilityPlacementModel:
    """Build a complete immutable model and independently verify its start."""
    templates = derive_effective_templates(request, blueprints, envelope)
    coordinates = build_decision_coordinates(templates)
    initial_values = build_initial_values(
        templates,
        envelope.minimum_separation.value,
    )
    initial_points: list[tuple[float, ...]] = []
    for fraction in _SENSIBLE_START_FRACTIONS:
        values = dict(initial_values)
        for coordinate in coordinates:
            if coordinate.coordinate.field is DecisionField.TEMPERATURE_SPAN:
                values[coordinate.coordinate] = coordinate.bounds.lower + fraction * (
                    coordinate.bounds.upper - coordinate.bounds.lower
                )
        point = tuple(values[coordinate.coordinate] for coordinate in coordinates)
        if point not in initial_points:
            initial_points.append(point)
    model = UtilityPlacementModel(
        request=request,
        envelope=envelope,
        templates=templates,
        coordinates=coordinates,
        initial_points=tuple(initial_points),
    )
    for point in model.initial_points:
        verification = verify_candidate(model, point)
        if not verification.feasible:
            raise PlacementModelValidationError(
                code="invalid_generated_start",
                message="Generated start failed independent candidate verification.",
                details=tuple(
                    ("diagnostic", diagnostic.code)
                    for diagnostic in verification.diagnostics
                ),
            )
    return model


__all__ = [
    "build_decision_coordinates",
    "build_utility_placement_model",
    "decode_placement",
    "encode_placement",
    "verify_candidate",
    "verify_placement",
]
