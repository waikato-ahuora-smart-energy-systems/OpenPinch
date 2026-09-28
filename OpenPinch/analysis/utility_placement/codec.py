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

from .bounds import (
    ABSOLUTE_ZERO_C,
    build_initial_values,
    derive_effective_templates,
)
from .errors import PlacementModelValidationError
from .start_points import generate_start_points


def build_decision_coordinates(
    templates: UtilityTemplateSet,
    *,
    envelope: PlacementFeasibilityEnvelope,
    paired: bool = False,
) -> tuple[DecisionCoordinate, ...]:
    """Build the fixed coordinate-family sequence for one template set."""
    ordered: list[tuple[CoordinateKey, QuantityInterval]] = []
    families = (templates.hot,) if paired else (templates.hot, templates.cold)
    for side_templates in families:
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


def _coordinate_for(
    model: UtilityPlacementModel,
    template: EffectiveUtilityTemplate,
    field: DecisionField,
) -> DecisionCoordinate:
    key = CoordinateKey(template_key=template.key, field=field)
    return next(item for item in model.coordinates if item.coordinate == key)


def _span_for(
    model: UtilityPlacementModel,
    template: EffectiveUtilityTemplate,
    values: dict[CoordinateKey, float],
) -> float:
    if template.kind is UtilityLevelKind.ISOTHERMAL:
        if template.fixed_span is None:
            raise PlacementModelValidationError(
                code="missing_fixed_span",
                message="Isothermal template is missing its fixed span.",
                template_key=template.key,
            )
        return template.fixed_span.value
    return values[
        CoordinateKey(
            template_key=template.key,
            field=DecisionField.TEMPERATURE_SPAN,
        )
    ]


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

    def check_descending(
        supplies: list[float],
        templates: tuple[EffectiveUtilityTemplate, ...],
    ) -> None:
        for index in range(len(supplies) - 1):
            actual = supplies[index] - supplies[index + 1]
            if actual < separation - model.request.tolerances.ordering:
                diagnostics.append(
                    _diagnostic(
                        "ordering_violation",
                        "minimum_separation",
                        "Adjacent utility supplies violate physical ordering.",
                        coordinate=_coordinate_for(
                            model,
                            templates[index + 1],
                            DecisionField.SUPPLY_TEMPERATURE,
                        ),
                        measured=actual,
                        limit=separation,
                    )
                )

    def check_physical_separation(
        entries: list[tuple[float, EffectiveUtilityTemplate]],
    ) -> None:
        ordered = sorted(entries, key=lambda item: item[0], reverse=True)
        check_descending(
            [supply for supply, _ in ordered],
            tuple(template for _, template in ordered),
        )

    if model.request.uses_generated_pairs:
        hot_supplies: list[float] = []
        cold_supplies: list[float] = []
        for hot_template, cold_template in zip(
            model.templates.hot,
            model.templates.cold,
            strict=True,
        ):
            supply = values[
                CoordinateKey(
                    template_key=hot_template.key,
                    field=DecisionField.SUPPLY_TEMPERATURE,
                )
            ]
            span = _span_for(model, hot_template, values)
            cold_supply = supply - span
            hot_supplies.append(supply)
            cold_supplies.append(cold_supply)
            cold_bounds = cold_template.supply_bounds
            if (
                cold_supply < cold_bounds.lower - model.request.tolerances.bounds
                or cold_supply > cold_bounds.upper + model.request.tolerances.bounds
            ):
                limit = (
                    cold_bounds.lower
                    if cold_supply < cold_bounds.lower
                    else cold_bounds.upper
                )
                diagnostics.append(
                    _diagnostic(
                        "paired_endpoint_out_of_bounds",
                        "paired_cold_supply_bounds",
                        "Reversed cold supply lies outside its effective bounds.",
                        coordinate=_coordinate_for(
                            model,
                            hot_template,
                            DecisionField.SUPPLY_TEMPERATURE,
                        ),
                        measured=cold_supply,
                        limit=limit,
                    )
                )
            if supply <= ABSOLUTE_ZERO_C or cold_supply <= ABSOLUTE_ZERO_C:
                diagnostics.append(
                    _diagnostic(
                        "nonpositive_kelvin",
                        "absolute_temperature",
                        (
                            "Supply and target temperatures must remain above "
                            "absolute zero."
                        ),
                        coordinate=_coordinate_for(
                            model,
                            hot_template,
                            DecisionField.SUPPLY_TEMPERATURE,
                        ),
                        measured=min(supply, cold_supply),
                        limit=ABSOLUTE_ZERO_C,
                    )
                )
        for kind in UtilityLevelKind:
            family_indices = [
                index
                for index, template in enumerate(model.templates.hot)
                if template.kind is kind
            ]
            family = tuple(model.templates.hot[index] for index in family_indices)
            check_descending(
                [hot_supplies[index] for index in family_indices],
                family,
            )
        check_physical_separation(
            list(zip(hot_supplies, model.templates.hot, strict=True))
        )
        check_physical_separation(
            list(zip(cold_supplies, model.templates.hot, strict=True))
        )
    else:
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
            ordered_supplies = supplies if side is UtilitySide.HOT else supplies[::-1]
            ordered_templates = (
                templates if side is UtilitySide.HOT else templates[::-1]
            )
            check_descending(ordered_supplies, ordered_templates)
        for template in model.templates.all:
            supply = values[
                CoordinateKey(
                    template_key=template.key,
                    field=DecisionField.SUPPLY_TEMPERATURE,
                )
            ]
            span = _span_for(model, template, values)
            target = (
                supply - span if template.key.side is UtilitySide.HOT else supply + span
            )
            if supply <= ABSOLUTE_ZERO_C or target <= ABSOLUTE_ZERO_C:
                diagnostics.append(
                    _diagnostic(
                        "nonpositive_kelvin",
                        "absolute_temperature",
                        (
                            "Supply and target temperatures must remain above "
                            "absolute zero."
                        ),
                        coordinate=_coordinate_for(
                            model,
                            template,
                            DecisionField.SUPPLY_TEMPERATURE,
                        ),
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
    return decode_verified_placement(model, point)


def decode_verified_placement(
    model: UtilityPlacementModel,
    point: Sequence[float],
) -> DecodedPlacement:
    """Decode a point that :func:`verify_candidate` has already accepted."""
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
        span = _span_for(model, template, values)
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

    hot = tuple(decode_level(template) for template in model.templates.hot)
    if model.request.uses_generated_pairs:
        cold = tuple(
            DecodedUtilityLevel(
                template_key=cold_template.key,
                kind=cold_template.kind,
                placement_rank=cold_template.placement_rank,
                supply_temperature=hot_level.target_temperature,
                target_temperature=hot_level.supply_temperature,
                temperature_span=hot_level.temperature_span,
            )
            for hot_level, cold_template in zip(
                hot,
                model.templates.cold,
                strict=True,
            )
        )
    else:
        cold = tuple(decode_level(template) for template in model.templates.cold)
    return DecodedPlacement(
        hot=hot,
        cold=cold,
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
    if model.request.uses_generated_pairs:
        for hot_template, cold_template in zip(
            model.templates.hot,
            model.templates.cold,
            strict=True,
        ):
            hot_level = by_key[hot_template.key]
            cold_level = by_key[cold_template.key]
            inverse_matches = (
                hot_level.kind is cold_level.kind
                and math.isclose(
                    cold_level.supply_temperature.value,
                    hot_level.target_temperature.value,
                    rel_tol=model.request.tolerances.relative,
                    abs_tol=model.request.tolerances.absolute,
                )
                and math.isclose(
                    cold_level.target_temperature.value,
                    hot_level.supply_temperature.value,
                    rel_tol=model.request.tolerances.relative,
                    abs_tol=model.request.tolerances.absolute,
                )
                and math.isclose(
                    cold_level.temperature_span.value,
                    hot_level.temperature_span.value,
                    rel_tol=model.request.tolerances.relative,
                    abs_tol=model.request.tolerances.absolute,
                )
            )
            if not inverse_matches:
                raise PlacementModelValidationError(
                    code="paired_endpoint_mismatch",
                    message=(
                        "Generated cold utility endpoints must exactly reverse "
                        "their hot partners."
                    ),
                    template_key=cold_template.key,
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
    coordinates = build_decision_coordinates(
        templates,
        envelope=envelope,
        paired=request.uses_generated_pairs,
    )
    initial_values = build_initial_values(
        templates,
        envelope.minimum_separation.value,
    )
    initial_points = generate_start_points(
        templates,
        coordinates,
        initial_values,
        envelope,
        paired=request.uses_generated_pairs,
    )
    provisional = UtilityPlacementModel(
        request=request,
        envelope=envelope,
        templates=templates,
        coordinates=coordinates,
        initial_points=(initial_points[0],),
    )
    feasible_points = tuple(
        point
        for point in initial_points
        if verify_candidate(provisional, point).feasible
    )
    if not feasible_points:
        verification = verify_candidate(provisional, initial_points[0])
        raise PlacementModelValidationError(
            code="invalid_generated_start",
            message="Generated starts failed independent candidate verification.",
            details=tuple(
                ("diagnostic", diagnostic.code)
                for diagnostic in verification.diagnostics
            ),
        )
    model = provisional.model_copy(update={"initial_points": feasible_points})
    return model


__all__ = [
    "build_decision_coordinates",
    "build_utility_placement_model",
    "decode_placement",
    "decode_verified_placement",
    "encode_placement",
    "verify_candidate",
    "verify_placement",
]
