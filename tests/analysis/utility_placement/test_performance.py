"""Representative performance evidence for the utility-placement pure model."""

from __future__ import annotations

import statistics
import time

from OpenPinch.analysis.utility_placement import (
    build_utility_placement_model,
    decode_placement,
    derive_effective_templates,
    normalize_utility_placement_request,
    prepare_template_blueprints,
    verify_candidate,
)
from OpenPinch.contracts.utility_placement import (
    CoordinateKey,
    DecisionField,
    PhysicalCoordinateBound,
    PlacementFeasibilityEnvelope,
    PlacementPeriodEnvelope,
    QuantityInterval,
    QuantityValue,
    UtilityLevelKind,
)


def _representative_case(period_count: int = 100):
    request = normalize_utility_placement_request(
        isothermal_level_count=10,
        sensible_level_count=10,
    )
    blueprints = prepare_template_blueprints(request)
    coordinates = []
    for blueprint in blueprints.all:
        coordinates.append(
            CoordinateKey(
                template_key=blueprint.key,
                field=DecisionField.SUPPLY_TEMPERATURE,
            )
        )
        if blueprint.kind is UtilityLevelKind.SENSIBLE:
            coordinates.append(
                CoordinateKey(
                    template_key=blueprint.key,
                    field=DecisionField.TEMPERATURE_SPAN,
                )
            )
    periods = tuple(
        PlacementPeriodEnvelope(
            period_id=f"period-{period_index}",
            weight=1.0 / period_count,
            coordinate_bounds=tuple(
                PhysicalCoordinateBound(
                    coordinate=coordinate,
                    bounds=QuantityInterval(
                        lower=(
                            1.0
                            if coordinate.field is DecisionField.TEMPERATURE_SPAN
                            else -100.0
                        ),
                        upper=(
                            50.0
                            if coordinate.field is DecisionField.TEMPERATURE_SPAN
                            else 500.0
                        ),
                        unit=(
                            "delta_degC"
                            if coordinate.field is DecisionField.TEMPERATURE_SPAN
                            else "degC"
                        ),
                    ),
                    reason="representative performance fixture",
                )
                for coordinate in coordinates
            ),
        )
        for period_index in range(period_count)
    )
    envelope = PlacementFeasibilityEnvelope(
        periods=periods,
        minimum_separation=QuantityValue(value=5.0, unit="delta_degC"),
        scope="direct",
        base_target_id="performance",
    )
    return request, blueprints, envelope


def test_representative_bound_reduction_p95_is_within_unit_budget() -> None:
    request, blueprints, envelope = _representative_case()
    durations = []
    for _ in range(8):
        started = time.perf_counter()
        result = derive_effective_templates(request, blueprints, envelope)
        durations.append(time.perf_counter() - started)
        assert len(result.all) == 40

    p95 = statistics.quantiles(durations, n=20)[18]
    assert p95 <= 0.150


def test_representative_complete_pure_model_p95_is_within_250_ms() -> None:
    request, blueprints, envelope = _representative_case()
    durations = []
    for _ in range(8):
        started = time.perf_counter()
        model = build_utility_placement_model(request, blueprints, envelope)
        placement = decode_placement(model, model.initial_points[0])
        verification = verify_candidate(model, placement.coordinates)
        durations.append(time.perf_counter() - started)
        assert verification.feasible

    p95 = statistics.quantiles(durations, n=20)[18]
    assert p95 <= 0.250


def test_bound_reduction_scales_linearly_with_period_coordinate_count() -> None:
    small = _representative_case(period_count=20)
    large = _representative_case(period_count=100)

    started = time.perf_counter()
    derive_effective_templates(*small)
    small_duration = time.perf_counter() - started
    started = time.perf_counter()
    derive_effective_templates(*large)
    large_duration = time.perf_counter() - started

    assert large_duration <= max(0.150, small_duration * 8.0)
