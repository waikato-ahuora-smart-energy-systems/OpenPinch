"""Examples for the stable decision schema, codec, and verifier."""

from __future__ import annotations

import math

import pytest

from OpenPinch.analysis.utility_placement import (
    build_utility_placement_model,
    decode_placement,
    encode_placement,
    normalize_utility_placement_request,
    prepare_template_blueprints,
    verify_candidate,
)
from OpenPinch.analysis.utility_placement.errors import PlacementModelValidationError
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


def _model(*, isothermal_count: int = 2, sensible_count: int = 0):
    request = normalize_utility_placement_request(
        isothermal_level_count=isothermal_count,
        sensible_level_count=sensible_count,
    )
    blueprints = prepare_template_blueprints(request)
    coordinate_bounds = []
    for blueprint in blueprints.all:
        coordinate_bounds.append(
            PhysicalCoordinateBound(
                coordinate=CoordinateKey(
                    template_key=blueprint.key,
                    field=DecisionField.SUPPLY_TEMPERATURE,
                ),
                bounds=QuantityInterval(lower=0.0, upper=500.0, unit="degC"),
                reason="codec fixture",
            )
        )
        if blueprint.kind is UtilityLevelKind.SENSIBLE:
            coordinate_bounds.append(
                PhysicalCoordinateBound(
                    coordinate=CoordinateKey(
                        template_key=blueprint.key,
                        field=DecisionField.TEMPERATURE_SPAN,
                    ),
                    bounds=QuantityInterval(
                        lower=1.0,
                        upper=50.0,
                        unit="delta_degC",
                    ),
                    reason="codec fixture",
                )
            )
    envelope = PlacementFeasibilityEnvelope(
        periods=(
            PlacementPeriodEnvelope(
                period_id="only",
                weight=1.0,
                coordinate_bounds=tuple(coordinate_bounds),
            ),
        ),
        minimum_separation=QuantityValue(value=10.0, unit="delta_degC"),
        scope="direct",
        base_target_id="codec",
    )
    return build_utility_placement_model(request, blueprints, envelope)


@pytest.mark.parametrize("isothermal_count,sensible_count", [(2, 0), (2, 1), (4, 3)])
def test_schema_dimension_and_coordinate_uniqueness(
    isothermal_count: int,
    sensible_count: int,
) -> None:
    model = _model(
        isothermal_count=isothermal_count,
        sensible_count=sensible_count,
    )

    assert len(model.coordinates) == 2 * isothermal_count + 4 * sensible_count
    assert [coordinate.index for coordinate in model.coordinates] == list(
        range(len(model.coordinates))
    )
    assert len({coordinate.coordinate for coordinate in model.coordinates}) == len(
        model.coordinates
    )


def test_coordinate_family_order_is_stable() -> None:
    model = _model(isothermal_count=2, sensible_count=1)

    observed = [
        (
            coordinate.coordinate.template_key.side.value,
            coordinate.coordinate.template_key.name,
            coordinate.coordinate.field.value,
        )
        for coordinate in model.coordinates
    ]
    assert observed == [
        ("hot", "hot_iso_1", "supply_temperature"),
        ("hot", "hot_iso_2", "supply_temperature"),
        ("hot", "hot_sensible_1", "supply_temperature"),
        ("hot", "hot_sensible_1", "temperature_span"),
        ("cold", "cold_iso_1", "supply_temperature"),
        ("cold", "cold_iso_2", "supply_temperature"),
        ("cold", "cold_sensible_1", "supply_temperature"),
        ("cold", "cold_sensible_1", "temperature_span"),
    ]


def test_both_codec_round_trips_preserve_identity_order_and_values() -> None:
    model = _model(isothermal_count=2, sensible_count=1)
    point = model.initial_points[0]

    placement = decode_placement(model, point)
    restored_point = encode_placement(model, placement)
    restored_placement = decode_placement(model, restored_point)

    assert restored_point == point
    assert restored_placement == placement
    assert [level.template_key for level in placement.hot] == [
        item.key for item in model.templates.hot
    ]
    assert [level.template_key for level in placement.cold] == [
        item.key for item in model.templates.cold
    ]


def test_decoding_derives_hot_and_cold_target_directions() -> None:
    model = _model(isothermal_count=2, sensible_count=1)
    placement = decode_placement(model, model.initial_points[0])

    assert all(
        level.target_temperature.value
        == pytest.approx(level.supply_temperature.value - level.temperature_span.value)
        for level in placement.hot
    )
    assert all(
        level.target_temperature.value
        == pytest.approx(level.supply_temperature.value + level.temperature_span.value)
        for level in placement.cold
    )


def test_invalid_points_return_diagnostics_and_strict_decode_rejects() -> None:
    model = _model()
    wrong_length = model.initial_points[0][:-1]
    non_finite = (math.nan, *model.initial_points[0][1:])
    out_of_bounds = (
        model.coordinates[0].bounds.upper + 1.0,
        *model.initial_points[0][1:],
    )

    for point, code in (
        (wrong_length, "dimension_mismatch"),
        (non_finite, "non_finite_coordinate"),
        (out_of_bounds, "coordinate_out_of_bounds"),
    ):
        verification = verify_candidate(model, point)
        assert not verification.feasible
        assert verification.diagnostics[0].code == code
        with pytest.raises(PlacementModelValidationError):
            decode_placement(model, point)


def test_every_generated_start_passes_independent_candidate_verifier() -> None:
    model = _model(isothermal_count=3, sensible_count=2)

    assert len(model.initial_points) > 1
    assert all(
        verify_candidate(model, point).feasible for point in model.initial_points
    )


def test_fixed_coordinates_remain_in_vector_schema() -> None:
    model = _model()
    coordinate = model.coordinates[0]
    fixed_coordinate = coordinate.model_copy(
        update={
            "bounds": QuantityInterval(
                lower=coordinate.bounds.upper,
                upper=coordinate.bounds.upper,
                unit=coordinate.bounds.unit,
            )
        }
    )
    fixed_model = model.model_copy(
        update={
            "coordinates": (fixed_coordinate, *model.coordinates[1:]),
            "initial_points": (
                (fixed_coordinate.bounds.lower, *model.initial_points[0][1:]),
            ),
        }
    )

    assert len(fixed_model.coordinates) == len(model.coordinates)
    assert verify_candidate(fixed_model, fixed_model.initial_points[0]).feasible
