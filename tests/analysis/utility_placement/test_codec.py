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


def _model(
    *,
    isothermal_count: int = 2,
    sensible_count: int = 0,
    generated_pairs: bool = True,
):
    request = normalize_utility_placement_request(
        isothermal_level_count=isothermal_count,
        sensible_level_count=sensible_count,
    )
    blueprints = prepare_template_blueprints(request)
    if not generated_pairs:
        request = request.model_copy(
            update={
                "hot_templates": tuple(item.as_template() for item in blueprints.hot),
                "cold_templates": tuple(item.as_template() for item in blueprints.cold),
            }
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
                residual_hot_duty=100.0,
                residual_cold_duty=80.0,
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

    level_count = isothermal_count + sensible_count
    temperature_dimension = isothermal_count + 2 * sensible_count
    dispatch_dimension = 2 * (level_count - 1)
    assert len(model.coordinates) == temperature_dimension + dispatch_dimension
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
        ("hot", "hot_iso_1", "duty_fraction"),
        ("hot", "hot_iso_2", "duty_fraction"),
        ("cold", "cold_iso_1", "duty_fraction"),
        ("cold", "cold_iso_2", "duty_fraction"),
    ]


def test_explicit_hot_and_cold_templates_keep_independent_schema() -> None:
    model = _model(
        isothermal_count=2,
        sensible_count=1,
        generated_pairs=False,
    )

    assert len(model.coordinates) == 2 * 2 + 4 * 1 + 2 * (3 - 1)
    assert {item.coordinate.template_key.side.value for item in model.coordinates} == {
        "hot",
        "cold",
    }


def test_both_codec_round_trips_preserve_identity_order_and_values() -> None:
    model = _model(isothermal_count=2, sensible_count=1)
    point = model.initial_points[0]

    placement = decode_placement(model, point)
    restored_point = encode_placement(model, placement)
    restored_placement = decode_placement(model, restored_point)

    assert restored_point == pytest.approx(point)
    assert restored_placement.hot == placement.hot
    assert restored_placement.cold == placement.cold
    for restored, original in zip(
        restored_placement.period_dispatches,
        placement.period_dispatches,
        strict=True,
    ):
        assert tuple(item.value for item in restored.hot_duties) == pytest.approx(
            tuple(item.value for item in original.hot_duties)
        )
        assert tuple(item.value for item in restored.cold_duties) == pytest.approx(
            tuple(item.value for item in original.cold_duties)
        )
    assert [level.template_key for level in placement.hot] == [
        item.key for item in model.templates.hot
    ]
    assert [level.template_key for level in placement.cold] == [
        item.key for item in model.templates.cold
    ]
    dispatch = placement.period_dispatches[0]
    assert sum(item.value for item in dispatch.hot_duties) == pytest.approx(
        model.envelope.periods[0].residual_hot_duty
    )
    assert sum(item.value for item in dispatch.cold_duties) == pytest.approx(
        model.envelope.periods[0].residual_cold_duty
    )
    assert all(item.value >= 0.0 for item in dispatch.hot_duties)
    assert all(item.value >= 0.0 for item in dispatch.cold_duties)


def test_encoding_rejects_a_generated_cold_endpoint_that_is_not_reversed() -> None:
    model = _model(isothermal_count=2, sensible_count=1)
    placement = decode_placement(model, model.initial_points[0])
    cold = placement.cold[0]
    invalid_cold = cold.model_copy(
        update={
            "supply_temperature": cold.supply_temperature.model_copy(
                update={"value": cold.supply_temperature.value + 1.0}
            )
        }
    )
    invalid = placement.model_copy(update={"cold": (invalid_cold, *placement.cold[1:])})

    with pytest.raises(PlacementModelValidationError) as captured:
        encode_placement(model, invalid)

    assert captured.value.code == "paired_endpoint_mismatch"


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
    for hot, cold in zip(placement.hot, placement.cold, strict=True):
        assert cold.kind is hot.kind
        assert cold.supply_temperature.value == pytest.approx(
            hot.target_temperature.value
        )
        assert cold.target_temperature.value == pytest.approx(
            hot.supply_temperature.value
        )
        assert cold.temperature_span == hot.temperature_span


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

    assert 1 < len(model.initial_points) <= 28
    supply_coordinates = [
        coordinate
        for coordinate in model.coordinates
        if coordinate.coordinate.field is DecisionField.SUPPLY_TEMPERATURE
    ]
    assert (
        1
        < len(
            {
                tuple(point[coordinate.index] for coordinate in supply_coordinates)
                for point in model.initial_points
            }
        )
        <= 8
    )
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
