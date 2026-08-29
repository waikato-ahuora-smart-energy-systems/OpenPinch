"""Property tests for utility-placement pure transformations."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from OpenPinch.analysis.utility_placement import (
    build_initial_values,
    decode_placement,
    derive_effective_templates,
    encode_placement,
    normalize_utility_placement_request,
    prepare_template_blueprints,
    verify_candidate,
)
from OpenPinch.analysis.utility_placement.errors import (
    PlacementModelValidationError,
)
from OpenPinch.analysis.utility_placement.normalization import convert_placement_value
from OpenPinch.contracts.common import ValueWithUnit
from OpenPinch.contracts.utility_placement import (
    CoordinateKey,
    DecisionField,
    PhysicalCoordinateBound,
    PlacementFeasibilityEnvelope,
    PlacementPeriodEnvelope,
    QuantityInterval,
    QuantityValue,
    UtilityLevelKind,
    UtilityPlacementRequest,
    UtilityPlacementResult,
    UtilitySide,
)
from OpenPinch.domain.value import Value
from tests.analysis.utility_placement.test_codec import _model
from tests.analysis.utility_placement.test_contracts import _nested_result
from tests.strategies.utility_placement import count_only_requests


def _explicit_request_and_blueprints():
    generated_request = normalize_utility_placement_request(isothermal_level_count=2)
    generated_blueprints = prepare_template_blueprints(generated_request)
    request = generated_request.model_copy(
        update={
            "hot_templates": tuple(
                item.as_template() for item in generated_blueprints.hot
            ),
            "cold_templates": tuple(
                item.as_template() for item in generated_blueprints.cold
            ),
        }
    )
    return request, prepare_template_blueprints(request)


@given(count_only_requests())
def test_request_normalization_is_idempotent(request) -> None:
    once = normalize_utility_placement_request(request)
    twice = normalize_utility_placement_request(once)

    assert twice == once


@given(count_only_requests())
def test_request_json_round_trip_is_structurally_exact(request) -> None:
    restored = UtilityPlacementRequest.model_validate_json(request.model_dump_json())

    assert restored == request


@given(count_only_requests())
def test_generated_blueprint_inventory_is_symmetric_and_unique(request) -> None:
    blueprints = prepare_template_blueprints(request)

    assert len(blueprints.all) == (
        2 * request.isothermal_level_count + 2 * request.sensible_level_count
    )
    assert len({item.key.name for item in blueprints.all}) == len(blueprints.all)
    for side, items in (
        (UtilitySide.HOT, blueprints.hot),
        (UtilitySide.COLD, blueprints.cold),
    ):
        assert all(item.key.side is side for item in items)
        assert sum(item.kind is UtilityLevelKind.ISOTHERMAL for item in items) == (
            request.isothermal_level_count
        )
        assert sum(item.kind is UtilityLevelKind.SENSIBLE for item in items) == (
            request.sensible_level_count
        )
    assert all(
        item.fixed_span is None
        or (item.fixed_span.unit == "delta_degC" and item.fixed_span.value > 0.0)
        for item in blueprints.all
    )


@given(count_only_requests())
def test_blueprint_normalization_is_idempotent(request) -> None:
    once = prepare_template_blueprints(request)
    normalized_request = request.model_copy(
        update={
            "hot_templates": tuple(item.as_template() for item in once.hot),
            "cold_templates": tuple(item.as_template() for item in once.cold),
        }
    )
    twice = prepare_template_blueprints(normalized_request)

    assert twice == once


@given(st.permutations(("a", "b", "c")))
def test_envelope_intersection_is_commutative_under_period_permutation(order) -> None:
    request, blueprints = _explicit_request_and_blueprints()
    period_limits = {"a": (-50.0, 500.0), "b": (-25.0, 450.0), "c": (0.0, 400.0)}
    periods = tuple(
        PlacementPeriodEnvelope(
            period_id=period_id,
            weight=1.0,
            coordinate_bounds=tuple(
                PhysicalCoordinateBound(
                    coordinate=CoordinateKey(
                        template_key=blueprint.key,
                        field=DecisionField.SUPPLY_TEMPERATURE,
                    ),
                    bounds=QuantityInterval(
                        lower=period_limits[period_id][0],
                        upper=period_limits[period_id][1],
                        unit="degC",
                    ),
                    reason="property fixture",
                )
                for blueprint in blueprints.all
            ),
        )
        for period_id in order
    )
    envelope = PlacementFeasibilityEnvelope(
        periods=periods,
        minimum_separation=QuantityValue(value=1.0, unit="delta_degC"),
        scope="direct",
        base_target_id="property",
    )

    effective = derive_effective_templates(request, blueprints, envelope)

    reference = derive_effective_templates(
        request,
        blueprints,
        envelope.model_copy(
            update={"periods": tuple(sorted(periods, key=lambda item: item.period_id))}
        ),
    )
    assert [item.supply_bounds for item in effective.all] == [
        item.supply_bounds for item in reference.all
    ]


@given(
    st.lists(
        st.tuples(
            st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            st.floats(
                min_value=50.0,
                max_value=100.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        ),
        min_size=1,
        max_size=6,
    )
)
def test_envelope_intersection_matches_explicit_max_min_oracle(period_limits) -> None:
    request, blueprints = _explicit_request_and_blueprints()
    base_by_name = {
        "hot_iso_1": 300.0,
        "hot_iso_2": 100.0,
        "cold_iso_1": 0.0,
        "cold_iso_2": 200.0,
    }
    periods = tuple(
        PlacementPeriodEnvelope(
            period_id=f"period-{period_index}",
            weight=1.0,
            coordinate_bounds=tuple(
                PhysicalCoordinateBound(
                    coordinate=CoordinateKey(
                        template_key=blueprint.key,
                        field=DecisionField.SUPPLY_TEMPERATURE,
                    ),
                    bounds=QuantityInterval(
                        lower=base_by_name[blueprint.key.name] + lower_offset,
                        upper=(base_by_name[blueprint.key.name] + lower_offset + width),
                        unit="degC",
                    ),
                    reason="oracle property fixture",
                )
                for blueprint in blueprints.all
            ),
        )
        for period_index, (lower_offset, width) in enumerate(period_limits)
    )
    envelope = PlacementFeasibilityEnvelope(
        periods=periods,
        minimum_separation=QuantityValue(value=1.0, unit="delta_degC"),
        scope="direct",
        base_target_id="oracle",
    )

    effective = derive_effective_templates(request, blueprints, envelope)

    expected_lower_offset = max(lower for lower, _ in period_limits)
    expected_upper_offset = min(lower + width for lower, width in period_limits)
    for template in effective.all:
        base = base_by_name[template.key.name]
        assert (
            abs(template.supply_bounds.lower - (base + expected_lower_offset))
            <= request.tolerances.absolute
        )
        assert (
            abs(template.supply_bounds.upper - (base + expected_upper_offset))
            <= request.tolerances.absolute
        )


@given(
    st.floats(
        min_value=-100.0,
        max_value=500.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    st.sampled_from(("degC", "degF", "K")),
)
def test_unit_conversion_matches_value_to_oracle(magnitude, source_unit) -> None:
    actual = convert_placement_value(
        ValueWithUnit(value=magnitude, unit=source_unit),
        canonical_unit="degC",
        default_unit="degC",
        field_path="temperature",
    )
    expected = float(Value(magnitude, source_unit).to("degC"))

    assert actual == expected


@given(st.floats(min_value=0.01, max_value=50.0, allow_nan=False))
def test_order_propagation_and_start_satisfy_separation(separation) -> None:
    request, blueprints = _explicit_request_and_blueprints()
    coordinate_bounds = tuple(
        PhysicalCoordinateBound(
            coordinate=CoordinateKey(
                template_key=blueprint.key,
                field=DecisionField.SUPPLY_TEMPERATURE,
            ),
            bounds=QuantityInterval(lower=0.0, upper=500.0, unit="degC"),
            reason="property fixture",
        )
        for blueprint in blueprints.all
    )
    envelope = PlacementFeasibilityEnvelope(
        periods=(
            PlacementPeriodEnvelope(
                period_id="only",
                weight=1.0,
                coordinate_bounds=coordinate_bounds,
            ),
        ),
        minimum_separation=QuantityValue(
            value=separation,
            unit="delta_degC",
        ),
        scope="direct",
        base_target_id="property",
    )
    templates = derive_effective_templates(request, blueprints, envelope)
    values = build_initial_values(templates, separation)

    for side, items in (
        (UtilitySide.HOT, templates.hot),
        (UtilitySide.COLD, templates.cold),
    ):
        supplies = [
            values[
                CoordinateKey(
                    template_key=item.key,
                    field=DecisionField.SUPPLY_TEMPERATURE,
                )
            ]
            for item in items
        ]
        if side is UtilitySide.HOT:
            assert supplies[0] >= supplies[1] + separation - 1e-6
        else:
            assert supplies[1] >= supplies[0] + separation - 1e-6


@given(count_only_requests())
def test_vector_codec_round_trips_across_valid_dimensions(request) -> None:
    isothermal_count = request.isothermal_level_count
    sensible_count = request.sensible_level_count
    model = _model(
        isothermal_count=isothermal_count,
        sensible_count=sensible_count,
    )
    point = model.initial_points[0]
    placement = decode_placement(model, point)

    assert len(point) == isothermal_count + 2 * sensible_count
    encoded = encode_placement(model, placement)
    assert encoded == pytest.approx(point)
    restored = decode_placement(model, encoded)
    assert restored.hot == placement.hot
    assert restored.cold == placement.cold
    assert verify_candidate(model, point).feasible
    for hot, cold in zip(placement.hot, placement.cold, strict=True):
        assert cold.supply_temperature.value == hot.target_temperature.value
        assert cold.target_temperature.value == hot.supply_temperature.value
        assert cold.temperature_span == hot.temperature_span


@given(st.text(min_size=1), st.text(min_size=1))
def test_error_context_always_has_stable_code_and_message(code, message) -> None:
    normalized_code = code.strip() or "fallback_code"
    normalized_message = message.strip() or "fallback message"
    error = PlacementModelValidationError(
        code=normalized_code,
        message=normalized_message,
        field_path="model",
        period_id="period-1",
        details=(("reason", "property"),),
    )

    assert error.context["code"] == normalized_code
    assert error.context["message"] == normalized_message
    assert error.context["field_path"] == "model"
    assert error.context["period_id"] == "period-1"


@given(st.data())
def test_nested_result_json_round_trip_property(data) -> None:
    result = _nested_result()
    objective_value = data.draw(
        st.floats(
            min_value=0.0,
            max_value=1e9,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    best = result.best.model_copy(
        update={
            "aggregate_objective": result.best.aggregate_objective.model_copy(
                update={"value": objective_value}
            )
        }
    )
    varied = result.model_copy(update={"best": best})

    restored = UtilityPlacementResult.model_validate_json(varied.model_dump_json())

    assert restored == varied
