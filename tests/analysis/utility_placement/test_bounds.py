"""Examples for physical-bound reduction, ordering, and starts."""

from __future__ import annotations

import pytest

from OpenPinch.analysis.utility_placement import (
    build_initial_values,
    derive_effective_templates,
    normalize_utility_placement_request,
    prepare_template_blueprints,
)
from OpenPinch.analysis.utility_placement.errors import (
    EmptyPlacementFeasibleRegionError,
    UtilityTemplateValidationError,
)
from OpenPinch.contracts.utility_placement import (
    CoordinateKey,
    DecisionField,
    PhysicalCoordinateBound,
    PlacementFeasibilityEnvelope,
    PlacementPeriodEnvelope,
    QuantityInterval,
    QuantityValue,
)


def _request_and_blueprints(*, fixed_high: bool = False):
    request = normalize_utility_placement_request(isothermal_level_count=2)
    blueprints = prepare_template_blueprints(request)
    if fixed_high:
        first = blueprints.hot[0].model_copy(
            update={
                "supply_bounds": QuantityInterval(
                    lower=210.0,
                    upper=210.0,
                    unit="degC",
                )
            }
        )
        blueprints = blueprints.model_copy(update={"hot": (first, *blueprints.hot[1:])})
    return request, blueprints


def _envelope(blueprints, *, reverse_periods: bool = False):
    per_key = {
        ("hot", "hot_iso_1"): ((150.0, 250.0), (180.0, 240.0)),
        ("hot", "hot_iso_2"): ((50.0, 170.0), (70.0, 160.0)),
        ("cold", "cold_iso_1"): ((-10.0, 40.0), (0.0, 35.0)),
        ("cold", "cold_iso_2"): ((30.0, 100.0), (40.0, 90.0)),
    }
    periods = []
    for period_index, period_id in enumerate(("summer", "winter")):
        coordinate_bounds = tuple(
            PhysicalCoordinateBound(
                coordinate=CoordinateKey(
                    template_key=blueprint.key,
                    field=DecisionField.SUPPLY_TEMPERATURE,
                ),
                bounds=QuantityInterval(
                    lower=per_key[(blueprint.key.side.value, blueprint.key.name)][
                        period_index
                    ][0],
                    upper=per_key[(blueprint.key.side.value, blueprint.key.name)][
                        period_index
                    ][1],
                    unit="degC",
                ),
                reason="test fixture",
            )
            for blueprint in blueprints.all
        )
        periods.append(
            PlacementPeriodEnvelope(
                period_id=period_id,
                weight=0.5,
                coordinate_bounds=coordinate_bounds,
            )
        )
    if reverse_periods:
        periods.reverse()
    return PlacementFeasibilityEnvelope(
        periods=tuple(periods),
        minimum_separation=QuantityValue(value=10.0, unit="delta_degC"),
        scope="direct",
        base_target_id="fixture",
    )


def test_period_bounds_use_max_lower_and_min_upper_oracle() -> None:
    request, blueprints = _request_and_blueprints()

    templates = derive_effective_templates(
        request,
        blueprints,
        _envelope(blueprints),
    )

    assert templates.hot[0].supply_bounds == QuantityInterval(
        lower=180.0,
        upper=240.0,
        unit="degC",
    )
    assert templates.hot[1].supply_bounds == QuantityInterval(
        lower=70.0,
        upper=160.0,
        unit="degC",
    )
    assert templates.cold[0].supply_bounds == QuantityInterval(
        lower=0.0,
        upper=35.0,
        unit="degC",
    )
    assert templates.cold[1].supply_bounds == QuantityInterval(
        lower=40.0,
        upper=90.0,
        unit="degC",
    )


def test_period_permutation_preserves_effective_bounds_and_source_order() -> None:
    request, blueprints = _request_and_blueprints()
    forward = _envelope(blueprints)
    reverse = _envelope(blueprints, reverse_periods=True)

    forward_templates = derive_effective_templates(request, blueprints, forward)
    reverse_templates = derive_effective_templates(request, blueprints, reverse)

    assert [item.supply_bounds for item in forward_templates.all] == [
        item.supply_bounds for item in reverse_templates.all
    ]
    assert [period.period_id for period in forward.periods] == ["summer", "winter"]
    assert [period.period_id for period in reverse.periods] == ["winter", "summer"]


def test_caller_bounds_can_narrow_or_fix_but_cannot_expand() -> None:
    request, fixed_blueprints = _request_and_blueprints(fixed_high=True)
    templates = derive_effective_templates(
        request,
        fixed_blueprints,
        _envelope(fixed_blueprints),
    )

    assert templates.hot[0].supply_bounds.lower == 210.0
    assert templates.hot[0].supply_bounds.upper == 210.0

    expanded = fixed_blueprints.hot[0].model_copy(
        update={
            "supply_bounds": QuantityInterval(
                lower=100.0,
                upper=210.0,
                unit="degC",
            )
        }
    )
    invalid_blueprints = fixed_blueprints.model_copy(
        update={"hot": (expanded, *fixed_blueprints.hot[1:])}
    )
    with pytest.raises(UtilityTemplateValidationError) as captured:
        derive_effective_templates(
            request,
            invalid_blueprints,
            _envelope(invalid_blueprints),
        )
    assert captured.value.code == "caller_bounds_expand_physical_region"


def test_ordering_contradiction_raises_empty_region_error() -> None:
    request, blueprints = _request_and_blueprints()
    envelope = _envelope(blueprints).model_copy(
        update={
            "minimum_separation": QuantityValue(
                value=200.0,
                unit="delta_degC",
            )
        }
    )

    with pytest.raises(EmptyPlacementFeasibleRegionError) as captured:
        derive_effective_templates(request, blueprints, envelope)

    assert captured.value.code == "empty_ordered_bounds"


def test_primary_start_is_deterministic_and_locally_feasible() -> None:
    request, blueprints = _request_and_blueprints()
    envelope = _envelope(blueprints)
    templates = derive_effective_templates(request, blueprints, envelope)

    first = build_initial_values(templates, envelope.minimum_separation.value)
    second = build_initial_values(templates, envelope.minimum_separation.value)

    assert first == second
    hot_supplies = [
        first[
            CoordinateKey(
                template_key=template.key,
                field=DecisionField.SUPPLY_TEMPERATURE,
            )
        ]
        for template in templates.hot
    ]
    cold_supplies = [
        first[
            CoordinateKey(
                template_key=template.key,
                field=DecisionField.SUPPLY_TEMPERATURE,
            )
        ]
        for template in templates.cold
    ]
    assert hot_supplies[-1] == templates.hot[-1].supply_bounds.lower
    assert cold_supplies[-1] == templates.cold[-1].supply_bounds.upper
    for blueprint in templates.all:
        supply_key = CoordinateKey(
            template_key=blueprint.key,
            field=DecisionField.SUPPLY_TEMPERATURE,
        )
        value = first[supply_key]
        assert blueprint.supply_bounds.lower <= value <= blueprint.supply_bounds.upper
    assert (
        first[
            CoordinateKey(
                template_key=templates.hot[0].key,
                field=DecisionField.SUPPLY_TEMPERATURE,
            )
        ]
        >= first[
            CoordinateKey(
                template_key=templates.hot[1].key,
                field=DecisionField.SUPPLY_TEMPERATURE,
            )
        ]
        + 10.0
    )
    assert (
        first[
            CoordinateKey(
                template_key=templates.cold[1].key,
                field=DecisionField.SUPPLY_TEMPERATURE,
            )
        ]
        >= first[
            CoordinateKey(
                template_key=templates.cold[0].key,
                field=DecisionField.SUPPLY_TEMPERATURE,
            )
        ]
        + 10.0
    )


def test_envelope_requires_exact_coordinate_coverage() -> None:
    request, blueprints = _request_and_blueprints()
    envelope = _envelope(blueprints)
    first_period = envelope.periods[0].model_copy(
        update={"coordinate_bounds": envelope.periods[0].coordinate_bounds[:-1]}
    )
    invalid = envelope.model_copy(
        update={"periods": (first_period, *envelope.periods[1:])}
    )

    with pytest.raises(Exception, match="coordinate"):
        derive_effective_templates(request, blueprints, invalid)
