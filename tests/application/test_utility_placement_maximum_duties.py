"""Public maximum-duty utility-placement contract tests."""

from __future__ import annotations

import inspect

import pytest
from hypothesis import given
from hypothesis import strategies as st

from OpenPinch import PinchProblem, PinchWorkspace
from OpenPinch.application import utility_placement as placement_application
from OpenPinch.application._problem.accessors.target import _TargetAccessor
from OpenPinch.application._problem.input.utilities import _get_hot_and_cold_utilities
from OpenPinch.contracts.input import UtilitySchema
from OpenPinch.contracts.utility_placement import (
    QuantityValue,
    UtilityDutyLimit,
    UtilityPlacementRequest,
)
from OpenPinch.domain.configuration import Configuration


def _problem_and_zone():
    problem = PinchWorkspace(source="chocolate_factory.json").use_case("baseline")
    zone = placement_application._resolve_placement_zone(problem, "Almond")
    return problem, zone


def test_public_signature_exposes_maximum_duties_as_physical_input() -> None:
    parameters = inspect.signature(_TargetAccessor.utility_placement).parameters

    assert tuple(parameters) == (
        "self",
        "isothermal",
        "sensible",
        "zone",
        "period_ids",
        "maximum_duties",
        "options",
    )


def test_generated_maximum_duties_normalize_names_units_and_periods() -> None:
    problem, zone = _problem_and_zone()

    request = placement_application._build_problem_placement_request(
        problem,
        selected_zone=zone,
        isothermal=2,
        sensible=0,
        period_ids=("0",),
        maximum_duties={
            "hot_iso_1": {"value": 0.05, "unit": "MW"},
            "cold_iso_1": 25.0,
        },
        options=None,
    )

    limits = {limit.name: limit for limit in request.maximum_duties}
    assert tuple(limits) == ("cold_iso_1", "hot_iso_1")
    assert limits["hot_iso_1"].period_ids == ("0",)
    assert limits["hot_iso_1"].values[0].value == pytest.approx(50.0)
    assert limits["hot_iso_1"].values[0].unit == "kW"
    assert limits["cold_iso_1"].values[0].value == pytest.approx(25.0)


def test_inferred_utility_names_accept_independent_maximum_duties() -> None:
    problem, zone = _problem_and_zone()

    request = placement_application._build_problem_placement_request(
        problem,
        selected_zone=zone,
        isothermal=None,
        sensible=None,
        period_ids=("0",),
        maximum_duties={"HPS": 20.0, "CW": 0.0},
        options=None,
    )

    limits = {limit.name: limit.values[0].value for limit in request.maximum_duties}
    assert limits == {"CW": 0.0, "HPS": 20.0}


def test_default_utilities_are_not_inferred_placement_options() -> None:
    problem = PinchWorkspace(source="basic_pinch.json").use_case("baseline")
    zone = placement_application._resolve_placement_zone(problem, None)

    with pytest.raises(Exception, match="utilities other than HU/CU defaults"):
        placement_application._build_problem_placement_request(
            problem,
            selected_zone=zone,
            isothermal=None,
            sensible=None,
            period_ids=("0",),
            maximum_duties={"HU": 10.0},
            options=None,
        )


def test_period_resolved_maximum_duties_follow_selected_period_identity() -> None:
    _, zone = _problem_and_zone()

    limits = placement_application._normalize_maximum_duties(
        {
            "hot_iso_1": {
                "values": [0.1, 0.2],
                "period_ids": ["summer", "winter"],
                "unit": "MW",
            }
        },
        known_names={"hot_iso_1"},
        selected_period_ids=("winter", "summer"),
        available_period_ids=("summer", "winter"),
        config=zone.config,
        heat_flow_unit="kW",
    )

    assert limits[0].period_ids == ("winter", "summer")
    assert tuple(value.value for value in limits[0].values) == pytest.approx(
        (200.0, 100.0)
    )


def _period_resolved_request() -> UtilityPlacementRequest:
    return UtilityPlacementRequest(
        isothermal_level_count=2,
        period_ids=("winter", "summer"),
        maximum_duties=(
            UtilityDutyLimit(
                name="hot_iso_1",
                period_ids=("winter", "summer"),
                values=(
                    QuantityValue(value=200.0, unit="kW"),
                    QuantityValue(value=100.0, unit="kW"),
                ),
            ),
        ),
    )


@st.composite
def _identity_aware_maximum_heat_flows(draw):
    period_ids = draw(
        st.lists(
            st.sampled_from(("summer", "shoulder", "winter", "maintenance")),
            min_size=1,
            max_size=4,
            unique=True,
        )
    )
    values = draw(
        st.lists(
            st.floats(
                min_value=0.0,
                max_value=1.0e6,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=len(period_ids),
            max_size=len(period_ids),
        )
    )
    return {
        "values": values,
        "period_ids": period_ids,
        "unit": "kW",
    }


def test_candidate_replay_serializes_only_the_selected_period_limit() -> None:
    request = _period_resolved_request()

    assert placement_application._serialized_limit(
        request,
        "hot_iso_1",
        period_id="summer",
    ) == {"value": 100.0, "unit": "kW"}
    assert placement_application._serialized_limit(
        request,
        "hot_iso_1",
        period_id="winter",
    ) == {"value": 200.0, "unit": "kW"}


def test_equal_selected_period_limits_retain_period_identity() -> None:
    request = UtilityPlacementRequest(
        isothermal_level_count=2,
        period_ids=("winter", "summer"),
        maximum_duties=(
            UtilityDutyLimit(
                name="hot_iso_1",
                period_ids=("winter", "summer"),
                values=(
                    QuantityValue(value=100.0, unit="kW"),
                    QuantityValue(value=100.0, unit="kW"),
                ),
            ),
        ),
    )

    assert placement_application._serialized_limit(request, "hot_iso_1") == {
        "values": [100.0, 100.0],
        "period_ids": ["winter", "summer"],
        "unit": "kW",
    }


@given(
    period_ids=st.lists(
        st.sampled_from(("summer", "shoulder", "winter", "maintenance")),
        min_size=1,
        max_size=4,
        unique=True,
    ),
    value=st.floats(
        min_value=0.0,
        max_value=1.0e6,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_uniform_limits_always_serialize_with_selected_period_ids(
    period_ids,
    value: float,
) -> None:
    request = UtilityPlacementRequest(
        isothermal_level_count=2,
        period_ids=tuple(period_ids),
        maximum_duties=(
            UtilityDutyLimit(
                name="hot_iso_1",
                period_ids=tuple(period_ids),
                values=tuple(QuantityValue(value=value, unit="kW") for _ in period_ids),
            ),
        ),
    )

    serialized = placement_application._serialized_limit(request, "hot_iso_1")

    assert serialized["period_ids"] == period_ids
    assert serialized["values"] == pytest.approx([value] * len(period_ids))


def test_period_identity_limit_is_accepted_by_returned_case_schema() -> None:
    request = _period_resolved_request()
    maximum_heat_flow = placement_application._serialized_limit(
        request,
        "hot_iso_1",
    )

    utility = UtilitySchema.model_validate(
        {
            "name": "hot_iso_1",
            "type": "Hot",
            "t_supply": {"value": 200.0, "unit": "degC"},
            "t_target": {"value": 199.99, "unit": "degC"},
            "heat_flow": {"value": 0.0, "unit": "kW"},
            "maximum_heat_flow": maximum_heat_flow,
        }
    )

    assert utility.maximum_heat_flow.model_dump(mode="python") == {
        "values": [200.0, 100.0],
        "period_ids": ["winter", "summer"],
        "unit": "kW",
    }


@given(maximum_heat_flow=_identity_aware_maximum_heat_flows())
def test_identity_aware_maximum_heat_flow_schema_round_trip(
    maximum_heat_flow,
) -> None:
    utility = UtilitySchema.model_validate(
        {
            "name": "HPS",
            "type": "Hot",
            "t_supply": 200.0,
            "t_target": 199.99,
            "maximum_heat_flow": maximum_heat_flow,
        }
    )

    assert UtilitySchema.model_validate_json(utility.model_dump_json()) == utility


def test_period_identity_limit_canonicalizes_and_leaves_unselected_unbounded() -> None:
    config = Configuration(
        options={
            "PROBLEM_PERIOD_IDS": ["summer", "shoulder", "winter"],
            "PROBLEM_PERIOD_WEIGHTS": [1.0, 1.0, 1.0],
        }
    )
    utility = UtilitySchema.model_validate(
        {
            "name": "HPS",
            "type": "Hot",
            "t_supply": {"value": 200.0, "unit": "degC"},
            "t_target": {"value": 199.99, "unit": "degC"},
            "heat_flow": {"value": 0.0, "unit": "kW"},
            "maximum_heat_flow": {
                "values": [200.0, 100.0],
                "period_ids": ["winter", "summer"],
                "unit": "kW",
            },
        }
    )

    prepared = _get_hot_and_cold_utilities(
        utilities=[utility],
        hu_t_min=250.0,
        cu_t_max=10.0,
        config=config,
    )
    maximum = prepared.get_hot_utility_streams().get_stream_by_name(
        "HPS"
    ).maximum_heat_flow

    assert maximum.period_values[[0, 2]] == pytest.approx([100.0, 200.0])
    assert maximum.period_values[1] != maximum.period_values[1]


@pytest.mark.parametrize(
    ("maximum_duties", "message"),
    [
        ({"missing": 1.0}, "unknown utility"),
        ({"hot_iso_1": -1.0}, "non-negative"),
        ({"hot_iso_1": {"value": 1.0, "unit": "degC"}}, "unit"),
    ],
)
def test_maximum_duty_validation_fails_before_optimization(
    maximum_duties,
    message,
) -> None:
    problem, zone = _problem_and_zone()

    with pytest.raises(Exception, match=message):
        placement_application._build_problem_placement_request(
            problem,
            selected_zone=zone,
            isothermal=2,
            sensible=0,
            period_ids=("0",),
            maximum_duties=maximum_duties,
            options=None,
        )


def test_maximum_duties_round_trip_on_request_and_returned_case() -> None:
    problem = PinchWorkspace(source="chocolate_factory.json").use_case("baseline")

    case = problem.target.utility_placement(
        isothermal=2,
        zone="Almond",
        period_ids=("0",),
        maximum_duties={"hot_iso_1": 1.0},
        options={
            "iteration_limit": 1,
            "evaluation_limit": 100,
            "candidate_limit": 2,
            "run_count": 1,
        },
    )

    result = case.utility_placement_result
    assert result == result.model_validate_json(result.model_dump_json())
    limits = {
        utility["name"]: utility.get("maximum_heat_flow")
        for utility in case.to_problem_json()["utilities"]
    }
    assert limits["hot_iso_1"] == {
        "values": [1.0],
        "period_ids": ["0"],
        "unit": "kW",
    }
    assert any(name in {"HU", "CU"} for name in limits)

    evidence = result.best.period_results[0]
    target = case.target.direct_heat_integration(zone="Almond", period_id="0")
    almond = case.master_zone.get_subzone("Almond")
    hot = {utility.name: utility for utility in almond.hot_utilities}
    assert hot["hot_iso_1"].heat_flow.value <= 1.0 + 1e-9
    assert hot["hot_iso_1"].maximum_heat_flow.value == pytest.approx(1.0)
    assert {
        utility.name: float(utility.heat_flow.value)
        for utility in target.hot_utilities
    } == pytest.approx(
        {
            level.template_key.name: level.allocated_duty.value
            for level in evidence.hot_levels
        },
        abs=1e-6,
    )


def test_unequal_multiperiod_caps_complete_and_retarget_by_identity() -> None:
    baseline = PinchWorkspace(source="chocolate_factory.json").use_case("baseline")
    source = baseline.to_problem_json()
    source["options"]["PROBLEM_PERIOD_IDS"] = ["summer", "winter"]
    source["options"]["PROBLEM_PERIOD_WEIGHTS"] = [1.0, 1.0]
    problem = PinchProblem(source=source, project_name="Site")

    case = problem.target.utility_placement(
        isothermal=2,
        zone="Almond",
        period_ids=("winter", "summer"),
        maximum_duties={
            "hot_iso_1": {
                "values": [2.0, 1.0],
                "period_ids": ["winter", "summer"],
                "unit": "kW",
            }
        },
        options={
            "iteration_limit": 1,
            "evaluation_limit": 100,
            "candidate_limit": 2,
            "run_count": 1,
        },
    )

    serialized = {
        utility["name"]: utility.get("maximum_heat_flow")
        for utility in case.to_problem_json()["utilities"]
    }
    assert serialized["hot_iso_1"] == {
        "values": [2.0, 1.0],
        "period_ids": ["winter", "summer"],
        "unit": "kW",
    }
    optimized = case.master_zone.get_subzone("Almond")
    maximum = optimized.hot_utilities.get_stream_by_name(
        "hot_iso_1"
    ).maximum_heat_flow
    assert maximum.period_values == pytest.approx([1.0, 2.0])

    for period_id, expected_limit in (("summer", 1.0), ("winter", 2.0)):
        target = case.target.direct_heat_integration(
            zone="Almond",
            period_id=period_id,
        )
        period_idx = case.period_ids[period_id]
        hot = {utility.name: utility for utility in target.hot_utilities}
        assert hot["hot_iso_1"].heat_flow[period_idx] <= expected_limit + 1e-9
