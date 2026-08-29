"""Public maximum-duty utility-placement contract tests."""

from __future__ import annotations

import inspect

import pytest

from OpenPinch import PinchWorkspace
from OpenPinch.application import utility_placement as placement_application
from OpenPinch.application._problem.accessors.target import _TargetAccessor


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
    assert limits["hot_iso_1"] == {"value": 1.0, "unit": "kW"}
    assert any(name in {"HU", "CU"} for name in limits)

    case.target.direct_heat_integration(zone="Almond", period_id="0")
    almond = case.master_zone.get_subzone("Almond")
    hot = {utility.name: utility for utility in almond.hot_utilities}
    assert hot["hot_iso_1"].heat_flow.value <= 1.0 + 1e-9
    assert hot["hot_iso_1"].maximum_heat_flow.value == pytest.approx(1.0)
    assert hot["HU"].heat_flow.value > 0.0
