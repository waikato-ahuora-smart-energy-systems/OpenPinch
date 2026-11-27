"""Designing a network for a problem with only hot or only cold streams.

Such a network has heaters (or coolers) alone. Each utility level takes the
duty targeting gives it, and the level split follows targeting: the least
valuable level serves the composite curve from the pinch end first.
"""

from collections import defaultdict

import pytest

from OpenPinch.analysis.heat_exchanger_networks.reporting.verification import (
    verify_network_feasibility,
)
from OpenPinch.application.problem import PinchProblem
from OpenPinch.domain.enums import HeatExchangerKind
from tests.support.paths import REPOSITORY_ROOT

ZONAL_SITE = REPOSITORY_ROOT / "OpenPinch/tutorials/sample_cases/zonal_site.json"


def _quantity(value, unit):
    if isinstance(value, list):
        return {"values": value, "unit": unit}
    return {"value": value, "unit": unit}


def _stream(name, t_supply, t_target, heat_flow):
    return {
        "zone": "Plant",
        "name": name,
        "t_supply": _quantity(t_supply, "degC"),
        "t_target": _quantity(t_target, "degC"),
        "heat_flow": _quantity(heat_flow, "kW"),
        "dt_cont": _quantity(5.0, "delta_degC"),
        "htc": _quantity(1.0, "kW/m^2/delta_degC"),
    }


def _utility(name, kind, t_supply, t_target, price):
    return {
        "name": name,
        "type": kind,
        "t_supply": _quantity(t_supply, "degC"),
        "t_target": _quantity(t_target, "degC"),
        "price": _quantity(price, "$/MWh"),
        "dt_cont": _quantity(2.5, "delta_degC"),
        "htc": _quantity(1.0, "kW/m^2/delta_degC"),
    }


UTILITIES = [
    _utility("MPS", "Both", 180.0, 179.9, 30.0),
    _utility("LPS", "Both", 140.0, 139.9, 20.0),
    _utility("HW", "Both", 90.0, 70.0, 10.0),
    _utility("CW", "Cold", 20.0, 25.0, 5.0),
]


def _problem(streams, options=None):
    return PinchProblem(
        {
            "streams": streams,
            "utilities": UTILITIES,
            "options": options or {},
        }
    )


def _cold_only(heat_flows=(110.0, 120.0), options=None):
    first, second = heat_flows
    return _problem(
        [
            _stream("C1", 40.0, 150.0, first),
            _stream("C2", 60.0, 120.0, second),
        ],
        options,
    )


def _hot_only():
    return _problem(
        [
            _stream("H1", 200.0, 60.0, 140.0),
            _stream("H2", 150.0, 40.0, 220.0),
        ]
    )


def _targets(problem, period_id=None):
    """Each utility's targeted duty (kW) in one period, by utility name."""
    if period_id is None:
        result = problem.target.direct_heat_integration()
    else:
        result = problem.target.all_periods.direct_heat_integration()[period_id]
        result = result.targets[0]
    return {
        utility.name: float(utility.heat_flow)
        for utility in (*result.hot_utilities, *result.cold_utilities)
        if float(utility.heat_flow) > 1e-6
    }


def _design_duties(network, period_id=None):
    """Each utility's duty (kW) in the network, by utility name."""
    duties = defaultdict(float)
    for exchanger in network.exchangers:
        if exchanger.kind is HeatExchangerKind.RECOVERY:
            continue
        utility = (
            exchanger.source_stream
            if exchanger.kind is HeatExchangerKind.HOT_UTILITY
            else exchanger.sink_stream
        )
        duties[utility.split(".", 1)[1]] += exchanger.state(period_id).duty
    return {name: duty for name, duty in duties.items() if duty > 1e-6}


def _assert_same_duties(actual, expected):
    assert set(actual) == set(expected)
    for name, duty in expected.items():
        assert actual[name] == pytest.approx(duty, rel=1e-6, abs=1e-6)


def test_a_cold_only_problem_is_heated_at_the_targeted_levels():
    problem = _cold_only()
    expected = _targets(problem)

    network = problem.design.heat_exchanger_network(width=0, depth=0).selected_network

    assert {exchanger.kind for exchanger in network.exchangers} == {
        HeatExchangerKind.HOT_UTILITY
    }
    assert len(expected) == 3
    _assert_same_duties(_design_duties(network), expected)


def test_a_hot_only_problem_is_cooled_at_the_targeted_levels():
    problem = _hot_only()
    expected = _targets(problem)

    network = problem.design.heat_exchanger_network(width=0, depth=0).selected_network

    assert {exchanger.kind for exchanger in network.exchangers} == {
        HeatExchangerKind.COLD_UTILITY
    }
    assert len(expected) >= 2
    _assert_same_duties(_design_duties(network), expected)


def test_a_stream_spanning_levels_has_heaters_in_series_coldest_first():
    network = (
        _cold_only().design.heat_exchanger_network(width=0, depth=0).selected_network
    )

    heaters = sorted(
        (
            exchanger.state()
            for exchanger in network.exchangers
            if exchanger.sink_stream == "Plant.C1"
        ),
        key=lambda state: state.sink_inlet_temperature,
    )
    assert len(heaters) >= 2
    assert heaters[0].sink_inlet_temperature == pytest.approx(313.15)
    assert heaters[-1].sink_outlet_temperature == pytest.approx(423.15)
    for lower, upper in zip(heaters, heaters[1:]):
        assert lower.sink_outlet_temperature == pytest.approx(
            upper.sink_inlet_temperature
        )


@pytest.mark.parametrize("problem", [_cold_only, _hot_only])
def test_a_utility_only_network_passes_verification_and_costs_add_up(problem):
    network = problem().design.heat_exchanger_network(width=0, depth=0).selected_network

    assert verify_network_feasibility(network) == ()
    assert network.total_annual_cost == pytest.approx(
        network.utility_cost + network.capital_cost, rel=1e-9
    )
    assert network.capital_cost > 0.0


def test_utility_cost_is_each_levels_duty_at_its_price():
    problem = _cold_only()
    network = problem.design.heat_exchanger_network(width=0, depth=0).selected_network
    prices = {"MPS": 30.0, "LPS": 20.0, "HW": 10.0}
    annual_hours = problem.config["COSTING_ANNUAL_OP_TIME"]

    expected = sum(
        prices[name] * duty * annual_hours / 1000.0
        for name, duty in _design_duties(network).items()
    )

    assert network.utility_cost == pytest.approx(expected, rel=1e-9)


def test_a_multiperiod_utility_only_network_meets_each_periods_targets():
    problem = _cold_only(
        heat_flows=([110.0, 80.0], [120.0, 150.0]),
        options={"PROBLEM_PERIOD_IDS": ["base", "peak"]},
    )

    network = problem.design.heat_exchanger_network(width=0, depth=0).selected_network

    assert network.period_ids == ("base", "peak")
    for period_id in ("base", "peak"):
        _assert_same_duties(
            _design_duties(network, period_id), _targets(problem, period_id)
        )


def test_overdesign_is_rejected_for_a_utility_only_network():
    with pytest.raises(ValueError, match="only hot or only cold"):
        _cold_only().design.heat_exchanger_network(
            width=0, depth=0, overdesign_percentage=10.0
        )


def test_a_one_sided_zone_of_a_site_is_designed_at_its_targeted_levels():
    site = PinchProblem(str(ZONAL_SITE))
    expected = _targets_of_zone(site, "Residential[S]")

    network = site.design.heat_exchanger_network(
        zone="Residential[S]", width=0, depth=0
    ).selected_network

    _assert_same_duties(_design_duties(network), expected)


def test_price_placement_cools_a_hot_only_problem_on_the_cheapest_utility():
    """``HENS_UTILITY_PLACEMENT: "price"`` was ignored: a one-sided network
    took targeting's levels, the least valuable first (review finding 33).
    CW at 5 $/MWh can take every hot stream's heat, so by price it takes
    all of it, not MPS, LPS and HW at 30, 20 and 10 $/MWh."""
    problem = _hot_only()
    total = sum(_targets(problem).values())

    network = problem.design.heat_exchanger_network(
        width=0, depth=0, options={"HENS_UTILITY_PLACEMENT": "price"}
    ).selected_network

    _assert_same_duties(_design_duties(network), {"CW": total})


def _targets_of_zone(problem, zone):
    result = problem.target.direct_heat_integration(zone=zone)
    return {
        utility.name: float(utility.heat_flow)
        for utility in (*result.hot_utilities, *result.cold_utilities)
        if float(utility.heat_flow) > 1e-6
    }
