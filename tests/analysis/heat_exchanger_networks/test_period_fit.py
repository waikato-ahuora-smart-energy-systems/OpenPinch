"""Stream-end utilities added before fitting a network to every period."""

from __future__ import annotations

from OpenPinch.analysis.heat_exchanger_networks.period_fit import (
    with_stream_end_utilities,
)
from OpenPinch.application.problem import PinchProblem
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork
from tests.analysis.heat_exchanger_networks.test_hen_period_handling import (
    _two_period_problem,
)

MILK, CIP = "Process A.Milk Concentrate", "Process A.CIP Water"
RAW, FLASH = "Process A.Raw Milk", "Process A.HT Flash"


def _with_low_pressure_steam() -> PinchProblem:
    # A cheaper steam at 520 K reaches CIP Water (to 500 K) but not Milk
    # Concentrate (to 650 K).
    source = _two_period_problem().to_problem_json()
    steam = dict(source["utilities"][0])
    steam |= {
        "name": "LPS",
        "t_supply": {"value": 520.0, "unit": "K"},
        "t_target": {"value": 520.0, "unit": "K"},
        "price": {"value": 30.0, "unit": "$/MWh"},
    }
    source["utilities"].append(steam)
    return PinchProblem(source, project_name="two-period")


def _placements(network):
    return sorted(
        (exchanger.kind.value, exchanger.source_stream, exchanger.sink_stream)
        for exchanger in network.exchangers
        if exchanger.kind.value != "recovery"
    )


def test_each_stream_end_gets_the_cheapest_utility_that_reaches_it():
    network = HeatExchangerNetwork.from_structure(
        recovery=[(RAW, MILK, 1), (FLASH, CIP, 1)], stage_count=1
    )

    padded = with_stream_end_utilities(
        _with_low_pressure_steam(), network, approach=1.0
    )

    assert _placements(padded) == [
        ("cold_utility", FLASH, "Cold Utility.CW"),
        ("cold_utility", RAW, "Cold Utility.CW"),
        ("hot_utility", "Hot Utility.HPS", MILK),
        ("hot_utility", "Hot Utility.LPS", CIP),
    ]


def test_a_stream_end_already_served_gets_no_second_utility():
    network = HeatExchangerNetwork.from_structure(
        recovery=[(RAW, MILK, 1)],
        heaters=[(MILK, "Hot Utility.HPS")],
        coolers=[(RAW, "Cold Utility.CW")],
        stage_count=1,
    )

    padded = with_stream_end_utilities(
        _with_low_pressure_steam(), network, approach=1.0
    )

    assert _placements(padded).count(("hot_utility", "Hot Utility.HPS", MILK)) == 1
    assert _placements(padded).count(("cold_utility", RAW, "Cold Utility.CW")) == 1


def test_an_end_utility_one_period_cannot_use_is_replaced():
    # CIP Water heats to 500 K in base but 530 K in peak: the 520 K steam the
    # basis design used cannot finish it at peak, so HPS takes its place.
    source = _with_low_pressure_steam().to_problem_json()
    cip = next(s for s in source["streams"] if s["name"] == "CIP Water")
    cip["t_target"] = {"values": [500.0, 530.0], "unit": "K"}
    cip["heat_capacity_flowrate"] = None
    cip["heat_flow"] = {"values": [1500.0, 1800.0], "unit": "kW"}
    network = HeatExchangerNetwork.from_structure(
        recovery=[(FLASH, CIP, 1)],
        heaters=[(CIP, "Hot Utility.LPS")],
        stage_count=1,
    )

    padded = with_stream_end_utilities(
        PinchProblem(source, project_name="two-period"), network, approach=1.0
    )

    heaters = [p for p in _placements(padded) if p[0] == "hot_utility"]
    assert ("hot_utility", "Hot Utility.LPS", CIP) not in heaters
    assert ("hot_utility", "Hot Utility.HPS", CIP) in heaters


def _dairy_chilling() -> PinchProblem:
    return PinchProblem("dairy_chilling_multiperiod.json")


def _ice_water_coolers(network):
    return [
        exchanger.sink_stream
        for exchanger in network.exchangers
        if exchanger.source_stream == "Dairy.Ice-water return"
    ]


def test_without_an_approach_each_pair_is_padded_at_its_contributions():
    """Ice-water return (8 to 2 C, dt_cont 2.5) and Glycol (-5 to 0 C, dt_cont
    1) need 3.5 K at multiplier 1, as the fit holds, with end gaps of 8 and
    7 K: the cooler stays, where twice THERMAL_DT_CONT (10 K) dropped it. At
    multiplier 3 the pair needs 10.5 K and the cooler goes."""
    network = HeatExchangerNetwork.from_structure(
        recovery=[], coolers=[("Dairy.Ice-water return", "Cold Utility.Glycol")]
    )

    kept = with_stream_end_utilities(_dairy_chilling(), network, dt_cont_multiplier=1.0)
    dropped = with_stream_end_utilities(
        _dairy_chilling(), network, dt_cont_multiplier=3.0
    )

    assert _ice_water_coolers(kept) == ["Cold Utility.Glycol"]
    assert "Cold Utility.Glycol" not in _ice_water_coolers(dropped)
