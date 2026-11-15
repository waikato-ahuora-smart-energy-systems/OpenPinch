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
