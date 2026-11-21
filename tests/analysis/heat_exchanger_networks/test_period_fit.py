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


def _plain_streams_with_segmented_steam(dt_cont: tuple[float, float]) -> PinchProblem:
    # Steam gives 250 to 225 C over its inlet segment and 225 to 200 C over
    # its outlet segment, heating a plain stream from 50 to 150 C: an inlet
    # gap of 100 K and an outlet gap of 150 K.
    def segment(t_supply, t_target, dt):
        return {
            "t_supply": t_supply,
            "t_target": t_target,
            "heat_flow": 100.0,
            "htc": 1.0,
            "price": 20.0,
            "dt_cont": dt,
        }

    def stream(name, t_supply, t_target):
        return {
            "zone": "Site",
            "name": name,
            "t_supply": t_supply,
            "t_target": t_target,
            "heat_flow": 150.0,
        }

    return PinchProblem(
        {
            "streams": [stream("Hot", 200.0, 100.0), stream("Cold", 50.0, 150.0)],
            "utilities": [
                {
                    "name": "HU",
                    "type": "Hot",
                    "segments": [
                        segment(250.0, 225.0, dt_cont[0]),
                        segment(225.0, 200.0, dt_cont[1]),
                    ],
                },
                {"name": "CU", "type": "Cold", "t_supply": 20.0, "t_target": 20.01},
            ],
        },
        project_name="Site",
    )


def _heaters(network):
    return [p for p in _placements(network) if p[0] == "hot_utility"]


def test_a_segmented_utility_is_screened_at_its_terminal_segments():
    """A 120 K contribution on the steam's outlet segment fits the 150 K
    outlet gap; the inlet segment's 0.1 K fits the 100 K inlet gap. The
    profile's largest contribution at both ends would reject the steam, where
    the fit's approach equations (inlet and outlet segments) accept it."""
    network = HeatExchangerNetwork.from_structure(recovery=[])

    terminal = with_stream_end_utilities(
        _plain_streams_with_segmented_steam((0.1, 120.0)),
        network,
        dt_cont_multiplier=1.0,
    )
    inlet_too_large = with_stream_end_utilities(
        _plain_streams_with_segmented_steam((120.0, 0.1)),
        network,
        dt_cont_multiplier=1.0,
    )

    assert _heaters(terminal) == [("hot_utility", "Hot Utility.HU", "Site.Cold")]
    assert _heaters(inlet_too_large) == []


def test_a_segmented_stream_is_screened_like_a_plain_one():
    """A segmented stream heated from 50 to 150 C has no single heat capacity
    flowrate, but it still runs: the cheap 120 C steam cannot finish it, so
    its end heater is the dearer 250 C steam."""

    def steam(name, temperature, price):
        return {
            "name": name,
            "type": "Hot",
            "t_supply": temperature,
            "t_target": temperature - 0.01,
            "price": price,
        }

    problem = PinchProblem(
        {
            "streams": [
                {
                    "zone": "Site",
                    "name": "Hot",
                    "t_supply": 200.0,
                    "t_target": 100.0,
                    "heat_flow": 150.0,
                },
                {
                    "zone": "Site",
                    "name": "Cold",
                    "segments": [
                        {"t_supply": 50.0, "t_target": 100.0, "heat_flow": 50.0},
                        {"t_supply": 100.0, "t_target": 150.0, "heat_flow": 100.0},
                    ],
                },
            ],
            "utilities": [
                steam("LPS", 120.0, 10.0),
                steam("HPS", 250.0, 50.0),
                {"name": "CU", "type": "Cold", "t_supply": 20.0, "t_target": 20.01},
            ],
        },
        project_name="Site",
    )

    padded = with_stream_end_utilities(
        problem, HeatExchangerNetwork.from_structure(recovery=[]), approach=1.0
    )

    assert _heaters(padded) == [("hot_utility", "Hot Utility.HPS", "Site.Cold")]
