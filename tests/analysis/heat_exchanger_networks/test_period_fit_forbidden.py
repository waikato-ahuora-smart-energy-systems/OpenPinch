"""The fit to every period adds no forbidden stream-end utility.

Before a network is fitted to every period, each stream end without a heater
or cooler gets the cheapest utility that reaches it. A pair that
``HENS_FORBIDDEN_MATCHES`` names is not offered there, and an end utility of
the network that the list forbids is dropped like one that cannot reach.
"""

from __future__ import annotations

from OpenPinch.analysis.heat_exchanger_networks.period_fit import (
    with_stream_end_utilities,
)
from OpenPinch.application.problem import PinchProblem
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork
from tests.analysis.heat_exchanger_networks.test_period_fit import (
    CIP,
    FLASH,
    MILK,
    RAW,
    _placements,
    _with_low_pressure_steam,
)


def test_a_stream_end_gets_no_forbidden_utility():
    # LPS is the cheapest heater that reaches CIP Water; forbidden there,
    # HPS takes its place.
    source = _with_low_pressure_steam().to_problem_json()
    source["options"] = source.get("options", {}) | {
        "HENS_FORBIDDEN_MATCHES": [["LPS", "CIP Water"]]
    }
    network = HeatExchangerNetwork.from_structure(
        recovery=[(RAW, MILK, 1), (FLASH, CIP, 1)], stage_count=1
    )

    padded = with_stream_end_utilities(
        PinchProblem(source, project_name="two-period"), network, approach=1.0
    )

    assert _placements(padded) == [
        ("cold_utility", FLASH, "Cold Utility.CW"),
        ("cold_utility", RAW, "Cold Utility.CW"),
        ("hot_utility", "Hot Utility.HPS", CIP),
        ("hot_utility", "Hot Utility.HPS", MILK),
    ]
