"""The case a heat pump network was designed on, built from the network.

``heat_pump_network_case`` adds the streams the network's machine carries
to the problem it is given. The network may come back as JSON to a problem
loaded under another project name, whose zones have other addresses.
"""

from __future__ import annotations

from OpenPinch import PinchProblem
from OpenPinch.application.heat_pump_design import (
    heat_pump_design_case,
    heat_pump_network_case,
)
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork
from tests.application.heat_pump_stream_cases import basic_source


def test_the_streams_join_their_zone_of_a_problem_of_another_name():
    run = heat_pump_design_case(
        PinchProblem(source=basic_source(), project_name="basic"),
        1.0,
        cycle="carnot",
        levels={"condensers": 1, "evaporators": 1},
        search={"restarts": 0, "evaluations": 1},
    )
    network = HeatExchangerNetwork(heat_pumps=(run.record,))
    other = PinchProblem(source=basic_source(), project_name="other")

    case = heat_pump_network_case(other, network)

    added = {
        record["name"]: record["zone"]
        for record in case.to_problem_json()["streams"]
        if record.get("heat_pump")
    }
    assert added == {"HP_H1": "other/Plant", "HP_C1": "other/Plant"}
