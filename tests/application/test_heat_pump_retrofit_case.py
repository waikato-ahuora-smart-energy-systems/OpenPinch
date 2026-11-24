"""The retrofit's heat pump joins a copy of the problem as streams.

The second search runs on the problem plus the machine's condensers and
evaporators: process streams of the network's process zone, marked with
the machine and their role, at the heat pump's own contribution
(``HPR_DT_CONT``), which is how far the as-piped target stood them off
the coolers and heaters.
"""

from __future__ import annotations

from types import SimpleNamespace

from OpenPinch.application.heat_pump_retrofit import heat_pump_case
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.stream_collection import StreamCollection
from tests.analysis.heat_exchanger_networks.heat_pump_levels_case import (
    levels_problem,
)


def _stream(name: str, supply: float, target: float, duty: float) -> Stream:
    return Stream(
        name=name,
        supply_temperature=supply,
        target_temperature=target,
        heat_flow=duty,
        heat_transfer_coefficient=1.0,
    )


def test_the_case_holds_the_machines_streams_at_the_heat_pumps_contribution():
    problem = levels_problem()
    solved = SimpleNamespace(
        hpr_hot_streams={"a": _stream("HP_H1", 110.0, 109.99, 120.0)},
        hpr_cold_streams={"b": _stream("HP_C1", 33.99, 34.0, 80.0)},
        amb_streams=StreamCollection(),
    )

    case = heat_pump_case(problem, solved, machine="HP")

    marked = {
        record["name"]: record
        for record in case.to_problem_json()["streams"]
        if record["heat_pump"] is not None
    }
    assert {name: record["heat_pump"]["role"] for name, record in marked.items()} == {
        "HP_H1": "condenser",
        "HP_C1": "evaporator",
    }
    approach = float(problem._require_prepared_root_zone().config.hpr.dt_cont)
    for record in marked.values():
        assert record["zone"] == "Site/P"
        assert record["dt_cont"]["value"] == approach
        assert record["heat_pump"]["machine"] == "HP"
