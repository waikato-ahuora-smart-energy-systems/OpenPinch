"""The case holds only the heat pump levels that carry duty.

A level the as-piped target left at 0 kW would be a process stream of the
case with no exchanger in the search's first design, which the solve
refuses. The case leaves it out, as the step does.
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


def test_a_level_without_duty_is_not_a_stream_of_the_case():
    solved = SimpleNamespace(
        hpr_hot_streams={
            "a": _stream("HP_H1", 110.0, 109.99, 120.0),
            "b": _stream("HP_H2", 28.81, 28.80, 0.0),
        },
        hpr_cold_streams={"c": _stream("HP_C1", 33.99, 34.0, 80.0)},
        amb_streams=StreamCollection(),
    )

    case = heat_pump_case(levels_problem(), solved, machine="HP")

    marked = [
        record["name"]
        for record in case.to_problem_json()["streams"]
        if record["heat_pump"] is not None
    ]
    assert marked == ["HP_H1", "HP_C1"]
