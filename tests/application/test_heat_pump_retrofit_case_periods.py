"""The retrofit case of a heat pump designed over several periods.

The as-piped target of a network of several periods is a shared design:
its own stream collections hold one period, its period outputs every one.
The case writes each of its streams in every period, at that period's
temperatures and duty, and with no duty where the machine leaves it idle.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch.application.heat_pump_retrofit import heat_pump_case
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.stream_collection import StreamCollection
from tests.analysis.heat_exchanger_networks import (
    test_retrofit_as_piped_period_profiles as periods,
)


def _stream(name: str, supply: float, target: float, duty: float) -> Stream:
    return Stream(
        name=name,
        supply_temperature=supply,
        target_temperature=target,
        heat_flow=duty,
        heat_transfer_coefficient=1.0,
    )


def _output(condenser: float, evaporator: float) -> dict:
    return {
        "hpr_hot_streams": {"a": _stream("HP_H1", 280.0, 279.99, condenser)},
        "hpr_cold_streams": {"b": _stream("HP_C1", 99.99, 100.0, evaporator)},
        "w_net": condenser - evaporator,
        "Q_amb_hot": 0.0,
    }


def test_the_case_writes_the_heat_pumps_streams_in_every_period():
    base = _output(600.0, 450.0)
    solved = SimpleNamespace(
        hpr_hot_streams=base["hpr_hot_streams"],
        hpr_cold_streams=base["hpr_cold_streams"],
        amb_streams=StreamCollection(),
        period_outputs={"base": base, "peak": _output(1000.0, 700.0)},
    )

    case = heat_pump_case(periods.two_period_problem(), solved, machine="HP")

    duties = {
        record["name"]: record["heat_flow"].get("values")
        for record in case.to_problem_json()["streams"]
        if record["heat_pump"] is not None
    }
    assert duties == {
        "HP_H1": pytest.approx([600.0, 1000.0]),
        "HP_C1": pytest.approx([450.0, 700.0]),
    }
