"""The retrofit case keeps the heat pump's approach under a multiplier.

The as-piped target stands the heat pump ``HPR_DT_CONT`` off the
coolers and heaters as given; only the process streams' contributions
carry the zone's dt_cont multiplier. The case's streams inherit the
multiplier, so they are written with ``HPR_DT_CONT`` over it: at a
multiplier of 2 and ``HPR_DT_CONT`` of 1 K, 0.5 K, which the zone scales
back to 1 K.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

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


def test_the_heat_pump_streams_carry_hpr_dt_cont_over_the_multiplier():
    problem = levels_problem()
    problem.set_dt_cont_multiplier(2.0)
    solved = SimpleNamespace(
        hpr_hot_streams={"a": _stream("HP_H1", 110.0, 109.99, 120.0)},
        hpr_cold_streams={"b": _stream("HP_C1", 33.99, 34.0, 80.0)},
        amb_streams=StreamCollection(),
    )

    case = heat_pump_case(problem, solved, machine="HP")

    written = {
        record["name"]: record["dt_cont"]["value"]
        for record in case.to_problem_json()["streams"]
        if record["heat_pump"] is not None
    }
    approach = float(problem._require_prepared_root_zone().config.hpr.dt_cont)
    assert written == {
        "HP_H1": pytest.approx(approach / 2.0),
        "HP_C1": pytest.approx(approach / 2.0),
    }


def test_the_air_carries_its_own_contribution_over_the_multiplier():
    # The target stood the air its own contribution (here 4 K) off the
    # evaporators as given; written at 4 K, the zone scaled it to 8 K.
    problem = levels_problem()
    problem.set_dt_cont_multiplier(2.0)
    ambient = StreamCollection()
    air = _stream("AIR_H1", 15.0, 14.99, 50.0)
    air.delta_t_contribution = 4.0
    ambient.add(air)
    solved = SimpleNamespace(
        hpr_hot_streams={"a": _stream("HP_H1", 110.0, 109.99, 120.0)},
        hpr_cold_streams={"b": _stream("HP_C1", 9.99, 10.0, 80.0)},
        amb_streams=ambient,
    )

    case = heat_pump_case(problem, solved, machine="HP")

    written = {
        record["name"]: record["dt_cont"]["value"]
        for record in case.to_problem_json()["streams"]
    }
    assert written["AIR_H1"] == pytest.approx(2.0)
