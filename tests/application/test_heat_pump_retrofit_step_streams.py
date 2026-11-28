"""A retrofit's heat pump step carries its own streams only.

A retrofit of a network that already has a heat pump runs on a case that
holds that machine's streams. A second machine of the same name, added by
the retrofit's heat pump step, carries the streams its case added: not the
existing machine's, which the existing machine's record carries.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
    RetrofitSettings,
)
from OpenPinch.application import heat_pump_retrofit
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.stream_collection import StreamCollection
from tests.analysis.heat_exchanger_networks.heat_pump_levels_case import (
    levels_problem,
)
from tests.application.heat_pump_stream_cases import step_of_no_size


def _stream(name: str, supply: float, target: float, duty: float) -> Stream:
    return Stream(
        name=name,
        supply_temperature=supply,
        target_temperature=target,
        heat_flow=duty,
        heat_transfer_coefficient=1.0,
    )


def _solved(condenser: Stream, evaporator: Stream) -> SimpleNamespace:
    return SimpleNamespace(
        hpr_hot_streams={"a": condenser},
        hpr_cold_streams={"b": evaporator},
        amb_streams=StreamCollection(),
    )


def test_a_second_machine_of_the_same_name_carries_its_own_streams(monkeypatch):
    # The case of a network that already has machine "HP".
    existing_case = heat_pump_retrofit.heat_pump_case(
        levels_problem(),
        _solved(
            _stream("HP_H1", 110.0, 109.99, 120.0),
            _stream("HP_C1", 33.99, 34.0, 80.0),
        ),
        machine="HP",
    )
    condenser = _stream("HP_H2", 100.0, 99.99, 60.0)
    evaporator = _stream("HP_C2", 39.99, 40.0, 40.0)
    step = replace(
        step_of_no_size(), condensers=(condenser,), evaporators=(evaporator,)
    )
    base = SimpleNamespace(frontier=(SimpleNamespace(network=object()),))
    steps = []

    def search(case, existing, settings, **arguments):
        if "heat_pump" not in arguments:
            return base
        steps.append(arguments["heat_pump"])
        return "search from the heat pump"

    monkeypatch.setattr(heat_pump_retrofit, "run_retrofit_search", search)
    monkeypatch.setattr(
        heat_pump_retrofit,
        "heat_pump_step",
        lambda *a, **k: (step, _solved(condenser, evaporator)),
    )
    monkeypatch.setattr(heat_pump_retrofit, "joined", lambda first, *rest: rest)

    heat_pump_retrofit.retrofit_with_heat_pump(
        existing_case, object(), RetrofitSettings(heat_pump_cycle="carnot")
    )

    assert [record["name"] for record in steps[0].record.streams] == [
        "HP_H2",
        "HP_C2",
    ]
