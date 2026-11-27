"""A level the heat pump target leaves without duty is not a stream.

The as-piped target can return a level that carries nothing: on Ciric
and Floudas at 70% load two of its three condensers carry 0 kW, one at
28.8 degC, far below the heater. As a stream it would still need a match
it cannot have, and the whole heat pump step would be dropped. The step
keeps only the levels that carry duty.
"""

from __future__ import annotations

from types import SimpleNamespace

from OpenPinch.application import heat_pump_retrofit
from OpenPinch.domain.configuration import Configuration
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.stream_collection import StreamCollection


def _stream(name: str, supply: float, target: float, duty: float) -> Stream:
    return Stream(
        name=name, supply_temperature=supply, target_temperature=target, heat_flow=duty
    )


def test_levels_without_duty_are_left_out_of_the_step(monkeypatch):
    solved = SimpleNamespace(
        hpr_hot_streams={
            "a": _stream("HP_H1", 209.85, 209.84, 5526.6),
            "b": _stream("HP_H2", 28.81, 28.80, 0.0),
        },
        hpr_cold_streams={
            "c": _stream("HP_C1", 125.29, 125.3, 1976.6),
            "d": _stream("HP_C2", 117.89, 117.9, 0.0),
        },
        amb_streams=StreamCollection(),
        w_net=1700.0,
        cop_h=3.24,
    )
    monkeypatch.setattr(
        heat_pump_retrofit, "as_piped_heat_pump", lambda *args, **kwargs: solved
    )
    problem = SimpleNamespace(
        master_zone=SimpleNamespace(
            config=SimpleNamespace(costing=Configuration().costing)
        )
    )
    settings = SimpleNamespace(
        heat_pump_cycle="carnot", heat_pump_cost=0.0, dt_cont_multiplier=1.0
    )

    step, _solved = heat_pump_retrofit.heat_pump_step(
        problem, SimpleNamespace(network=object()), settings
    )

    assert [s.name for s in step.condensers] == ["HP_H1"]
    assert [s.name for s in step.evaporators] == ["HP_C1"]


def test_a_level_that_carries_round_off_is_not_a_stream():
    # A search leaves a level it does not use at round-off, not at 0: a
    # condenser at 0.0003 kW beside two of 760 and 740 kW on a two-period
    # case. As a stream it got a match that did nothing, and the retrofit
    # refused the heat pump step for it.
    levels = {
        "a": _stream("HP_H1", 139.02, 139.01, 0.00032667),
        "b": _stream("HP_H2", 136.0, 135.99, 759.97588026),
        "c": _stream("HP_H3", 120.8, 120.79, 739.98498251),
    }

    carrying = heat_pump_retrofit._carrying(levels)

    assert [stream.name for stream in carrying] == ["HP_H2", "HP_H3"]
