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
