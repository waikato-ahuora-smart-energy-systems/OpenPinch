"""The retrofit's heat pump step takes the air the target draws on.

An as-piped target can let its evaporators take heat from ambient air
(a hot stream among its ambient streams) and leave heat to air (a cold
one). The step carries the first, which the search has to match to the
evaporators; the second is not a stream of the case, whose coolers keep
that heat.
"""

from __future__ import annotations

from types import SimpleNamespace

from OpenPinch.application import heat_pump_retrofit
from OpenPinch.domain.configuration import Configuration
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.stream_collection import StreamCollection


def _stream(name: str, supply: float, target: float) -> Stream:
    return Stream(
        name=name, supply_temperature=supply, target_temperature=target, heat_flow=10.0
    )


def test_the_step_lists_the_air_the_evaporators_draw_on(monkeypatch):
    ambient = StreamCollection()
    ambient.add(_stream("AIR_H1", 15.0, 14.99))
    ambient.add(_stream("AIR_C1", 14.99, 15.0))
    solved = SimpleNamespace(
        hpr_hot_streams={"a": _stream("HP_H1", 100.0, 99.99)},
        hpr_cold_streams={"b": _stream("HP_C1", 4.99, 5.0)},
        amb_streams=ambient,
        w_net=3.0,
        cop_h=3.3,
    )
    monkeypatch.setattr(
        heat_pump_retrofit, "as_piped_heat_pump", lambda *args, **kwargs: solved
    )
    problem = SimpleNamespace(
        master_zone=SimpleNamespace(
            config=SimpleNamespace(costing=Configuration().costing)
        )
    )
    settings = SimpleNamespace(heat_pump_cycle="carnot", heat_pump_cost=0.0)

    step, _solved = heat_pump_retrofit.heat_pump_step(
        problem, SimpleNamespace(network=object()), settings
    )

    assert [stream.name for stream in step.ambient_sources] == ["AIR_H1"]
