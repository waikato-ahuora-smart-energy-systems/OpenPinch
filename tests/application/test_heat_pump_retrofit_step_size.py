"""A heat pump step can be built at a share of the heaters' load.

``heat_pump_step(..., load_fraction=)`` asks the as-piped target for that
share and records it on the machine (``NetworkHeatPump.load_fraction``),
so a result that tried several sizes says which one it kept.
"""

from __future__ import annotations

from types import SimpleNamespace

from OpenPinch.application import heat_pump_retrofit
from OpenPinch.domain.configuration import Configuration
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.stream_collection import StreamCollection


def test_the_step_asks_for_its_share_and_records_it(monkeypatch):
    asked = []
    solved = SimpleNamespace(
        hpr_hot_streams={
            "a": Stream(
                name="HP_H1",
                supply_temperature=120.0,
                target_temperature=119.99,
                heat_flow=50.0,
            )
        },
        hpr_cold_streams={
            "b": Stream(
                name="HP_C1",
                supply_temperature=39.99,
                target_temperature=40.0,
                heat_flow=35.0,
            )
        },
        amb_streams=StreamCollection(),
        w_net=15.0,
        cop_h=50.0 / 15.0,
    )

    def target(*args, **kwargs):
        asked.append(kwargs.get("load_fraction"))
        return solved

    monkeypatch.setattr(heat_pump_retrofit, "as_piped_heat_pump", target)
    problem = SimpleNamespace(
        master_zone=SimpleNamespace(
            config=SimpleNamespace(costing=Configuration().costing)
        )
    )
    settings = SimpleNamespace(heat_pump_cycle="carnot", heat_pump_cost=0.0)

    step, _solved = heat_pump_retrofit.heat_pump_step(
        problem, SimpleNamespace(network=object()), settings, load_fraction=0.5
    )

    assert asked == [0.5]
    assert step.record.load_fraction == 0.5
