"""A heat pump step built from a shared design over several periods.

The as-piped target of a network of several periods holds one period in
its own collections and every period in its outputs. The step's levels
are those carrying duty in any period, each with its stream in every
period (``None`` where it is off), and the machine is priced over the
periods at the problem's weights.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch.application import heat_pump_retrofit
from OpenPinch.domain.configuration import Configuration
from OpenPinch.domain.stream import Stream
from OpenPinch.domain.stream_collection import StreamCollection


def _stream(name: str, temperature: float, duty: float) -> Stream:
    return Stream(
        name=name,
        supply_temperature=temperature,
        target_temperature=temperature - 0.01,
        heat_flow=duty,
    )


def _output(condensers: dict, evaporator: float, work: float) -> dict:
    return {
        "hpr_hot_streams": {
            name: _stream(name, 200.0, duty) for name, duty in condensers.items()
        },
        "hpr_cold_streams": {"HP_C1": _stream("HP_C1", 120.0, evaporator)},
        "w_net": work,
    }


def test_the_step_holds_each_level_in_every_period(monkeypatch):
    base = _output({"HP_H1": 600.0, "HP_H2": 0.0}, 450.0, 150.0)
    peak = _output({"HP_H1": 700.0, "HP_H2": 300.0}, 700.0, 300.0)
    solved = SimpleNamespace(
        hpr_hot_streams=base["hpr_hot_streams"],
        hpr_cold_streams=base["hpr_cold_streams"],
        amb_streams=StreamCollection(),
        w_net=150.0,
        cop_h=4.0,
        period_outputs={"base": base, "peak": peak},
    )
    monkeypatch.setattr(
        heat_pump_retrofit, "as_piped_heat_pump", lambda *args, **kwargs: solved
    )
    problem = SimpleNamespace(
        period_ids={"base": 0, "peak": 1},
        master_zone=SimpleNamespace(
            config=SimpleNamespace(costing=Configuration().costing),
            weights=[0.5, 0.5],
        ),
    )
    settings = SimpleNamespace(heat_pump_cycle="carnot", heat_pump_cost=0.0)

    step, _solved = heat_pump_retrofit.heat_pump_step(
        problem, SimpleNamespace(network=object()), settings
    )

    assert [s.name for s in step.condensers] == ["HP_H1", "HP_H2"]
    duties = {
        name: [None if s is None else float(s.heat_flow.value) for s in each]
        for name, each in (step.in_periods or {}).items()
    }
    assert duties == {
        "HP_H1": [600.0, 700.0],
        "HP_H2": [None, 300.0],
        "HP_C1": [450.0, 700.0],
    }
    assert step.record.condenser_duty == pytest.approx(1000.0)
