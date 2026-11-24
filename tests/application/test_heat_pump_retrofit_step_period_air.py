"""A heat pump step of several periods carries the air of every period.

The shared design's own air stream is its selected period's, which may
draw none while another period does. The step's ambient sources are the
air of any period, each with its stream in every period at that period's
duty (``None`` where none is drawn), so the search can match it.
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


def _output(drawn: float) -> dict:
    return {
        "hpr_hot_streams": {"HP_H1": _stream("HP_H1", 200.0, 600.0)},
        "hpr_cold_streams": {"HP_C1": _stream("HP_C1", 5.0, 450.0)},
        "w_net": 150.0,
        "Q_amb_hot": drawn,
    }


def test_the_step_carries_the_air_of_every_period(monkeypatch):
    solved = SimpleNamespace(
        hpr_hot_streams={"HP_H1": _stream("HP_H1", 200.0, 600.0)},
        hpr_cold_streams={"HP_C1": _stream("HP_C1", 5.0, 450.0)},
        amb_streams=StreamCollection(),
        w_net=150.0,
        cop_h=4.0,
        period_outputs={"base": _output(0.0), "peak": _output(50.0)},
    )
    monkeypatch.setattr(
        heat_pump_retrofit, "as_piped_heat_pump", lambda *args, **kwargs: solved
    )
    problem = SimpleNamespace(
        period_ids={"base": 0, "peak": 1},
        master_zone=SimpleNamespace(config=Configuration(), weights=[0.5, 0.5]),
    )
    settings = SimpleNamespace(heat_pump_cycle="carnot", heat_pump_cost=0.0)

    step, _solved = heat_pump_retrofit.heat_pump_step(
        problem, SimpleNamespace(network=object()), settings
    )

    assert [s.name for s in step.ambient_sources] == ["AIR_H1"]
    duties = [
        None if s is None else float(s.heat_flow.value)
        for s in (step.in_periods or {}).get("AIR_H1", ())
    ]
    assert duties == [None, pytest.approx(50.0)]
