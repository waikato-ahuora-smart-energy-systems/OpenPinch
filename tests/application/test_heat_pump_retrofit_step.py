"""The retrofit's heat pump step is built from the as-piped target.

Given the existing network's design from the search without the heat
pump, the step is the as-piped target solved on that design's coolers
and heaters: its condenser and evaporator streams, the machine priced at
the search's ``heat_pump_cost``, and the design itself as the base the
second search measures against.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pump_targets import (
    as_piped_heat_pump,
)
from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
    RetrofitSettings,
)
from OpenPinch.application.heat_pump_retrofit import heat_pump_step
from tests.analysis.heat_exchanger_networks.test_retrofit_heat_pump_targets import (
    _problem,
    _solved_network,
)

# One evaluation takes the heat pump search's starting point as it is.
LIMITS = {"maximum_evaluations": 1}


def test_the_step_holds_the_as_piped_machine_priced_by_the_search():
    problem = _problem()
    base = SimpleNamespace(network=_solved_network())
    settings = RetrofitSettings(heat_pump_cycle="carnot", heat_pump_cost=300.0)

    step, solved = heat_pump_step(problem, base, settings, limits=LIMITS, machine="HP")

    expected = as_piped_heat_pump(problem, base.network, args=LIMITS)
    delivered = sum(
        float(s.heat_flow.value) for _k, s in expected.hpr_hot_streams.items()
    )
    assert delivered > 0.0
    assert sum(float(s.heat_flow.value) for s in step.condensers) == pytest.approx(
        delivered
    )
    assert [s.name for s in step.evaporators] == [
        s.name for _key, s in solved.hpr_cold_streams.items()
    ]
    assert step.record.capital_cost == pytest.approx(300.0 * delivered)
    assert (step.record.machine, step.record.cycle) == ("HP", "carnot")
    assert step.base is base


def test_the_step_targets_the_searchs_cycle(monkeypatch):
    from OpenPinch.application import heat_pump_retrofit

    asked = []

    def target(problem, network, **arguments):
        asked.append(arguments.get("cycle"))

    monkeypatch.setattr(heat_pump_retrofit, "as_piped_heat_pump", target)
    settings = RetrofitSettings(heat_pump_cycle="vapour_compression")

    heat_pump_step(None, SimpleNamespace(network=None), settings)

    assert asked == ["vapour_compression"]


def test_the_step_targets_at_the_searchs_approach(monkeypatch):
    # The approach a retrofit holds its matches to (min_approach_temperature)
    # is the one its heat pump has to stand off by.
    from OpenPinch.application import heat_pump_retrofit

    asked = []

    def target(problem, network, **arguments):
        asked.append(arguments.get("approach"))

    monkeypatch.setattr(heat_pump_retrofit, "as_piped_heat_pump", target)
    settings = RetrofitSettings(heat_pump_cycle="carnot", approach=10.0)

    heat_pump_step(None, SimpleNamespace(network=None), settings)

    assert asked == [10.0]
