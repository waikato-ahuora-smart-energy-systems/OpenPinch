"""The stream step replaces the linked heat pump move.

With the stream step the heat pump is the as-piped machine added as
streams. The plain search the step is joined with offers no linked heat
pump of its own: on the guide's case that move nets $1.23M/y against the
stream machine's $2.81M/y. The step itself still prices the machine with
the search's heat pump settings.
"""

from __future__ import annotations

from types import SimpleNamespace

from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
    RetrofitSettings,
)
from OpenPinch.application import heat_pump_retrofit


def test_the_plain_search_offers_no_linked_heat_pump(monkeypatch):
    cycles = []
    base = SimpleNamespace(frontier=(SimpleNamespace(network=object()),))

    def search(problem, existing, settings, **arguments):
        cycles.append(("search", settings.heat_pump_cycle))
        return base

    def step(problem, base, settings, **arguments):
        cycles.append(("step", settings.heat_pump_cycle))
        return None, None

    monkeypatch.setattr(heat_pump_retrofit, "run_retrofit_search", search)
    monkeypatch.setattr(heat_pump_retrofit, "heat_pump_step", step)

    heat_pump_retrofit.retrofit_with_heat_pump(
        SimpleNamespace(period_ids={"0": 0}),
        object(),
        RetrofitSettings(heat_pump_cycle="carnot", heat_pump_cost=300.0),
    )

    assert cycles == [("search", None), ("step", "carnot")]
