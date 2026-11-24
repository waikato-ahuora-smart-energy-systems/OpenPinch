"""The retrofit's heat pump target stops at 2000 evaluations by default.

The stream step solves one heat pump target per size. On the guide's
case the full target search found the same machine as 2000 evaluations
in ten times as long (317 s against 30 s for three sizes), so the
default stops there; ``heat_pump_search`` sets other limits, and ``{}``
asks for the full search.
"""

from __future__ import annotations

from tests.analysis.heat_exchanger_networks.heat_pump_levels_case import (
    EXISTING,
    levels_problem,
)


def test_the_heat_pump_search_stops_at_2000_evaluations_unless_told(monkeypatch):
    from OpenPinch.application import heat_pump_retrofit

    limits = []

    def streams(problem, existing, settings, **arguments):
        limits.append(arguments.get("limits"))
        return object()

    monkeypatch.setattr(heat_pump_retrofit, "retrofit_with_heat_pump", streams)
    arguments = dict(heat_pump_cycle="carnot", heat_pump_cost=100.0)

    levels_problem().design.retrofit(EXISTING, **arguments)
    levels_problem().design.retrofit(EXISTING, heat_pump_search={}, **arguments)

    assert limits == [{"maximum_evaluations": 2000}, {}]
