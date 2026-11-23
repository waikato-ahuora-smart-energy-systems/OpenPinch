"""A Carnot retrofit heat pump is tried as streams by default.

On the guide's case the as-piped machine added as streams nets
$2.81M/y where the linked heat pump nets $1.23M/y, so with
``heat_pump_cycle="carnot"`` the retrofit runs the stream step unless
``heat_pump_streams=False`` asks for the linked move.
"""

from __future__ import annotations

from tests.analysis.heat_exchanger_networks.heat_pump_levels_case import (
    EXISTING,
    levels_problem,
)


def _calls(monkeypatch):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import search
    from OpenPinch.application import heat_pump_retrofit

    calls = []

    def streams(problem, existing, settings, **arguments):
        calls.append(("streams", settings.heat_pump_cycle))
        return object()

    def linked(problem, existing, settings, **arguments):
        calls.append(("linked", settings.heat_pump_cycle))
        return object()

    monkeypatch.setattr(heat_pump_retrofit, "retrofit_with_heat_pump", streams)
    monkeypatch.setattr(search, "run_retrofit_search", linked)
    return calls


def test_a_carnot_retrofit_runs_the_stream_step_unless_told_not_to(monkeypatch):
    calls = _calls(monkeypatch)
    arguments = dict(heat_pump_cycle="carnot", heat_pump_cost=100.0)

    levels_problem().design.retrofit(EXISTING, **arguments)
    levels_problem().design.retrofit(EXISTING, heat_pump_streams=False, **arguments)

    assert calls == [("streams", "carnot"), ("linked", "carnot")]
