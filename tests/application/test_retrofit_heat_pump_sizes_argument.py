"""``design.retrofit`` passes the heat pump sizes to try.

``heat_pump_sizes`` lists shares of the existing heaters' load at which
the as-piped heat pump is tried as streams; by default the full load,
three quarters and a half.
"""

from __future__ import annotations

from tests.analysis.heat_exchanger_networks.heat_pump_levels_case import (
    EXISTING,
    levels_problem,
)


def test_design_retrofit_passes_the_sizes(monkeypatch):
    from OpenPinch.application import heat_pump_retrofit

    calls = []

    def retrofit(problem, existing, settings, **arguments):
        # No solve: the Pyodide lane has no solver, and only the sizes
        # that reach the sweep matter here.
        calls.append(arguments.get("sizes"))
        return object()

    monkeypatch.setattr(heat_pump_retrofit, "retrofit_with_heat_pump", retrofit)
    problem = levels_problem()
    for sizes in (None, (0.6, 0.3)):
        arguments = {} if sizes is None else {"heat_pump_sizes": sizes}
        problem.design.retrofit(
            EXISTING,
            max_modifications=0,
            heat_pump_cycle="carnot",
            heat_pump_cost=100.0,
            **arguments,
        )

    assert calls == [(1.0, 0.75, 0.5), (0.6, 0.3)]
