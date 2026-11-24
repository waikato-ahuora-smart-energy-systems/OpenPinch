"""design.retrofit passes utility_switch_cost to the retrofit search.

Utility switching is off unless a cost is given: None leaves the search's
switches out, as relocation_cost does for relocations.
"""

from __future__ import annotations

from tests.analysis.heat_exchanger_networks.heat_pump_levels_case import (
    EXISTING,
    levels_problem,
)


def test_design_retrofit_passes_the_utility_switch_cost(monkeypatch):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import search

    seen = []

    def run(problem, existing, settings, **arguments):
        seen.append(settings.utility_switch_cost)
        return object()

    monkeypatch.setattr(search, "run_retrofit_search", run)

    levels_problem().design.retrofit(EXISTING)
    levels_problem().design.retrofit(EXISTING, utility_switch_cost=5_000.0)

    assert seen == [None, 5_000.0]
