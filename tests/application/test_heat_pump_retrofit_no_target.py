"""A network with no heat pump target gets the plain search's result.

The as-piped target is ``None`` when the existing network's heaters
carry no load: there is nothing for a heat pump to deliver. The retrofit
then has no heat pump step to try, and its result is the search without
one.
"""

from __future__ import annotations

from types import SimpleNamespace

from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
    RetrofitSettings,
)
from OpenPinch.application import heat_pump_retrofit


def test_with_no_as_piped_target_the_result_is_the_search_without_a_heat_pump(
    monkeypatch,
):
    base = SimpleNamespace(frontier=(SimpleNamespace(network=object()),))
    searches = []

    def search(problem, existing, settings, **arguments):
        searches.append(arguments)
        return base

    monkeypatch.setattr(heat_pump_retrofit, "run_retrofit_search", search)
    monkeypatch.setattr(
        heat_pump_retrofit, "as_piped_heat_pump", lambda *args, **kwargs: None
    )

    settings = RetrofitSettings(
        heat_pump_cycle="carnot", heat_pump_cost=0.0, max_modifications=1
    )
    problem = SimpleNamespace(
        master_zone=SimpleNamespace(config=SimpleNamespace(costing=None))
    )
    result = heat_pump_retrofit.retrofit_with_heat_pump(problem, object(), settings)

    assert result is base
    assert searches == [{}]
