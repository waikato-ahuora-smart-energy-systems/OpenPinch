"""``design.retrofit`` refuses heat pump sizes outside 0 to 1 up front.

A size is a share of the existing heaters' load. One outside 0 to 1 is
refused before any search runs, naming the argument.
"""

from __future__ import annotations

import pytest

from tests.analysis.heat_exchanger_networks.heat_pump_levels_case import (
    EXISTING,
    levels_problem,
)


def test_a_size_outside_zero_to_one_is_refused_before_the_search(monkeypatch):
    from OpenPinch.analysis.heat_exchanger_networks.retrofit import search

    def no_search(*args, **kwargs):
        raise AssertionError("the search ran")

    monkeypatch.setattr(search, "run_retrofit_search", no_search)

    with pytest.raises(ValueError, match="heat_pump_sizes.*between 0 and 1.*1.5"):
        levels_problem().design.retrofit(
            EXISTING,
            heat_pump_cycle="carnot",
            heat_pump_cost=100.0,
            heat_pump_streams=True,
            heat_pump_sizes=(1.0, 1.5),
        )
    # The stream step is the Carnot default, so the sizes are checked
    # without asking for it.
    with pytest.raises(ValueError, match="heat_pump_sizes.*between 0 and 1.*1.5"):
        levels_problem().design.retrofit(
            EXISTING,
            heat_pump_cycle="carnot",
            heat_pump_cost=100.0,
            heat_pump_sizes=(1.5,),
        )
