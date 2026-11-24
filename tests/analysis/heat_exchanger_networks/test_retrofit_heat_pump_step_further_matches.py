"""Steps after a heat pump step leave the machine's streams alone.

Once the heat pump is in the network the search goes on as usual, one
modification at a time. The new matches it screens are between the
process's own streams: a match between a condenser and an evaporator
would hand the machine's output back to its input, and the machine's
streams already have the matches its step gave them.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.retrofit import search
from tests.analysis.heat_exchanger_networks.heat_pump_levels_case import (
    heat_pump_search,
    solvers_available,
)


@pytest.mark.synthesis
@pytest.mark.solver
@pytest.mark.skipif(not solvers_available(), reason="needs ipopt")
def test_new_matches_are_screened_among_the_process_streams_only(monkeypatch):
    screened = []
    original = search.screen_new_matches

    def spy(network, ends, **arguments):
        screened.append(set(ends))
        return original(network, ends, **arguments)

    monkeypatch.setattr(search, "screen_new_matches", spy)

    heat_pump_search(max_modifications=2)

    assert screened
    assert all(streams == {"H1", "C1"} for streams in screened)
