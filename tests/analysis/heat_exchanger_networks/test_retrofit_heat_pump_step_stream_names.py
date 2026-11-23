"""A heat pump step names process streams as the existing network does.

The existing network of the levels case calls its streams ``H1`` and
``C1``; the solver knows them as ``P.H1`` and ``P.C1``. A search's
modifications use the network's names, and the matches of a heat pump
step do too, so a reader can find the stream each one is on.
"""

from __future__ import annotations

import pytest

from tests.analysis.heat_exchanger_networks.heat_pump_levels_case import (
    heat_pump_search,
    solvers_available,
)


@pytest.mark.synthesis
@pytest.mark.solver
@pytest.mark.skipif(not solvers_available(), reason="needs ipopt")
def test_the_matches_of_the_step_use_the_existing_networks_stream_names():
    run = heat_pump_search()

    matches = [m for m in run.result.frontier[0].modifications if m.kind == "new_match"]
    process = {
        name
        for m in matches
        for name in (m.hot_stream, m.cold_stream)
        if not name.startswith("HP_")
    }
    assert process == {"H1", "C1"}
