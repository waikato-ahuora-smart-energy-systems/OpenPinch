"""A heat pump step's network records the machine.

The design's network holds the heat pump's streams and the exchangers
that serve them, which alone would read as recovery between process
streams. The machine is recorded on the network, as a synthesis design
with a heat pump records it, so the network says whose streams they are
and what the machine draws.
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
def test_the_designs_network_holds_the_heat_pump():
    run = heat_pump_search(electricity_cost=20_000.0)

    assert run.result.frontier[0].network.heat_pumps == (run.record,)
