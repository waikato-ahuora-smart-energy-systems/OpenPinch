"""The heat pump step is one of the search's modifications.

``max_modifications`` bounds the steps of a retrofit. A search started
from a heat pump step has taken one already, whatever number of
exchangers serve the machine, so with a limit of one it solves the heat
pump's design and stops.
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
def test_with_one_modification_allowed_only_the_heat_pump_design_is_solved():
    run = heat_pump_search(max_modifications=1)

    assert run.result.evaluated_designs == 1
    assert len(run.result.frontier) == 1
