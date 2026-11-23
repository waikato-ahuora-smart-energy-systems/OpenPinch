"""A heat pump step's savings are net of the machine's electricity.

The step's design uses less steam and cooling water than the existing
network. Its savings are measured against the existing network's
utility cost in the search without the heat pump, and the electricity
the machine draws, a constant of the step, counts as an operating cost
beside the utilities.
"""

from __future__ import annotations

import pytest

from tests.analysis.heat_exchanger_networks.heat_pump_levels_case import (
    heat_pump_search,
    solvers_available,
)

ELECTRICITY = 20_000.0


@pytest.mark.synthesis
@pytest.mark.solver
@pytest.mark.skipif(not solvers_available(), reason="needs ipopt")
def test_the_savings_are_the_utility_saved_less_the_electricity():
    free = heat_pump_search()
    priced = heat_pump_search(electricity_cost=ELECTRICITY)

    utilities = free.result.frontier[0].utility_cost
    design = priced.result.frontier[0]
    assert utilities < priced.base.utility_cost
    assert design.utility_cost == pytest.approx(utilities + ELECTRICITY, rel=1e-6)
    assert design.annual_savings == pytest.approx(
        priced.base.utility_cost - utilities - ELECTRICITY, rel=1e-6
    )
