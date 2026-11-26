"""The network of a retrofit's heat pump step is read again by the problem.

A retrofit with ``heat_pump_cycle`` adds its heat pump as streams of a case
it derives, and its networks name those streams. The machine each network
records carries them, as a designed heat pump network's does, so a second
retrofit of the selected network builds that case again from its JSON.
"""

from __future__ import annotations

import pytest

from tests.analysis.heat_exchanger_networks.heat_pump_levels_case import (
    EXISTING,
    levels_problem,
    solvers_available,
)


@pytest.mark.synthesis
@pytest.mark.solver
@pytest.mark.skipif(not solvers_available(), reason="needs ipopt")
def test_a_second_retrofit_takes_the_network_of_a_heat_pump_step():
    """It was refused: the problem has no stream of a condenser's name."""
    # The case and prices of the test that selects the as-piped heat pump.
    options = {"COSTING_HX_AREA_COEFF": 2500.0}
    first = levels_problem().design.retrofit(
        EXISTING,
        max_modifications=1,
        new_unit_cost=0.0,
        heat_pump_cycle="carnot",
        heat_pump_cost=0.0,
        heat_pump_search={"maximum_evaluations": 200},
        options=options,
    )
    pumped = first.selected.network
    assert pumped.heat_pumps

    second = levels_problem().design.retrofit(
        pumped.model_dump(mode="json"), max_modifications=0, options=options
    )

    again = second.frontier[0].network
    assert again.heat_pumps == pumped.heat_pumps
    assert again.utility_cost == pytest.approx(pumped.utility_cost, rel=1e-6)


@pytest.mark.synthesis
@pytest.mark.solver
@pytest.mark.skipif(not solvers_available(), reason="needs ipopt")
def test_a_second_retrofit_reports_the_existing_heat_pumps_electricity():
    """A retrofit design's utility cost includes its heat pump's
    electricity, so the second retrofit reports for the existing network
    what the first reported for it. It reported the utilities alone."""
    options = {"COSTING_HX_AREA_COEFF": 2500.0}
    first = levels_problem().design.retrofit(
        EXISTING,
        max_modifications=1,
        new_unit_cost=0.0,
        heat_pump_cycle="carnot",
        heat_pump_cost=0.0,
        heat_pump_search={"maximum_evaluations": 200},
        options=options,
    )

    second = levels_problem().design.retrofit(
        first.selected.network.model_dump(mode="json"),
        max_modifications=0,
        options=options,
    )

    assert second.frontier[0].utility_cost == pytest.approx(
        first.selected.utility_cost, rel=1e-6
    )
