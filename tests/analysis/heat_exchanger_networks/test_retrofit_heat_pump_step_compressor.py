"""A heat pump step lists the machine with its compressor's cost.

The exchangers that serve the heat pump's streams are new matches,
priced as any new match. The machine itself is one more modification,
of kind ``heat_pump``: it names the machine, gives its COP and its
condenser capacity, and holds the compressor's installed cost, which
joins the design's capital.
"""

from __future__ import annotations

import pytest

from tests.analysis.heat_exchanger_networks.heat_pump_levels_case import (
    heat_pump_search,
    solvers_available,
)

COMPRESSOR = 50_000.0


@pytest.mark.synthesis
@pytest.mark.solver
@pytest.mark.skipif(not solvers_available(), reason="needs ipopt")
def test_the_machine_is_a_modification_that_holds_the_compressors_cost():
    run = heat_pump_search(capital_cost=COMPRESSOR)

    design = run.result.frontier[0]
    (machine,) = [m for m in design.modifications if m.kind == "heat_pump"]
    assert machine.exchanger_id == "HP"
    assert machine.fixed_cost == pytest.approx(COMPRESSOR)
    assert machine.heat_pump_cop == pytest.approx(run.record.cop)
    assert machine.heat_pump_capacity == pytest.approx(run.record.condenser_duty)
    matches = [m for m in design.modifications if m.kind == "new_match"]
    assert design.capital_cost == pytest.approx(
        COMPRESSOR + sum(m.fixed_cost + m.area_cost for m in matches)
    )
