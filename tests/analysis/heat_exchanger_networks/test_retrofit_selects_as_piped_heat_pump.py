"""``design.retrofit`` finds the heat pump the as-piped target describes.

On the levels case a heat pump pays above a COP of 2.5. The one machine
the linked-utility move offers lifts from the cooler's cold end to the
heater's hot end at a COP of 2.33, so the search used to select no heat
pump at all. The as-piped target places three condensers along the
heater and two evaporators along the cooler at a COP above 3. Added to
the network as streams, with the matches that serve them, that machine
is selected.
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
def test_the_search_selects_the_as_piped_heat_pump():
    result = levels_problem().design.retrofit(
        EXISTING,
        max_modifications=1,
        new_unit_cost=0.0,
        heat_pump_cycle="carnot",
        heat_pump_cost=0.0,
        # A full heat pump search takes a minute here; 200 evaluations
        # (0.4 s) already give a machine at a COP of 3.1.
        heat_pump_search={"maximum_evaluations": 200},
        # Exchanger area at a quarter of the default price. The target
        # places its levels 6 K from the cooler and the heater, and at
        # 10,000 $/m2^0.6 the area that needs costs 14,980 $/y against
        # 14,787 $/y saved: the comparison would be of area, not of the
        # two machines.
        options={"COSTING_HX_AREA_COEFF": 2500.0},
    )

    selected = result.selected
    machines = [m for m in selected.modifications if m.kind == "heat_pump"]
    assert len(machines) == 1
    assert machines[0].heat_pump_cop > 3.0
    # Less steam than the existing 240 kW. With one size the full machine
    # left under 100 kW; tried at three sizes (the default since the duty
    # sweep) a smaller one nets more, 11,393 $/y against 11,042, and
    # leaves 123 kW.
    assert selected.hot_utility < 240.0
    assert selected.net_annual_benefit > 0.0
