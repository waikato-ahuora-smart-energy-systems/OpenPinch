"""A retrofit of two periods finds its heat pump, sized and run per period.

The levels case over two equal periods, its heater needing 720 kW in the
peak against 600 kW in the base, with electricity at 50 $/MWh. The heat
pump step designs one machine for both periods' coolers and heaters; its
condensers deliver about 240 kW in the base and 317 kW in the peak, the
steam heater is left with under 1 kW in each, and the design nets about
31,900 $/y (4 s with the target stopped at 200 evaluations).
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from tests.analysis.heat_exchanger_networks.heat_pump_levels_case import (
    EXISTING,
    levels_problem,
    solvers_available,
)


def _two_periods() -> PinchProblem:
    source = levels_problem().to_problem_json()
    source["options"] |= {
        "COSTING_HPR_ELE_PRICE": 50.0,
        "PROBLEM_PERIOD_IDS": ["base", "peak"],
        "PROBLEM_PERIOD_WEIGHTS": [0.5, 0.5],
    }
    for stream in source["streams"]:
        if stream["name"] == "C1":
            stream["heat_flow"] = {"values": [600.0, 720.0], "unit": "kW"}
    return PinchProblem(source=source, project_name="two periods")


@pytest.mark.synthesis
@pytest.mark.solver
@pytest.mark.skipif(not solvers_available(), reason="needs ipopt")
def test_a_retrofit_of_two_periods_runs_its_heat_pump_in_each():
    result = _two_periods().design.retrofit(
        EXISTING,
        max_modifications=1,
        heat_pump_cycle="carnot",
        heat_pump_cost=100.0,
        heat_pump_search={"maximum_evaluations": 200},
    )

    best = result.frontier[-1]
    assert [h.machine for h in best.network.heat_pumps] == ["HP"]
    assert best.net_annual_benefit > 0.0
    exchangers = {e.exchanger_id: e for e in best.network.exchangers}
    delivered = [
        sum(
            e.period_states[period].duty
            for name, e in exchangers.items()
            if name.startswith("new:HP:HP_H")
        )
        for period in (0, 1)
    ]
    # Each period's condensers deliver that period's duty.
    assert delivered[1] > delivered[0] + 50.0
    assert all(state.duty < 1.0 for state in exchangers["E2"].period_states)
