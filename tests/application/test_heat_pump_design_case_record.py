"""A heat pump design case carries the machine's record, priced.

Every network designed on the case serves the same heat pump, so its
record is made once with the case: the work of the target solved for that
multiplier, at the problem's electricity price, and the compressor at the
cost per kW the caller gives.
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from OpenPinch.application.heat_pump_design import heat_pump_design_case
from tests.application.heat_pump_stream_cases import basic_source


def test_the_case_prices_its_heat_pump_at_the_problems_electricity_price():
    problem = PinchProblem(source=basic_source(), project_name="basic")

    run = heat_pump_design_case(
        problem,
        1.0,
        cycle="carnot",
        levels={"condensers": 1, "evaporators": 1},
        search={"restarts": 0, "evaluations": 1},
        cost_per_kw=200.0,
    )

    work = float(run.target.hpr_work)
    assert run.record.machine == "HP"
    assert run.record.work == pytest.approx(work)
    # COSTING_HPR_ELE_PRICE 100 $/MWh over COSTING_ANNUAL_OP_TIME 8300 h.
    assert run.record.electricity_cost == pytest.approx(work * 100.0 * 8.3)
    assert run.record.capital_cost == pytest.approx(200.0 * run.record.condenser_duty)
