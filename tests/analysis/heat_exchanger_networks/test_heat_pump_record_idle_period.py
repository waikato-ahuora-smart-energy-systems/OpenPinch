"""A period in which the heat pump is off still counts in its year.

A shared design can leave a period out: an off season, or a load too
small to run for. The machine draws no work and gives no heat then, but
the period keeps its share of the year, so the annual electricity is the
running periods' work diluted by the idle one.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch.analysis.heat_exchanger_networks.heat_pump_cost import (
    heat_pump_record,
)
from OpenPinch.domain.stream import Stream

COSTING = SimpleNamespace(
    hpr_ele_price=100.0, annual_op_time=8300.0, discount_rate=0.07, service_life=20.0
)


def test_a_period_without_an_output_counts_as_no_work_and_no_duty():
    running = {
        "hpr_hot_streams": {
            "HP_H1": Stream(
                name="HP_H1",
                supply_temperature=230.0,
                target_temperature=229.99,
                heat_flow=600.0,
            )
        },
        "hpr_cold_streams": {
            "HP_C1": Stream(
                name="HP_C1",
                supply_temperature=100.0,
                target_temperature=100.01,
                heat_flow=400.0,
            )
        },
        "w_net": 200.0,
    }
    target = SimpleNamespace(
        hpr_cycle="Cascade Carnot cycles",
        hpr_work=200.0,
        hpr_cop=3.0,
        hpr_hot_streams=running["hpr_hot_streams"],
        hpr_cold_streams=running["hpr_cold_streams"],
        # The off season is not among the period outputs.
        hpr_details=SimpleNamespace(period_outputs={"season": running}),
    )

    record = heat_pump_record(
        target,
        machine="HP",
        costing=COSTING,
        cost_per_kw=200.0,
        period_weights={"off season": 0.5, "season": 0.5},
    )

    assert record.work == pytest.approx(100.0)
    assert record.electricity_cost == pytest.approx(100.0 * 100.0 * 8.3)
    assert (record.condenser_duty, record.evaporator_duty) == (600.0, 400.0)
    assert record.cop == pytest.approx(3.0)
