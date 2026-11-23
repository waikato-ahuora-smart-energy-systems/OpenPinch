"""A heat pump over several periods is priced on what it does in each.

Its electricity is its work in each period for that period's share of the
year. Its compressor is bought once, for the largest condenser duty any
period asks of it.
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


def _period(condenser: float, evaporator: float, work: float) -> dict:
    return {
        "hpr_hot_streams": {
            "HP_H1": Stream(
                name="HP_H1",
                supply_temperature=230.0,
                target_temperature=229.99,
                heat_flow=condenser,
            )
        },
        "hpr_cold_streams": {
            "HP_C1": Stream(
                name="HP_C1",
                supply_temperature=100.0,
                target_temperature=100.01,
                heat_flow=evaporator,
            )
        },
        "w_net": work,
    }


def test_electricity_follows_the_period_weights_and_capital_the_largest_duty():
    low, high = _period(300.0, 200.0, 100.0), _period(600.0, 400.0, 200.0)
    target = SimpleNamespace(
        hpr_cycle="Cascade Carnot cycles",
        # The target's own fields hold one period: the first.
        hpr_work=100.0,
        hpr_cop=3.0,
        hpr_hot_streams=low["hpr_hot_streams"],
        hpr_cold_streams=low["hpr_cold_streams"],
        hpr_details=SimpleNamespace(period_outputs={"low": low, "high": high}),
    )

    record = heat_pump_record(
        target,
        machine="HP",
        costing=COSTING,
        cost_per_kw=200.0,
        period_weights={"low": 0.25, "high": 0.75},
    )

    # 0.25 x 100 kW + 0.75 x 200 kW, at 100 $/MWh over 8300 h.
    assert record.work == pytest.approx(175.0)
    assert record.electricity_cost == pytest.approx(175.0 * 100.0 * 8.3)
    # The machine's capacity: the largest duty of any period.
    assert (record.condenser_duty, record.evaporator_duty) == (600.0, 400.0)
    assert record.capital_cost == pytest.approx(200.0 * 600.0)
    # Heat delivered over work drawn, both weighted over the periods.
    assert record.cop == pytest.approx((0.25 * 300.0 + 0.75 * 600.0) / 175.0)
