"""A heat pump target is priced as a machine on a network.

Electricity is the work at the electricity price over the operating
hours. The compressor's capital is a cost per kW of condenser capacity,
annualised like any other capital. The condenser and the evaporator are
exchangers of the network and are priced there, not here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch.analysis.heat_exchanger_networks.heat_pump_cost import (
    heat_pump_record,
)
from OpenPinch.domain.stream import Stream

COSTING = SimpleNamespace(
    hpr_ele_price=100.0,  # $/MWh
    annual_op_time=8300.0,  # h/y
    discount_rate=0.07,
    service_life=20.0,
)


def _target():
    """The basic pinch case's Carnot heat pump target at full load."""

    return SimpleNamespace(
        hpr_cycle="Cascade Carnot cycles",
        hpr_work=263.4,
        hpr_cop=2.847,
        hpr_hot_streams={
            "HP_H1": Stream(
                name="HP_H1",
                supply_temperature=226.0,
                target_temperature=225.99,
                heat_flow=750.0,
            )
        },
        hpr_cold_streams={
            "HP_C1": Stream(
                name="HP_C1",
                supply_temperature=119.66,
                target_temperature=119.67,
                heat_flow=486.6,
            )
        },
    )


def test_a_target_is_priced_at_its_work_and_its_condenser_capacity():
    record = heat_pump_record(
        _target(), machine="HP", costing=COSTING, cost_per_kw=200.0
    )

    assert (record.machine, record.cycle) == ("HP", "Cascade Carnot cycles")
    assert (record.condenser_duty, record.evaporator_duty) == (750.0, 486.6)
    assert (record.work, record.cop) == (263.4, 2.847)
    # 263.4 kW x 100 $/MWh x 8300 h.
    assert record.electricity_cost == pytest.approx(218_622.0)
    # 200 $/kW of 750 kW, at a capital recovery factor of 0.09439.
    assert record.capital_cost == pytest.approx(150_000.0)
    assert record.annual_capital_cost == pytest.approx(14_158.9, abs=0.1)
