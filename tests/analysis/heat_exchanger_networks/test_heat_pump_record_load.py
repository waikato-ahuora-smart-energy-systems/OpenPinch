"""A heat pump's record says how much of the full load it takes.

A design can rank heat pumps of several sizes. The share of the heating
the process could give a heat pump, which this one takes, tells them
apart in the ranking without a look at the duties.
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


def _target(**load):
    return SimpleNamespace(
        hpr_cycle="Cascade Carnot cycles",
        hpr_work=176.9,
        hpr_cop=1.7,
        hpr_hot_streams={
            "HP_H1": Stream(
                name="HP_H1",
                supply_temperature=226.0,
                target_temperature=225.99,
                heat_flow=300.0,
            )
        },
        hpr_cold_streams={
            "HP_C1": Stream(
                name="HP_C1",
                supply_temperature=119.66,
                target_temperature=119.67,
                heat_flow=123.1,
            )
        },
        **load,
    )


def test_the_record_holds_the_share_of_the_available_load_the_target_took():
    part = heat_pump_record(
        _target(hpr_load=SimpleNamespace(available=750.0, selected=300.0)),
        machine="HP",
        costing=COSTING,
        cost_per_kw=0.0,
    )
    unknown = heat_pump_record(
        _target(), machine="HP", costing=COSTING, cost_per_kw=0.0
    )

    assert part.load_fraction == pytest.approx(0.4)
    assert unknown.load_fraction is None
