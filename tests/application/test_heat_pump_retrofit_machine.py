"""The retrofit's heat pump is priced as a machine of fixed size.

The as-piped target gives the machine's streams and work. Its record
for the search takes the condenser capacity and the evaporator duty from
the streams, the electricity from the work at ``COSTING_HPR_ELE_PRICE``
over the year's operating hours, and the compressor from
``heat_pump_cost`` per kW of condenser capacity, annualised as the
exchangers are.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from OpenPinch.analysis.economics import compute_capital_recovery_factor
from OpenPinch.application.heat_pump_retrofit import heat_pump_machine
from OpenPinch.domain.configuration import Configuration
from OpenPinch.domain.stream import Stream


def _stream(name: str, duty: float) -> Stream:
    return Stream(
        name=name, supply_temperature=100.0, target_temperature=99.99, heat_flow=duty
    )


def test_the_machine_holds_its_duties_electricity_and_compressor_cost():
    config = Configuration()
    config.update_values(COSTING_HPR_ELE_PRICE=100.0)
    costing = config.costing
    solved = SimpleNamespace(
        hpr_hot_streams={"a": _stream("HP_H1", 60.0), "b": _stream("HP_H2", 40.0)},
        hpr_cold_streams={"c": _stream("HP_C1", 70.0)},
        w_net=30.0,
        cop_h=100.0 / 30.0,
    )

    machine = heat_pump_machine(
        solved, name="HP", cycle="carnot", cost_per_kw=300.0, costing=costing
    )

    assert (machine.machine, machine.cycle) == ("HP", "carnot")
    assert machine.condenser_duty == pytest.approx(100.0)
    assert machine.evaporator_duty == pytest.approx(70.0)
    assert (machine.work, machine.cop) == pytest.approx((30.0, 100.0 / 30.0))
    # 30 kW at 100 $/MWh over the year's operating hours.
    assert machine.electricity_cost == pytest.approx(
        30.0 * 100.0 * float(costing.annual_op_time) / 1e3
    )
    assert machine.capital_cost == pytest.approx(300.0 * 100.0)
    assert machine.annual_capital_cost == pytest.approx(
        30_000.0
        * compute_capital_recovery_factor(costing.discount_rate, costing.service_life)
    )
