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


def test_a_vapour_compression_machine_costs_all_but_its_process_exchangers():
    # The HPR service prices its compressors, motors and internal (cascade,
    # IHX) exchangers; the condensers and evaporators are the network's
    # exchangers, priced by area there.
    costing = Configuration().costing
    solved = SimpleNamespace(
        hpr_hot_streams={"a": _stream("HP_H1", 100.0)},
        hpr_cold_streams={"c": _stream("HP_C1", 70.0)},
        w_net=30.0,
        cop_h=100.0 / 30.0,
        hpr_machine_internal_capital_costs=(40_000.0, 15_000.0),
    )

    machine = heat_pump_machine(
        solved,
        name="HP",
        cycle="vapour_compression",
        cost_per_kw=300.0,
        costing=costing,
    )

    assert machine.cycle == "vapour_compression"
    assert machine.capital_cost == pytest.approx(55_000.0)


def test_a_machine_of_several_periods_is_sized_on_its_largest_and_run_on_each():
    # A shared design: the solve's own fields hold one period, its period
    # outputs every one. The machine is sized for the period it delivers
    # most in and draws each period's work for that period's share.
    costing = Configuration().costing
    solved = SimpleNamespace(
        hpr_hot_streams={"a": _stream("HP_H1", 60.0)},
        hpr_cold_streams={"c": _stream("HP_C1", 40.0)},
        w_net=20.0,
        cop_h=3.0,
        period_outputs={
            "base": {
                "hpr_hot_streams": {"a": _stream("HP_H1", 60.0)},
                "hpr_cold_streams": {"c": _stream("HP_C1", 40.0)},
                "w_net": 20.0,
            },
            "peak": {
                "hpr_hot_streams": {"a": _stream("HP_H1", 100.0)},
                "hpr_cold_streams": {"c": _stream("HP_C1", 60.0)},
                "w_net": 40.0,
            },
        },
    )

    machine = heat_pump_machine(
        solved,
        name="HP",
        cycle="carnot",
        cost_per_kw=300.0,
        costing=costing,
        period_weights={"base": 0.25, "peak": 0.75},
    )

    assert (machine.condenser_duty, machine.evaporator_duty) == pytest.approx(
        (100.0, 60.0)
    )
    assert machine.work == pytest.approx(0.25 * 20.0 + 0.75 * 40.0)
    assert machine.cop == pytest.approx((0.25 * 60.0 + 0.75 * 100.0) / 35.0)
    assert machine.capital_cost == pytest.approx(300.0 * 100.0)


def test_a_vapour_compression_machine_of_several_periods_is_installed_for_each():
    # Each period's solve installs its own compressors and internal
    # exchangers; the machine is the largest of each across the periods,
    # not the selected period's alone.
    costing = Configuration().costing

    def output(duty: float, internal: tuple) -> dict:
        return {
            "hpr_hot_streams": {"a": _stream("HP_H1", duty)},
            "hpr_cold_streams": {"c": _stream("HP_C1", 0.7 * duty)},
            "w_net": 0.3 * duty,
            "hpr_machine_internal_capital_costs": internal,
        }

    solved = SimpleNamespace(
        hpr_hot_streams={"a": _stream("HP_H1", 100.0)},
        hpr_cold_streams={"c": _stream("HP_C1", 70.0)},
        w_net=30.0,
        cop_h=100.0 / 30.0,
        hpr_machine_internal_capital_costs=(40_000.0, 15_000.0),
        period_outputs={
            "base": output(100.0, (40_000.0, 15_000.0)),
            "peak": output(120.0, (50_000.0, 10_000.0)),
        },
    )

    machine = heat_pump_machine(
        solved,
        name="HP",
        cycle="vapour_compression",
        cost_per_kw=300.0,
        costing=costing,
        period_weights={"base": 0.5, "peak": 0.5},
    )

    assert machine.capital_cost == pytest.approx(50_000.0 + 15_000.0)
