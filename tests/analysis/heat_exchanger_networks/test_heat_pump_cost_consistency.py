"""A network's total annual cost includes its heat pumps.

Verification holds a network's total to its parts. With a heat pump the
parts are three: the utilities (the machine's electricity included, decided
2026-11-27), the exchangers' capital, and the machine's annualised capital.
A total without the machine's cost would rank a heat pump design as if its
electricity were free.
"""

from __future__ import annotations

from OpenPinch.analysis.heat_exchanger_networks.reporting.verification import (
    _check_cost_consistency,
)
from OpenPinch.domain.heat_exchanger_network import (
    HeatExchangerNetwork,
    NetworkHeatPump,
)

HEAT_PUMP = NetworkHeatPump(
    machine="HP",
    cycle="carnot",
    condenser_duty=750.0,
    evaporator_duty=486.6,
    work=263.4,
    cop=2.847,
    electricity_cost=218_622.0,
    capital_cost=100_000.0,
    annual_capital_cost=9_439.0,
)


def _network(total: float) -> HeatExchangerNetwork:
    return HeatExchangerNetwork(
        total_annual_cost=total,
        utility_cost=42_600.0 + 218_622.0,
        capital_cost=126_000.0,
        heat_pumps=(HEAT_PUMP,),
    )


def test_the_total_is_utilities_capital_and_the_heat_pumps():
    assert _check_cost_consistency(_network(42_600.0 + 126_000.0 + 228_061.0)) == []


def test_a_total_that_leaves_the_heat_pump_out_is_inconsistent():
    failures = _check_cost_consistency(_network(42_600.0 + 126_000.0))

    assert len(failures) == 1
    assert "heat pump" in failures[0]
