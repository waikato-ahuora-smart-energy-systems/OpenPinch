"""Putting a heat pump on a network adds its cost to the network's total.

Synthesis designs the heat pump's streams and prices the exchangers and
utilities. The machine's own cost is a constant of the design, added once
the network is final, so that a ranking by total annual cost counts it.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.heat_pump_cost import (
    network_with_heat_pumps,
)
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


def test_the_networks_total_rises_by_the_heat_pumps_annual_cost():
    designed = HeatExchangerNetwork(
        total_annual_cost=168_600.0, utility_cost=42_600.0, capital_cost=126_000.0
    )

    priced = network_with_heat_pumps(designed, (HEAT_PUMP,))

    assert priced.heat_pumps == (HEAT_PUMP,)
    assert priced.total_annual_cost == pytest.approx(168_600.0 + 228_061.0)
    # The utility cost includes the machine's electricity, as a retrofit's
    # designs do (decided 2026-11-27); the capital stays the exchangers'.
    assert priced.utility_cost == pytest.approx(42_600.0 + 218_622.0)
    assert priced.capital_cost == 126_000.0
    assert _check_cost_consistency(priced) == []
    assert designed.total_annual_cost == 168_600.0
