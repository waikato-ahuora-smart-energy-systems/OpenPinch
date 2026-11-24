"""A site network's heat pump joins the site cost it ranks by.

A site network ranks by ``summary_metrics["site_total_annual_cost"]``,
the site paying only for its net utility use. A heat pump recorded on it
must add its electricity and annualised capital there too, or a site
design with the machine would rank as if the machine were free.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.heat_pump_cost import (
    network_with_heat_pumps,
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


def test_a_site_networks_heat_pump_adds_to_its_site_cost():
    network = HeatExchangerNetwork(
        total_annual_cost=500_000.0,
        summary_metrics={"site_total_annual_cost": 400_000.0},
    )

    priced = network_with_heat_pumps(network, (HEAT_PUMP,))

    assert priced.summary_metrics["site_total_annual_cost"] == pytest.approx(
        400_000.0 + 218_622.0 + 9_439.0
    )
