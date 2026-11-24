"""A network records the heat pumps designed into it.

A heat pump enters a design as streams of fixed duty, so the network's
exchangers and utilities say nothing of the machine itself. The record
holds what the streams do not: the work it draws, its electricity cost and
the capital of its compressor.
"""

from __future__ import annotations

import pytest

from OpenPinch.domain.heat_exchanger_network import (
    HeatExchangerNetwork,
    NetworkHeatPump,
)


def _heat_pump(**changes) -> NetworkHeatPump:
    return NetworkHeatPump(
        **{
            "machine": "HP",
            "cycle": "carnot",
            "condenser_duty": 750.0,
            "evaporator_duty": 486.6,
            "work": 263.4,
            "cop": 2.847,
            "electricity_cost": 218_622.0,
            "capital_cost": 100_000.0,
            "annual_capital_cost": 9_439.0,
        }
        | changes
    )


def test_a_network_keeps_its_heat_pumps_through_a_dump_and_reload():
    network = HeatExchangerNetwork(heat_pumps=(_heat_pump(),))

    again = HeatExchangerNetwork.model_validate(network.model_dump(mode="json"))

    assert again.heat_pumps == (_heat_pump(),)
    # Electricity and annualised capital, $/y.
    assert again.heat_pumps[0].annual_cost == pytest.approx(228_061.0)
    assert HeatExchangerNetwork().heat_pumps == ()
