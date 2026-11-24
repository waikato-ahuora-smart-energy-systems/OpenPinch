"""A two-period retrofit finds a vapour compression heat pump.

H1 70 to 35 degC (1500 kW) on a cooler and C1 75 to 95 degC on steam at
60 $/MWh, 1200 kW in the base and 1500 kW in the peak, electricity at
50 $/MWh, water and ammonia. The machine's condenser keeps its segments
in each period: at 0.75 of the load, stopped at 300 evaluations, it
delivers about 836 kW in the base and 1045 kW in the peak and nets about
318,000 $/y (13 s). With three sizes and 2000 evaluations it nets
349,700 $/y in 196 s.
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork
from tests.analysis.heat_exchanger_networks.heat_pump_levels_case import (
    solvers_available,
)
from tests.analysis.heat_exchanger_networks.low_temperature_retrofit_case import (
    _stream,
    _utility,
)


def _two_periods() -> PinchProblem:
    heated = _stream("C1", 75.0, 95.0, 1200.0)
    heated["heat_flow"] = {"values": [1200.0, 1500.0], "unit": "kW"}
    return PinchProblem(
        source={
            "streams": [_stream("H1", 70.0, 35.0, 1500.0), heated],
            "utilities": [
                _utility("HU", "Hot", 180.0, 179.0, 60.0),
                _utility("CU", "Cold", 20.0, 25.0, 5.0),
            ],
            "options": {
                "COSTING_HPR_ELE_PRICE": 50.0,
                "COSTING_ANNUAL_OP_TIME": 8000.0,
                "HPR_REFRIGERANTS": ["water", "ammonia"],
                "PROBLEM_PERIOD_IDS": ["base", "peak"],
                "PROBLEM_PERIOD_WEIGHTS": [0.5, 0.5],
            },
        },
        project_name="vapour compression periods",
    )


def _existing() -> HeatExchangerNetwork:
    structure = HeatExchangerNetwork.from_structure(
        recovery=[], heaters=["C1"], coolers=["H1"], hot_utility="HU", cold_utility="CU"
    )
    return structure.model_copy(
        update={
            "exchangers": tuple(
                e.model_copy(update={"area": 300.0}) for e in structure.exchangers
            )
        }
    )


@pytest.mark.synthesis
@pytest.mark.solver
@pytest.mark.skipif(not solvers_available(), reason="needs ipopt")
def test_a_two_period_retrofit_finds_a_vapour_compression_heat_pump():
    result = _two_periods().design.retrofit(
        _existing(),
        heat_pump_cycle="vapour_compression",
        max_modifications=1,
        new_unit_cost=0.0,
        heat_pump_sizes=[0.75],
        heat_pump_search={"maximum_evaluations": 300},
    )

    best = result.frontier[-1]
    assert [h.cycle for h in best.network.heat_pumps] == ["vapour_compression"]
    assert best.net_annual_benefit > 0.0
    condenser = next(
        e for e in best.network.exchangers if e.exchanger_id.startswith("new:HP:Cond")
    )
    base, peak = (state.duty for state in condenser.period_states)
    assert peak > base + 100.0
