"""A two-period retrofit finds a heat pump that draws on air.

H1 60 to 30 degC (200 kW) on cooling water and C1 40 to 90 degC on steam
at 60 $/MWh, 800 kW in one period and 1000 kW in the other, electricity
at 30 $/MWh: the process has too little heat below the heater, so the
machine's evaporator takes the rest from air, more in the second period.
At full load (300 evaluations) the air match carries about 311 and
437 kW, the steam is gone in both periods and the design nets about
325,000 $/y (2 s).
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork
from tests.analysis.heat_exchanger_networks.heat_pump_levels_case import (
    solvers_available,
)


def _value(value: float, unit: str) -> dict:
    return {"value": value, "unit": unit}


def _stream(name: str, supply: float, target: float, heat_flow) -> dict:
    return {
        "zone": "Plant",
        "name": name,
        "t_supply": _value(supply, "degC"),
        "t_target": _value(target, "degC"),
        "heat_flow": heat_flow,
        "dt_cont": _value(5.0, "delta_degC"),
        "htc": _value(1.0, "kW/m^2/delta_degC"),
    }


def _utility(name: str, kind: str, supply: float, target: float, price: float):
    return {
        "name": name,
        "type": kind,
        "t_supply": _value(supply, "degC"),
        "t_target": _value(target, "degC"),
        "dt_cont": _value(2.5, "delta_degC"),
        "price": _value(price, "$/MWh"),
        "htc": _value(1.0, "kW/m^2/delta_degC"),
        "heat_flow": None,
    }


def _short_of_heat_in_two_periods() -> PinchProblem:
    return PinchProblem(
        source={
            "streams": [
                _stream("C1", 40.0, 90.0, {"values": [800.0, 1000.0], "unit": "kW"}),
                _stream("H1", 60.0, 30.0, _value(200.0, "kW")),
            ],
            "utilities": [
                _utility("Steam", "Hot", 150.0, 149.0, 60.0),
                _utility("CW", "Cold", 10.0, 15.0, 5.0),
            ],
            "options": {
                "COSTING_HPR_ELE_PRICE": 30.0,
                "PROBLEM_PERIOD_IDS": ["low", "high"],
                "PROBLEM_PERIOD_WEIGHTS": [0.5, 0.5],
            },
        },
        project_name="air periods retrofit",
    )


def _existing() -> HeatExchangerNetwork:
    structure = HeatExchangerNetwork.from_structure(
        recovery=[],
        heaters=["C1"],
        coolers=["H1"],
        hot_utility="Steam",
        cold_utility="CW",
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
def test_a_two_period_retrofit_draws_on_air_in_each_period():
    result = _short_of_heat_in_two_periods().design.retrofit(
        _existing(),
        heat_pump_cycle="carnot",
        heat_pump_cost=100.0,
        max_modifications=1,
        new_unit_cost=0.0,
        heat_pump_sizes=[1.0],
        heat_pump_search={"maximum_evaluations": 300},
    )

    best = result.frontier[-1]
    assert [h.machine for h in best.network.heat_pumps] == ["HP"]
    assert best.net_annual_benefit > 0.0
    air = [
        e for e in best.network.exchangers if e.exchanger_id.startswith("new:HP:AIR")
    ]
    low, high = (sum(e.period_states[n].duty for e in air) for n in (0, 1))
    assert high > low > 100.0
