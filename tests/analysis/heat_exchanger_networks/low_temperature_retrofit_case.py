"""A low-temperature retrofit a water vapour compression heat pump can serve.

One hot stream H1 (95 to 45 degC, 2000 kW) on a cooler and one cold stream
C1 (100 to 130 degC, 1500 kW) on steam at 60 $/MWh, with no recovery in
the existing network; electricity at 50 $/MWh. The heat pump target has
one condenser and one evaporator level and water alone, since ammonia
cannot condense at the 135 degC the heater needs: a single-stage water
machine taking heat from the cooler to the bottom of the heater.
"""

from __future__ import annotations

from OpenPinch import PinchProblem
from OpenPinch.domain._heat_exchanger.period_state import HeatExchangerPeriodState
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork

KELVIN = 273.15


def _degc(value: float) -> dict:
    return {"value": value, "unit": "degC"}


def _stream(name: str, supply: float, target: float, duty: float) -> dict:
    return {
        "zone": "Plant",
        "name": name,
        "t_supply": _degc(supply),
        "t_target": _degc(target),
        "heat_flow": {"value": duty, "unit": "kW"},
        "dt_cont": {"value": 5, "unit": "degC"},
        "htc": {"value": 1, "unit": "kW/m^2/degC"},
    }


def _utility(name: str, kind: str, supply: float, target: float, price: float):
    return {
        "name": name,
        "type": kind,
        "t_supply": _degc(supply),
        "t_target": _degc(target),
        "dt_cont": {"value": 5, "unit": "degC"},
        "price": {"value": price, "unit": "$/MWh"},
        "htc": {"value": 1, "unit": "kW/m^2/degC"},
        "heat_flow": None,
    }


def low_temperature_problem() -> PinchProblem:
    from OpenPinch.analysis.heat_exchanger_networks.context import (
        ensure_target_results,
    )

    problem = PinchProblem(
        source={
            "streams": [
                _stream("H1", 95.0, 45.0, 2000.0),
                _stream("C1", 100.0, 130.0, 1500.0),
            ],
            "utilities": [
                _utility("HU", "Hot", 180.0, 179.0, 60.0),
                _utility("CU", "Cold", 20.0, 25.0, 5.0),
            ],
            "options": {
                "COSTING_HPR_ELE_PRICE": 50.0,
                "COSTING_ANNUAL_OP_TIME": 8000.0,
                "HPR_N_COND": 1,
                "HPR_N_EVAP": 1,
                "HPR_REFRIGERANTS": ["water"],
            },
        },
        project_name="low-temperature",
    )
    ensure_target_results(problem, {}, None)
    return problem


def existing_network() -> HeatExchangerNetwork:
    """The existing network as built: a heater on C1, a cooler on H1."""

    structure = HeatExchangerNetwork.from_structure(
        recovery=[], heaters=["C1"], coolers=["H1"], hot_utility="HU", cold_utility="CU"
    )
    return structure.model_copy(
        update={
            "exchangers": tuple(
                exchanger.model_copy(update={"area": 100.0})
                for exchanger in structure.exchangers
            )
        }
    )


def solved_network() -> HeatExchangerNetwork:
    """The existing network with its solved states (temperatures in K)."""

    states = {
        "cold-utility:H1->CU": (2000.0, 95.0, 45.0, 20.0, 25.0),
        "hot-utility:HU->C1": (1500.0, 180.0, 179.0, 100.0, 130.0),
    }

    def solved(exchanger):
        duty, *temperatures = states[exchanger.exchanger_id]
        source_in, source_out, sink_in, sink_out = (t + KELVIN for t in temperatures)
        state = HeatExchangerPeriodState(
            period_id="0",
            period_idx=0,
            duty=duty,
            source_inlet_temperature=source_in,
            source_outlet_temperature=source_out,
            sink_inlet_temperature=sink_in,
            sink_outlet_temperature=sink_out,
        )
        return exchanger.model_copy(update={"period_states": (state,)})

    network = existing_network()
    return network.model_copy(
        update={"exchangers": tuple(solved(e) for e in network.exchangers)}
    )
