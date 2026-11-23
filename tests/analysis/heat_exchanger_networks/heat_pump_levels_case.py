"""A retrofit case where only a partial-lift heat pump pays.

One process: H1 from 90 to 30 degC and C1 from 20 to 120 degC, 600 kW
each. The existing network recovers 360 kW in E1, heats C1 from 80 degC
with 240 kW of steam (E2) and cools H1 from 54 degC with 240 kW of
cooling water (E3). Steam costs 40 $/MWh and electricity 100 $/MWh, so a
heat pump pays above a COP of 2.5. Lifting from the cooler's cold end to
the heater's hot end gives a COP of 2.33; the as-piped target, which
places its levels along the cooler and the heater, gives 3.28.
"""

from __future__ import annotations

from types import SimpleNamespace

from OpenPinch import PinchProblem
from OpenPinch.application.case_derivation import _case_with_heat_pump_streams

EXISTING = {
    "recovery": [{"hot": "H1", "cold": "C1", "stage": 1, "area": 37.5, "id": "E1"}],
    "heaters": [{"stream": "C1", "utility": "HPS", "area": 5.0, "id": "E2"}],
    "coolers": [{"stream": "H1", "utility": "CW", "area": 20.0, "id": "E3"}],
}


def _value(value: float, unit: str) -> dict:
    return {"value": value, "unit": unit}


def _stream(name: str, supply: float, target: float) -> dict:
    return {
        "zone": "Site/P",
        "name": name,
        "t_supply": _value(supply, "degC"),
        "t_target": _value(target, "degC"),
        "heat_flow": _value(600.0, "kW"),
        "dt_cont": _value(5.0, "delta_degC"),
        "htc": _value(1.0, "kW/m^2/delta_degC"),
    }


def _utility(name: str, kind: str, supply: float, target: float, price: float) -> dict:
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


def levels_problem() -> PinchProblem:
    return PinchProblem(
        source={
            "streams": [_stream("H1", 90.0, 30.0), _stream("C1", 20.0, 120.0)],
            "utilities": [
                _utility("HPS", "Hot", 250.0, 249.9, 40.0),
                _utility("CW", "Cold", 10.0, 15.0, 5.0),
            ],
            "options": {"COSTING_HPR_ELE_PRICE": 100.0, "THERMAL_DT_CONT": 5.0},
            "zone_tree": {
                "name": "Site",
                "type": "Site",
                "children": [{"name": "P", "type": "Process Zone"}],
            },
        }
    )


def with_machine_streams(problem: PinchProblem, machine) -> PinchProblem:
    """``problem`` with the streams of ``machine``, a heat pump service
    result, at the heat pump's own contribution."""
    root = problem._require_prepared_root_zone()
    target = SimpleNamespace(
        provenance=SimpleNamespace(zone_address=root.address),
        hpr_details=SimpleNamespace(amb_streams=machine.amb_streams),
        hpr_hot_streams=machine.hpr_hot_streams,
        hpr_cold_streams=machine.hpr_cold_streams,
    )
    return _case_with_heat_pump_streams(
        problem, target, dt_cont=float(root.config.hpr.dt_cont)
    )
