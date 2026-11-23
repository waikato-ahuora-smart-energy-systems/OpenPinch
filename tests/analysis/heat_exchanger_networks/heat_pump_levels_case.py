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

from dataclasses import replace
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


def solvers_available() -> bool:
    from OpenPinch.analysis.heat_exchanger_networks.solver import backend
    from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
        MissingSynthesisDependencyError,
        MissingSynthesisSolverError,
    )

    try:
        backend.require_solver_backend("ipopt-pyomo")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


def heat_pump_search(*, max_modifications: int = 1, **costs) -> SimpleNamespace:
    """A retrofit search of the levels case started from its as-piped heat
    pump, with the base design it is measured against.

    The heat pump search stops at 200 evaluations (0.4 s; a full one takes
    64 s and finds 186.5 kW delivered at a COP of 3.28), which gives a
    machine of three condensers and two evaporators delivering 156 kW.
    ``costs`` are the machine's ``electricity_cost`` ($/y), ``capital_cost``
    ($) and ``annual_capital_cost`` ($/y), each 0 unless given.
    """
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pump_streams import (
        HeatPumpStep,
    )
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pump_targets import (
        as_piped_heat_pump,
    )
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
        RetrofitSettings,
        run_retrofit_search,
    )
    from OpenPinch.domain.heat_exchanger_network import NetworkHeatPump

    problem = levels_problem()
    existing = problem.design._network_from_structure(EXISTING)
    settings = RetrofitSettings(
        max_modifications=max_modifications, approach=None, new_unit_cost=0.0
    )
    base = run_retrofit_search(
        problem, existing, replace(settings, max_modifications=0)
    ).frontier[0]
    machine = as_piped_heat_pump(
        problem, base.network, args={"maximum_evaluations": 200}
    )

    def duty(streams) -> float:
        return sum(float(stream.heat_flow.value) for _key, stream in streams.items())

    record = NetworkHeatPump(
        machine="HP",
        cycle="carnot",
        condenser_duty=duty(machine.hpr_hot_streams),
        evaporator_duty=duty(machine.hpr_cold_streams),
        work=float(machine.w_net),
        cop=float(machine.cop_h),
        **(
            {"electricity_cost": 0.0, "capital_cost": 0.0, "annual_capital_cost": 0.0}
            | costs
        ),
    )
    step = HeatPumpStep(
        record=record,
        condensers=tuple(s for _key, s in machine.hpr_hot_streams.items()),
        evaporators=tuple(s for _key, s in machine.hpr_cold_streams.items()),
        base=base,
    )
    result = run_retrofit_search(
        with_machine_streams(problem, machine), existing, settings, heat_pump=step
    )
    return SimpleNamespace(base=base, step=step, record=record, result=result)


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
