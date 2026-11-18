"""A site network: each process designed at its levels, evolved as one.

``design.total_site_heat_exchanger_network`` designs each process zone with
the pinch design method, holding its utility levels at their targets, joins
the networks into one site network per dt_cont multiplier, and evolves those
seeds as one problem: every process keeps its levels and recovers heat only
within itself. There is no thermal derivative step.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.reporting.verification import (
    verify_network_feasibility,
)
from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from OpenPinch.application.problem import PinchProblem
from OpenPinch.domain.enums import HeatExchangerKind

D_TMIN = 10.0


def _solvers_available() -> bool:
    try:
        backend.require_solver_backend("highs")
        backend.require_solver_backend("ipopt-pyomo")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


def _quantity(value, unit):
    return {"value": value, "unit": unit}


def _stream(zone, name, supply, target, heat_flow):
    return {
        "zone": f"Site/{zone}",
        "name": name,
        "t_supply": _quantity(supply, "degC"),
        "t_target": _quantity(target, "degC"),
        "heat_flow": _quantity(heat_flow, "kW"),
        "dt_cont": _quantity(D_TMIN / 2, "delta_degC"),
        "htc": _quantity(1.0, "kW/m^2/delta_degC"),
    }


def _utility(name, kind, supply, target, price):
    return {
        "name": name,
        "type": kind,
        "t_supply": _quantity(supply, "degC"),
        "t_target": _quantity(target, "degC"),
        "dt_cont": _quantity(2.5, "delta_degC"),
        "price": _quantity(price, "$/MWh"),
        "htc": _quantity(1.0, "kW/m^2/delta_degC"),
        "heat_flow": None,
    }


def _site(processes=("Heating", "Cooling")) -> PinchProblem:
    """Heating needs 40 kW from LPS between two matches; Cooling raises 40 kW
    of LP steam between two. By price HPS and cooling water would take
    both."""
    streams = {
        "Heating": [
            _stream("Heating", "H1", 220.0, 150.0, 70.0),
            _stream("Heating", "H2", 100.0, 70.0, 30.0),
            _stream("Heating", "C1", 60.0, 200.0, 140.0),
        ],
        "Cooling": [
            _stream("Cooling", "H3", 200.0, 60.0, 140.0),
            _stream("Cooling", "C2", 50.0, 120.0, 70.0),
            _stream("Cooling", "C3", 160.0, 190.0, 30.0),
        ],
    }
    return PinchProblem(
        source={
            "streams": [stream for zone in processes for stream in streams[zone]],
            "utilities": [
                _utility("LPS", "Hot", 140.0, 139.9, 60.0),
                _utility("HPS", "Hot", 250.0, 249.9, 40.0),
                _utility("LPG", "Cold", 120.0, 120.1, 20.0),
                _utility("CW", "Cold", 20.0, 25.0, 5.0),
            ],
            "zone_tree": {
                "name": "Site",
                "type": "Site",
                "children": [
                    {"name": zone, "type": "Process Zone"} for zone in processes
                ],
            },
        }
    )


def _process(stream: str) -> str:
    return stream.rsplit(".", 1)[0]


@pytest.mark.synthesis
@pytest.mark.solver
@pytest.mark.skipif(not _solvers_available(), reason="needs highspy and ipopt")
def test_a_site_network_holds_each_processs_levels_and_matches_within_it():
    problem = _site()

    view = problem.design.total_site_heat_exchanger_network(
        width=0, depth=1, options={"HENS_EVM_N_AD_BRANCHES": 1}
    )
    network = view.selected_network

    assert view.result.method == "network_evolution_method"
    recovery = [
        exchanger
        for exchanger in network.exchangers
        if exchanger.kind is HeatExchangerKind.RECOVERY
    ]
    assert recovery
    for exchanger in recovery:
        assert _process(exchanger.source_stream) == _process(exchanger.sink_stream)
    duties: dict[tuple[str, str], float] = {}
    for exchanger in network.exchangers:
        if exchanger.kind is HeatExchangerKind.HOT_UTILITY:
            key = (_process(exchanger.sink_stream), exchanger.source_stream)
        elif exchanger.kind is HeatExchangerKind.COLD_UTILITY:
            key = (_process(exchanger.source_stream), exchanger.sink_stream)
        else:
            continue
        duties[key] = duties.get(key, 0.0) + exchanger.state().duty
    heating = next(p for p, _ in duties if "Heating" in p)
    cooling = next(p for p, _ in duties if "Cooling" in p)
    assert duties.get((heating, "Hot Utility.LPS"), 0.0) == pytest.approx(
        40.0, rel=1e-3
    )
    assert duties.get((cooling, "Cold Utility.LPG"), 0.0) == pytest.approx(
        40.0, rel=1e-3
    )
    assert verify_network_feasibility(network) == ()


def test_a_site_network_needs_more_than_one_process():
    with pytest.raises(ValueError, match="more than one process zone"):
        _site(("Heating",)).design.total_site_heat_exchanger_network()
