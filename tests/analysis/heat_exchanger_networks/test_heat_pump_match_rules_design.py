"""No designed network breaks the heat pump match rules, even when tempted.

With free utilities a design saves capital by dropping recovery: network
evolution then puts a heater on the evaporator (486.6 kW) and a cooler on
the condenser (750 kW), each one cheap exchanger in place of several, and
calls the heat pump's streams served. The rules keep both streams on the
process in every ranked network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import OpenPinch
from OpenPinch import PinchProblem
from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from OpenPinch.domain.enums import HeatExchangerKind

BASIC_PINCH = (
    Path(OpenPinch.__file__).parent / "tutorials/sample_cases/basic_pinch.json"
)
# ``target.carnot_heat_pump(load_fraction=1.0, condensers=1, evaporators=1)``
# on the basic pinch case: (name, role, supply degC, target degC, kW).
HEAT_PUMP = [
    ("HP_H1", "condenser", 225.999999, 225.989999, 749.997934),
    ("HP_C1", "evaporator", 119.660928, 119.670928, 486.58145),
]

pytestmark = pytest.mark.synthesis


def _available(solver: str) -> bool:
    try:
        backend.require_solver_backend(solver)
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


def _free_utility_case() -> dict:
    source = json.loads(BASIC_PINCH.read_text(encoding="utf-8"))
    for utility in source["utilities"]:
        utility["price"] = {"value": 0.0, "unit": "$/MWh"}
    for name, role, supply, target, duty in HEAT_PUMP:
        source["streams"].append(
            {
                "zone": "Plant",
                "name": name,
                "t_supply": {"value": supply, "unit": "degC"},
                "t_target": {"value": target, "unit": "degC"},
                "heat_flow": {"value": duty, "unit": "kW"},
                "dt_cont": {"value": 1.0, "unit": "delta_degC"},
                "heat_pump": {"machine": "HP", "role": role},
            }
        )
    return source


def _utility_units(network, stream: str, kind: HeatExchangerKind) -> list[float]:
    """Duties (kW) of the heaters or coolers on the stream named ``stream``."""

    return [
        max(state.duty for state in exchanger.period_states)
        for exchanger in network.exchangers
        if exchanger.kind is kind
        and stream
        in (
            exchanger.source_stream.rsplit(".", 1)[-1],
            exchanger.sink_stream.rsplit(".", 1)[-1],
        )
    ]


@pytest.mark.solver
@pytest.mark.skipif(
    not (_available("highs") and _available("ipopt-pyomo")),
    reason="needs HiGHS and Ipopt",
)
def test_no_ranked_network_cools_the_condenser_or_heats_the_evaporator():
    design = PinchProblem(
        source=_free_utility_case(), project_name="free"
    ).design.heat_exchanger_network(width=0, depth=1, max_parallel=1)

    ranked = design.top(100)
    assert any(o.method.value == "network_evolution_method" for o in ranked)
    for outcome in ranked:
        network = outcome.network
        assert _utility_units(network, "HP_H1", HeatExchangerKind.COLD_UTILITY) == []
        assert _utility_units(network, "HP_C1", HeatExchangerKind.HOT_UTILITY) == []
