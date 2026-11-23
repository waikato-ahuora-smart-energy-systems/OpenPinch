"""The basic pinch sample designed with a water vapour compression heat pump.

A vapour compression condenser is one stream in segments: desuperheating
from 224.75 degC, then condensing at 174.75 degC. At a quarter load the
design used to put the whole condenser on a cold stream that, past the
desuperheat, ran hotter than the condensing plateau: a crossing inside
the match that a check of its ends missed, so no network was designed.
The plan's row 6.7: a network is designed, and its condenser's
desuperheat is matched (a recovery exchanger takes the condenser from its
inlet).
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from OpenPinch.domain.enums import HeatExchangerKind
from tests.application.heat_pump_stream_cases import basic_source

KELVIN = 273.15


def _highs_available() -> bool:
    try:
        backend.require_solver_backend("highs")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


@pytest.mark.synthesis
@pytest.mark.solver
@pytest.mark.skipif(not _highs_available(), reason="needs highspy")
def test_a_water_vapour_compression_heat_pump_designs_with_its_desuperheat_matched():
    design = PinchProblem(
        source=basic_source(), project_name="basic"
    ).design.heat_exchanger_network(
        heat_pump={
            "cycle": "vapour_compression",
            "load": {"fraction": 0.25},
            "levels": {"condensers": 1, "evaporators": 1},
            "search": {"restarts": 0, "evaluations": 1},
            "cycle_options": {"refrigerants": ["water"]},
        },
        heat_pump_cost=200.0,
        width=0,
        depth=0,
        max_parallel=1,
    )

    pumped = [o.network for o in design.top(1000) if o.network.heat_pumps]
    assert pumped
    network = pumped[0]
    inlets = [
        float(x.period_states[0].source_inlet_temperature) - KELVIN
        for x in network.exchangers
        if x.kind is HeatExchangerKind.RECOVERY
        and x.source_stream.endswith("Condenser")
    ]
    assert inlets
    assert max(inlets) == pytest.approx(224.75, abs=0.1)
