"""``zone=`` and ``heat_pump=`` together design one zone with its own heat pump.

The heat pump is targeted on the zone's streams, not the whole problem's,
and its networks serve that zone alone, as a design of the zone without a
heat pump does.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from tests.application.heat_pump_stream_cases import two_zone_problem


def _highs_available() -> bool:
    try:
        backend.require_solver_backend("highs")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


pytestmark = [
    pytest.mark.synthesis,
    pytest.mark.skipif(not _highs_available(), reason="needs highspy"),
]

DESIGN = {"zone": "A", "width": 0, "depth": 0, "max_parallel": 1}


def _served(design) -> set[str]:
    """Every stream and utility the design's networks have an exchanger on."""

    return {
        name
        for outcome in design.top(1000)
        for exchanger in outcome.network.exchangers
        for name in (exchanger.source_stream, exchanger.sink_stream)
    }


def test_a_zone_is_designed_with_a_heat_pump_targeted_on_its_own_streams():
    plain = two_zone_problem().design.heat_exchanger_network(**DESIGN)

    design = two_zone_problem().design.heat_exchanger_network(
        heat_pump={"search": {"restarts": 0, "evaluations": 1}},
        heat_pump_cost=200.0,
        **DESIGN,
    )

    assert any(outcome.network.heat_pumps for outcome in design.top(1000))
    # Beyond the zone's own streams and utilities, only the heat pump's.
    added = {name.rsplit(".", 1)[-1] for name in _served(design) - _served(plain)}
    assert added and added <= {"HP_H1", "HP_C1", "AIR_H1"}
