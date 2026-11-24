"""No ranked network serves a heat pump stream through more exchangers than
``HENS_HEAT_PUMP_STREAM_EXCHANGERS``, and the design says how many it left out.

Basic pinch sample, Carnot heat pump at full load, heuristic pinch design:
its networks put one or two exchangers on the condenser and on the
evaporator. A cap of 1 keeps the networks with one of each.
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from tests.application.heat_pump_stream_cases import basic_source


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


def _most_exchangers_on_a_heat_pump_stream(design) -> int:
    most = 0
    for outcome in design.top(1000):
        network = outcome.network
        if not network.heat_pumps:
            continue
        for stream in ("HP_H1", "HP_C1"):
            most = max(
                most,
                sum(
                    1
                    for exchanger in network.exchangers
                    if stream
                    in (
                        exchanger.source_stream.rsplit(".", 1)[-1],
                        exchanger.sink_stream.rsplit(".", 1)[-1],
                    )
                    and max(state.duty for state in exchanger.period_states) > 1e-6
                ),
            )
    return most


def _design(**options):
    return PinchProblem(
        source=basic_source(), project_name="basic"
    ).design.heat_exchanger_network(
        heat_pump={
            "load": {"fraction": 1.0},
            "levels": {"condensers": 1, "evaporators": 1},
            "search": {"restarts": 0, "evaluations": 1},
        },
        heat_pump_cost=200.0,
        width=0,
        depth=0,
        max_parallel=1,
        **options,
    )


def test_a_cap_of_one_keeps_only_networks_with_one_exchanger_a_stream():
    assert _most_exchangers_on_a_heat_pump_stream(_design()) == 2

    with pytest.warns(UserWarning, match="HENS_HEAT_PUMP_STREAM_EXCHANGERS"):
        capped = _design(options={"HENS_HEAT_PUMP_STREAM_EXCHANGERS": 1})

    assert _most_exchangers_on_a_heat_pump_stream(capped) == 1
