"""A heat pump ranked with a free compressor says so.

A Carnot target has no machine, so it has no capital: its compressor is
priced by ``heat_pump_cost`` ($ per kW of condenser capacity). Left at 0,
the heat pump designs are ranked on electricity alone, which flatters
them; the design warns, as ``design.retrofit`` does.
"""

from __future__ import annotations

import warnings

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


def _design(**cost):
    return PinchProblem(
        source=basic_source(), project_name="basic"
    ).design.heat_exchanger_network(
        heat_pump={"search": {"restarts": 0, "evaluations": 1}},
        width=0,
        depth=0,
        max_parallel=1,
        **cost,
    )


def test_a_carnot_heat_pump_at_no_cost_per_kw_warns_that_its_compressor_is_free():
    with pytest.warns(UserWarning, match="heat_pump_cost.*compressor"):
        _design()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _design(heat_pump_cost=200.0)
    assert not [w for w in caught if "heat_pump_cost" in str(w.message)]
