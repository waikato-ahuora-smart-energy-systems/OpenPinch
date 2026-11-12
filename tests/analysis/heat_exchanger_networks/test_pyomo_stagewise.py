"""The stage-wise model on native Pyomo, against its GEKKO conversion.

Couenne and Ipopt solve the stage-wise models (TDM, ESM and stage reduction)
as native Pyomo models. While GEKKO remains for APOPT and the fixed-structure
model, the same equations built through GEKKO's Pyomo conversion are the
reference: both must reach the same solution.
"""

from __future__ import annotations

import json

import pytest

from OpenPinch.analysis.heat_exchanger_networks.models._base import (
    execution as base_execution,
)
from OpenPinch.analysis.heat_exchanger_networks.models.stagewise import (
    StageWiseModel,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    problem_to_solver_arrays,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
    require_solver_binary,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.pyomo_model import (
    PyomoModel,
)
from OpenPinch.application.problem import PinchProblem
from tests.support.paths import REPOSITORY_ROOT

FOUR_STREAM_JSON = (
    REPOSITORY_ROOT
    / "tests/fixtures/openhens/Four-stream-Yee-and-Grossmann-1990-1.json"
)


def _binary_available(name: str) -> bool:
    try:
        require_solver_binary(name, purpose="Pyomo stage-wise parity")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


class _GekkoStageWiseModel(StageWiseModel):
    """The stage-wise model built through GEKKO and its Pyomo conversion."""

    def setup_model(self) -> None:
        base_execution.setup_model(self)


def _four_stream_problem() -> PinchProblem:
    payload = json.loads(FOUR_STREAM_JSON.read_text(encoding="utf-8"))
    return PinchProblem(source=payload, project_name="Four-stream")


def _tdm(model_cls, *, stages: int, solver: str = "couenne"):
    problem = _four_stream_problem()
    return model_cls(
        name=f"TDM {stages}",
        framework="TDM",
        solver=solver,
        solver_arrays=problem_to_solver_arrays(problem, 14.0),
        stages=stages,
        dTmin=14.0,
        z_restriction=None,
        min_dqda=0.0,
        minimisation_goal="total cost",
        non_isothermal_model=False,
        integers=True,
        tol=1e-3,
    )


@pytest.mark.solver
# A string condition is evaluated when the test runs, so collecting this module
# does not import the solver packages.
@pytest.mark.skipif("not _binary_available('couenne')", reason="needs couenne")
def test_couenne_stagewise_is_a_native_pyomo_model_with_one_objective():
    import pyomo.environ as pyo

    model = _tdm(StageWiseModel, stages=1)

    assert isinstance(model.m, PyomoModel)
    assert len(list(model.m.model.component_data_objects(pyo.Objective))) == 1
