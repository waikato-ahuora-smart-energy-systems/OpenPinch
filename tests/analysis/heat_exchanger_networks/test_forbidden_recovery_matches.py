"""A forbidden recovery match is not a feasible match in the stage-wise model.

The thermal derivative and network evolution steps, and the candidates an
evolution move may add, read the model's ``z_feasible``.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.models.stagewise import (
    StageWiseModel,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    problem_to_solver_arrays,
)
from tests.analysis.heat_exchanger_networks.test_staged_utilities import (
    D_TMIN,
    _heating_problem,
    _highs_available,
)

pytestmark = pytest.mark.skipif(
    not _highs_available(), reason="needs highspy and pyomo"
)


def _feasible(pairs) -> dict[str, list[int]]:
    """Each hot stream's flags against C1, by stage."""
    problem = _heating_problem()
    problem.update_options({"HENS_FORBIDDEN_MATCHES": pairs})
    arrays = problem_to_solver_arrays(problem, D_TMIN)
    model = StageWiseModel(
        name="forbidden",
        framework="ESM",
        solver="highs",
        solver_arrays=arrays,
        stages=2,
        dTmin=D_TMIN,
        z_restriction=None,
        min_dqda=0.0,
        minimisation_goal="utility costs",
        non_isothermal_model=False,
        integers=True,
        tol=1e-6,
    )
    streams = [
        n.rsplit(".", 1)[-1] for n in arrays.stream_identities["hot_process_streams"]
    ]
    return {name: list(row[0]) for name, row in zip(streams, model.z_feasible)}


def test_a_forbidden_recovery_match_is_feasible_in_no_stage():
    assert _feasible([]) == {"H1": [1, 1], "H2": [1, 1]}
    assert _feasible([["H1", "C1"]]) == {"H1": [0, 0], "H2": [1, 1]}
