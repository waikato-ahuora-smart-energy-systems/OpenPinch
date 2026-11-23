"""A forbidden heater or cooler is not a feasible utility match.

Both model families, the pinch sides and the stage-wise model, take their
heater and cooler flags from one function, and the heuristic and the
evolution moves read those flags, so a pair ruled out there is ruled out
everywhere a design is built.
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


def _flags(pairs):
    """Heater flags by hot utility and cooler flags by hot stream, by name."""
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
    utilities = [
        n.rsplit(".", 1)[-1] for n in arrays.utility_identities["hot_utilities"]
    ]
    streams = [
        n.rsplit(".", 1)[-1] for n in arrays.stream_identities["hot_process_streams"]
    ]
    heaters = dict(zip(utilities, model.z_hu_feasible[0]))
    coolers = {name: row[0] for name, row in zip(streams, model.z_cu_feasible)}
    return heaters, coolers


def test_a_forbidden_heater_or_cooler_is_not_feasible():
    # C1 is heated to 200 C, which LPS at 140 C cannot reach; cooling water
    # can cool both hot streams.
    assert _flags([]) == ({"LPS": 0, "HPS": 1}, {"H1": 1, "H2": 1})

    heaters, coolers = _flags([["HPS", "C1"], ["H2", "CW"]])
    assert heaters == {"LPS": 0, "HPS": 0}
    assert coolers == {"H1": 1, "H2": 0}
