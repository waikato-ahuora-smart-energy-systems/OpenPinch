"""Holding a single process's utility levels through evolution on request.

A pinch design with utility placement by temperature holds each level at
its targeted duty. The steps after it took no targets unless the task was a
site's, so they traded utility for area by price: on a four-stream case with
two levels a side, the design held 450 kW of HP steam and its evolved
networks used 496 to 588 kW.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.models.problem import (
    InternalHeatExchangerNetworkProblem,
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

# What a pinch design of the heating case holds: 40 kW of LPS between the
# two recovery matches and no HPS.
HELD = {"Hot Utility.LPS": [40.0], "Hot Utility.HPS": [0.0]}


def _level_targets(holds: str):
    internal = InternalHeatExchangerNetworkProblem(
        solver_arrays=problem_to_solver_arrays(_heating_problem(), D_TMIN),
        name="held",
        framework="ESM",
        solver="highs",
        dTmin=D_TMIN,
        stages=2,
        minimisation_goal="utility costs",
        integers=True,
        tol=1e-6,
        staged_utilities=True,
        level_holds=holds,
    )
    internal.utility_level_targets = HELD
    internal.load_model()
    return tuple(getattr(internal.case, "level_targets", None) or ())


def test_an_evolution_model_holds_its_designs_levels_on_request():
    assert _level_targets("none") == ()

    fixed = _level_targets("fixed")
    assert sorted((t.side, t.duties, t.at_least) for t in fixed) == [
        ("hot", (0.0,), False),
        ("hot", (40.0,), False),
    ]
    # Each level is held over every cold stream of the process (one here).
    assert {t.streams for t in fixed} == {(0,)}
