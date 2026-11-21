"""Wall-clock budget of heuristic-only pinch design on a large sample case."""

from __future__ import annotations

import json
import time

import pytest

from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from OpenPinch.application.problem import PinchProblem
from tests.support.paths import REPOSITORY_ROOT

# Wall-clock budgets: the performance lane runs these, not the unit lanes.
pytestmark = pytest.mark.performance

PULP_MILL_JSON = REPOSITORY_ROOT / "OpenPinch/tutorials/sample_cases/pulp_mill.json"


def _highs_available() -> bool:
    try:
        backend.require_solver_backend("highs")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_heuristic_pinch_design_of_the_pulp_mill_finishes_in_ten_minutes():
    # Width 0, depth 0 took 1,267 s: 50 proposals per side, the below side
    # trying each with its utility levels and then without. The scaled count
    # (10 for 64 streams) and the stage's time limit bound it.
    problem = PinchProblem(
        source=json.loads(PULP_MILL_JSON.read_text(encoding="utf-8")),
        project_name="pulp_mill",
    )

    start = time.monotonic()
    view = problem.design.pinch_design(
        options={"HENS_SEARCH_WIDTH": 0, "HENS_SEARCH_DEPTH": 0}
    )
    elapsed = time.monotonic() - start

    assert elapsed < 600.0
    assert view.top(100)
