"""A forbidden recovery match is feasible on neither side of the pinch.

The MILP pinch design and the heuristic both build their networks from the
pinch sides' ``z_feasible``.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.models.pinch_decomposition import (
    PinchDecompModel,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    problem_to_solver_arrays,
)
from tests.analysis.heat_exchanger_networks.test_pinch_design_method import (
    _four_stream_problem,
    _pdm_decompositions,
)
from tests.analysis.heat_exchanger_networks.test_staged_utilities import (
    _highs_available,
)

pytestmark = pytest.mark.skipif(
    not _highs_available(), reason="needs highspy and pyomo"
)


def _pair_feasible(options: dict | None) -> dict[str, bool]:
    """Whether the first hot and first cold stream may match, by pinch side."""
    problem = _four_stream_problem(options=options)
    arrays = problem_to_solver_arrays(problem, 14.0)
    decompositions = _pdm_decompositions(problem, dTmin=14.0)
    feasible = {}
    for side, goal in (("above", "hot utility"), ("below", "cold utility")):
        model = PinchDecompModel(
            name=f"forbidden-{side}",
            framework="PDM",
            solver="highs",
            solver_arrays=arrays,
            dTmin=14.0,
            z_restriction=None,
            min_dqda=0,
            minimisation_goal=goal,
            non_isothermal_model=False,
            integers=True,
            tol=1e-3,
            pinch_loc=side,
            pinch_decomposition=decompositions[side],
            stage_selection="automated",
        )
        feasible[side] = any(model.z_feasible[0][0])
    return feasible


def test_a_forbidden_recovery_match_is_feasible_on_neither_pinch_side():
    arrays = problem_to_solver_arrays(_four_stream_problem(), 14.0)
    hot = arrays.stream_identities["hot_process_streams"][0]
    cold = arrays.stream_identities["cold_process_streams"][0]

    assert any(_pair_feasible(None).values())
    assert _pair_feasible({"HENS_FORBIDDEN_MATCHES": [[hot, cold]]}) == {
        "above": False,
        "below": False,
    }
