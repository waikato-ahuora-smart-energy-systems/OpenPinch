"""Network evolution's first solve is a model Ipopt accepts.

Without integers a segmented stream's temperature follows one segment's
line at a time, the solve loop moving it to the segment its heat lies in.
The model wrote one equality per segment and switched all but the active
one off with a zero factor. Ipopt counts equalities, not what is in them:
on the kink case with a second segmented stream the root model had 171
equalities, 31 of them switched off, against 170 variables, and Ipopt
refused it ("Problem has too few degrees of freedom"). Evolution then
returned no network.
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from tests.analysis.heat_exchanger_networks.kink_case import kink_problem


def _available(solver: str) -> bool:
    try:
        backend.require_solver_backend(solver)
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


@pytest.mark.solver
@pytest.mark.skipif(
    "not (_available('highs') and _available('ipopt-pyomo'))",
    reason="needs highspy and ipopt",
)
def test_evolution_solves_its_root_with_two_segmented_streams():
    # The kink case's condensing stream V beside a cold stream W whose
    # heat capacity flowrate rises from 1 to 3 kW/K at 100 degC.
    source = kink_problem().to_problem_json()
    source["streams"].append(
        {
            "zone": "Plant",
            "name": "W",
            "dt_cont": {"value": 5.0, "unit": "delta_degC"},
            "segments": [
                {"t_supply": 40.0, "t_target": 100.0, "heat_flow": 60.0},
                {"t_supply": 100.0, "t_target": 160.0, "heat_flow": 180.0},
            ],
        }
    )
    source["options"] = {
        **source.get("options", {}),
        "HENS_DT_CONT_MULTIPLIERS": [1.0],
        "HENS_MAX_PARALLEL": 1,
    }
    problem = PinchProblem(source=source, project_name="segmented root")
    seeds = [
        outcome.network
        for outcome in problem.design.pinch_design().result.ranked_networks
    ]

    view = problem.design.network_evolution(seeds)

    assert [outcome.network for outcome in view.result.ranked_networks]
