"""A segmented stream with a near-flat boiling interval (issue 332).

The boiling segment carries 41.84 kW over 0.01 K, so a stage temperature that
lands inside it maps to heat at about 4184 kW/K. PDM result assembly once
rounded below-pinch stage temperatures to five decimals, which moved the
heat coordinate by several watts and failed the segment area check.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.solver.backend import (
    require_solver_backend,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from OpenPinch.application.problem import PinchProblem

# (supply K, target K, duty kW) for each segment; the third is boiling.
BOILING_WATER = [
    (333.1499999999941, 338.92822304620734, 0.43579418437319967),
    (338.92822304620734, 342.2451741773144, 0.43579418437319967),
    (342.2451741773144, 342.2551741773144, 41.83624169982715),
    (342.2551741773144, 353.15000000000015, 0.4357941843732042),
]


def _highs_available() -> bool:
    try:
        require_solver_backend("highs")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


def _boiling_problem() -> PinchProblem:
    return PinchProblem(
        {
            "streams": [
                {
                    "zone": "Process",
                    "name": "Boiling water",
                    "segments": [
                        {
                            "t_supply": start,
                            "t_target": end,
                            "heat_flow": duty,
                            "htc": 2.0,
                            "dt_cont": 5.0,
                        }
                        for start, end, duty in BOILING_WATER
                    ],
                },
                {
                    "zone": "Process",
                    "name": "Hot stream",
                    "t_supply": 450.0,
                    "t_target": 400.0,
                    "heat_flow": 60.0,
                    "htc": 1.0,
                    "dt_cont": 5.0,
                },
            ],
            "utilities": [
                {
                    "name": "Hot utility",
                    "type": "Hot",
                    "t_supply": 373.15000000000015,
                    "t_target": 363.15000000000015,
                    "htc": 1.0,
                    "dt_cont": 5.0,
                    "price": 0.0,
                },
                {
                    "name": "Cold utility",
                    "type": "Cold",
                    "t_supply": 380.0,
                    "t_target": 390.0,
                    "htc": 1.0,
                    "dt_cont": 5.0,
                    "price": 0.0,
                },
            ],
            "options": {
                "INPUT_UNIT_TEMPERATURE": "K",
                "INPUT_UNIT_HEAT_FLOW": "kW",
            },
        },
        project_name="issue_332",
    )


@pytest.mark.synthesis
# A string condition is evaluated when the test runs, so collecting this module
# does not import the solver packages.
@pytest.mark.skipif("not _highs_available()", reason="needs highspy and pyomo")
def test_milp_pinch_design_handles_a_near_flat_boiling_segment():
    view = _boiling_problem().design.pinch_design(
        dt_cont_multipliers=[1.0],
        stages=[1, 2],
        best_solutions=1,
        max_parallel=1,
        solve_tolerance=0.01,
        # The MILP method builds its network by amalgamating pinch sides.
        options={"HENS_PDM_MILP": True, "HENS_HEURISTIC_PDM_NETWORKS": 0},
    )

    boiling_duty = sum(duty for _, _, duty in BOILING_WATER)
    assert view.total_heat_recovery == pytest.approx(boiling_duty, abs=1e-3)
    assert view.total_hot_utility == pytest.approx(0.0, abs=1e-3)
    assert view.total_cold_utility == pytest.approx(60.0 - boiling_duty, abs=1e-3)
