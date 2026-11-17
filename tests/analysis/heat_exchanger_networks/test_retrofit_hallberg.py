"""Retrofit rating checked against Hallberg and Scribner (2009), Example 2.

The case is their crude unit ("Heat Exchanger Network Retrofit Comparison",
University of Oklahoma design report, 2009; adapted from Barbaro and
Bagajewicz, 2005): 13 process streams, eight with several segments (Table
3), the existing network from Fig. 2 with the areas of Table 4, and the
utilities of Table 5. The report's figure names HU11, HU12 and CU4 to CU6
are the table's HU1, HU2 and CU1 to CU3. The report gives no retrofitted
utility for this example, so only the existing network is checked.
"""

from __future__ import annotations

import json

import pytest

from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
    require_solver_binary,
    require_synthesis_dependency,
)
from OpenPinch.application.problem import PinchProblem
from tests.support.paths import FIXTURES_ROOT

CASE_JSON = (
    FIXTURES_ROOT / "retrofit" / "Thirteen-stream-Hallberg-and-Scribner-2009-crude.json"
)

# Text, p. 11 of the report: base hot and cold utility (kW).
PUBLISHED_HOT_UTILITY = 67_964.0
PUBLISHED_COLD_UTILITY = 75_051.0

# Fig. 2, in stages (hot streams run stage 1 to 5, cold streams 5 to 1):
# C1 meets E12, E1, E5, E6, then heater E9; C2 meets E10, E11, then E18.
# No splits. Areas in m2 from Table 4.
EXISTING = {
    "recovery": [
        {"hot": "H3", "cold": "C2", "stage": 1, "area": 685.70, "id": "E11"},
        {"hot": "H8", "cold": "C1", "stage": 1, "area": 19.55, "id": "E6"},
        {"hot": "H5", "cold": "C2", "stage": 2, "area": 93.70, "id": "E10"},
        {"hot": "H5", "cold": "C1", "stage": 3, "area": 26.30, "id": "E5"},
        {"hot": "H3", "cold": "C1", "stage": 4, "area": 4303.20, "id": "E1"},
        {"hot": "H5", "cold": "C1", "stage": 5, "area": 40.00, "id": "E12"},
    ],
    "heaters": [
        {"stream": "C1", "utility": "HU1", "area": 1211.22, "id": "E9"},
        {"stream": "C3", "utility": "HU1", "area": 52.24, "id": "E17"},
        {"stream": "C2", "utility": "HU2", "area": 979.62, "id": "E18"},
    ],
    "coolers": [
        {"stream": "H7", "utility": "CU1", "area": 63.80, "id": "E2"},
        {"stream": "H8", "utility": "CU1", "area": 33.17, "id": "E3"},
        {"stream": "H5", "utility": "CU1", "area": 4.07, "id": "E4"},
        {"stream": "H2", "utility": "CU1", "area": 5.87, "id": "E7"},
        {"stream": "H6", "utility": "CU1", "area": 146.59, "id": "E8"},
        {"stream": "H9", "utility": "CU1", "area": 182.39, "id": "E13"},
        {"stream": "H4", "utility": "CU2", "area": 101.47, "id": "E14"},
        {"stream": "H1", "utility": "CU3", "area": 93.87, "id": "E15"},
        {"stream": "H10", "utility": "CU1", "area": 288.97, "id": "E16"},
    ],
}


def _live_solver_available() -> bool:
    try:
        require_synthesis_dependency("pyomo.environ", purpose="retrofit test")
        require_solver_binary("ipopt", purpose="retrofit test")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


_needs_live_solver = pytest.mark.skipif(
    "not _live_solver_available()",
    reason="needs the ipopt binary and the pyomo package",
)


@pytest.mark.synthesis
@pytest.mark.solver
@_needs_live_solver
def test_the_existing_segmented_network_rates_close_to_its_published_utilities():
    problem = PinchProblem(json.loads(CASE_JSON.read_text(encoding="utf-8")))

    # The published areas are slightly short of the published duties (E6
    # most), so the rating needs 5 % on every installed area, not the 2 %
    # default.
    result = problem.design.retrofit(EXISTING, max_modifications=0, area_tolerance=0.05)

    base = result.frontier[0]
    assert base.hot_utility == pytest.approx(PUBLISHED_HOT_UTILITY, rel=0.015)
    assert base.cold_utility == pytest.approx(PUBLISHED_COLD_UTILITY, rel=0.005)
