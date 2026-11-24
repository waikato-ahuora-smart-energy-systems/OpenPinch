"""No cooler may go on a condenser stream.

A condenser's heat is the heat pump's product, paid for with electricity.
A cooler on it would throw that heat away and call the stream served, so
no cold utility may take it: a condenser is cooled by the process alone.
"""

from __future__ import annotations

import numpy as np

from OpenPinch import PinchProblem
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    problem_to_solver_arrays,
)
from tests.analysis.heat_exchanger_networks.test_heat_pump_case_pinch_split import (
    _half_load_case,
)

D_TMIN = 7.5


def test_every_cold_utility_is_forbidden_on_the_condenser_and_on_it_alone():
    prepared = problem_to_solver_arrays(
        PinchProblem(source=_half_load_case(), project_name="half"), D_TMIN
    )
    hot = prepared.axis_maps["hot_process_streams"]
    condenser = next(i for key, i in hot.items() if key.endswith(".HP_H1"))
    cold_utilities = len(prepared.axis_maps["cold_utilities"])

    coolers = prepared.arrays.get(
        "forbidden_cu", np.zeros((cold_utilities, len(hot)), dtype=int)
    )

    assert cold_utilities >= 1
    assert [row[condenser] for row in coolers] == [1] * cold_utilities
    assert coolers.sum() == cold_utilities
