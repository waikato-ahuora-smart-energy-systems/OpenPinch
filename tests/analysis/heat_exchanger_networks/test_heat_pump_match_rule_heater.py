"""No heater may go on an evaporator stream.

An evaporator is there to take waste heat. Steam on it would feed the
compressor with bought heat and leave the waste heat for the cooling
water, so no hot utility may serve it: an evaporator is heated by the
process (or by its ambient source) alone.
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


def test_every_hot_utility_is_forbidden_on_the_evaporator_and_on_it_alone():
    prepared = problem_to_solver_arrays(
        PinchProblem(source=_half_load_case(), project_name="half"), D_TMIN
    )
    cold = prepared.axis_maps["cold_process_streams"]
    evaporator = next(j for key, j in cold.items() if key.endswith(".HP_C1"))
    hot_utilities = len(prepared.axis_maps["hot_utilities"])

    heaters = prepared.arrays.get(
        "forbidden_hu", np.zeros((hot_utilities, len(cold)), dtype=int)
    )

    assert hot_utilities >= 1
    assert [row[evaporator] for row in heaters] == [1] * hot_utilities
    assert heaters.sum() == hot_utilities
