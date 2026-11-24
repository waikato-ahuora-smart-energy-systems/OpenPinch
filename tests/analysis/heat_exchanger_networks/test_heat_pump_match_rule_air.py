"""An ambient source serves its evaporator and nothing else.

The air an evaporator draws on is in the case as a hot stream so that the
evaporator's balance closes. It is not heat the process may use, and it is
not heat to be cooled away: it may meet evaporators only.
"""

from __future__ import annotations

import numpy as np

from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    problem_to_solver_arrays,
)
from tests.application.heat_pump_stream_cases import full_load_target
from tests.application.test_heat_pump_stream_case_air_source import _short_of_heat

D_TMIN = 10.0


def test_the_air_may_meet_the_evaporator_but_no_process_stream_and_no_cooler():
    problem = _short_of_heat()
    case = problem.with_heat_pump_streams(full_load_target(problem))
    prepared = problem_to_solver_arrays(case, D_TMIN)
    hot = prepared.axis_maps["hot_process_streams"]
    cold = prepared.axis_maps["cold_process_streams"]
    air = next(i for key, i in hot.items() if key.endswith(".AIR_H1"))
    evaporator = next(j for key, j in cold.items() if key.endswith(".HP_C1"))
    process = next(j for key, j in cold.items() if key.endswith(".C1"))
    cold_utilities = len(prepared.axis_maps["cold_utilities"])

    recovery = prepared.arrays["forbidden_recovery"]
    coolers = prepared.arrays.get(
        "forbidden_cu", np.zeros((cold_utilities, len(hot)), dtype=int)
    )

    assert recovery[air][evaporator] == 0
    assert recovery[air][process] == 1
    assert [row[air] for row in coolers] == [1] * cold_utilities
