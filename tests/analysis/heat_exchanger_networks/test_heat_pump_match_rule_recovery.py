"""A condenser stream may not be matched to an evaporator stream.

A heat pump's condenser is hotter than its evaporator, so the pair is a
feasible match and a cheap one: it would hand the machine's output straight
back to its input and leave the process unheated. The marks on the two
streams forbid it, through the same masks as ``HENS_FORBIDDEN_MATCHES``,
so every design step honours it.
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

# The solver's scalar approach at the case's 5 K contribution, multiplier 0.75.
D_TMIN = 7.5


def _by_name(axis: dict[str, int]) -> dict[str, int]:
    return {key.rsplit(".", 1)[-1]: index for key, index in axis.items()}


def test_the_condenser_to_evaporator_match_is_forbidden_and_no_other():
    prepared = problem_to_solver_arrays(
        PinchProblem(source=_half_load_case(), project_name="half"), D_TMIN
    )
    hot = _by_name(prepared.axis_maps["hot_process_streams"])
    cold = _by_name(prepared.axis_maps["cold_process_streams"])

    # A problem with nothing forbidden carries no mask.
    recovery = prepared.arrays.get(
        "forbidden_recovery", np.zeros((len(hot), len(cold)), dtype=int)
    )

    assert recovery[hot["HP_H1"]][cold["HP_C1"]] == 1
    assert recovery.sum() == 1
