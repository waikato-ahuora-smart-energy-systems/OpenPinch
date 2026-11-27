"""A residual utility problem named with a dot has the root other problems do.

A zone's name has its dots written as dashes, the root named for the
project among them. A residual utility problem is prepared on a path of
its own, which named its zone as given: ``Plant v1.2`` where every other
problem of that name has ``Plant v1-2``.
"""

from __future__ import annotations

import warnings

from OpenPinch import PinchProblem
from tests.application.heat_pump_stream_cases import (
    basic_source,
    full_load_target,
)


def test_a_residual_problem_named_with_a_dot_has_its_dots_as_dashes():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = PinchProblem(source=basic_source(), project_name="basic")
        target = full_load_target(problem)
        residual = problem.residual_utility(
            base_target=target, project_name="Plant v1.2"
        )

        address = residual._require_prepared_root_zone().address

    assert address == "Plant v1-2"
