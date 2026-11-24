"""A problem a heat pump design cannot take is refused before any design.

The refusal does not depend on anything a design finds, so it comes
first: not after the design without a heat pump has run its solvers and
replaced the problem's stored design.

The problem refused here has process components. This test first used a
problem with several periods, which a heat pump design now takes (its
heat pump is solved as a shared design).
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from tests.application.heat_pump_stream_cases import basic_source


def test_process_components_are_refused_before_the_design_without_a_heat_pump():
    problem = PinchProblem(source=basic_source(), project_name="basic")
    problem._process_components = {"mvr_1": object()}

    with pytest.raises(ValueError, match="process components"):
        problem.design.heat_exchanger_network(
            heat_pump={"search": {"restarts": 0, "evaluations": 1}},
            heat_pump_cost=200.0,
            width=0,
            depth=0,
            max_parallel=1,
        )

    assert problem.results is None or problem.results.design is None
