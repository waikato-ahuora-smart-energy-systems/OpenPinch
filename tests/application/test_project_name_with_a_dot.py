"""A project name with a dot survives the problem's own JSON.

A stream's zone label has its dots written as dashes when it is read (a
dot separates a zone from a stream in a stream's key). The project name,
which names the root zone, kept its dot, so the problem's own JSON named
zones under ``Plant v1.2`` that the reader looked for under
``Plant v1-2``: every path that builds a problem from another's JSON (a
heat pump design, ``with_heat_pump_streams``, a retrofit with a heat pump)
raised "could not resolve zone".
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from tests.application.heat_pump_stream_cases import basic_source


def test_a_problem_named_with_a_dot_is_read_again_from_its_own_json():
    with pytest.warns(UserWarning, match="'Plant v1.2' was changed to 'Plant v1-2'"):
        problem = PinchProblem(source=basic_source(), project_name="Plant v1.2")

    again = PinchProblem(problem.to_problem_json(), project_name=problem.project_name)

    assert problem._require_prepared_root_zone().address == "Plant v1-2"
    assert again._require_prepared_root_zone().address == "Plant v1-2"
