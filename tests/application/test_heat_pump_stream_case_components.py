"""A problem with process components is refused.

Process components (a process MVR, say) live on the running problem, not
in its input records. A case rebuilt from the records would lose them
without a word, and with them the streams they replace.
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from tests.application.heat_pump_stream_cases import basic_source, full_load_target


def test_a_problem_with_process_components_is_refused():
    problem = PinchProblem(source=basic_source(), project_name="basic")
    target = full_load_target(problem)
    problem._process_components = {"mvr_1": object()}

    with pytest.raises(ValueError, match="process components"):
        problem.with_heat_pump_streams(target)
