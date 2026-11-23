"""A utility heat pump's zones are not built over several periods yet.

On a site with several periods a utility heat pump target holds one
period's streams, or as a shared design one set per period. The zones
would have to carry a value per period and move each period's streams
clear of that period's level bands, which is not built. Written from one
period the case would be wrong in the others, so it is refused by name.
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from tests.analysis.heat_pumps.two_level_site import (
    two_level_site_source,
    utility_heat_pump_target,
)


def test_a_site_with_several_periods_is_refused():
    source = two_level_site_source(
        {"PROBLEM_PERIOD_IDS": ["base", "peak"], "PROBLEM_PERIOD_WEIGHTS": [0.5, 0.5]}
    )
    problem = PinchProblem(source=source, project_name="periods")
    target = utility_heat_pump_target(problem)

    with pytest.raises(ValueError, match="utility.*2 periods"):
        problem.with_heat_pump_streams(target)
