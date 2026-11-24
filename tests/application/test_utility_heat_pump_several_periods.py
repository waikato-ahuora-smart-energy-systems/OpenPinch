"""A utility heat pump over several periods needs a shared design.

On a site with several periods the heat pump's zones are written with its
streams in every period, which only a shared-design target holds
(``HPR_MULTIPERIOD_OPTIMIZATION_ENABLED``). A target solved for one period
is refused, as a process heat pump's is; until 2026-11-25 every utility
heat pump on such a site was refused, the zones not being built for it.
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from tests.analysis.heat_pumps.two_level_site import (
    two_level_site_source,
    utility_heat_pump_target,
)


def test_a_one_period_target_on_a_site_with_several_periods_is_refused():
    source = two_level_site_source(
        {"PROBLEM_PERIOD_IDS": ["base", "peak"], "PROBLEM_PERIOD_WEIGHTS": [0.5, 0.5]}
    )
    problem = PinchProblem(source=source, project_name="periods")
    target = utility_heat_pump_target(problem)

    with pytest.raises(ValueError, match="2 periods.*shared design"):
        problem.with_heat_pump_streams(target)
