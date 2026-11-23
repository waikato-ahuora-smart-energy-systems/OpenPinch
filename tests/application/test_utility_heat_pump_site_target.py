"""A site with its utility heat pump's zones targets as the heat pump did.

The site imports 800 kW of MPS and dumps 1000 kW of LPS. With the heat
pump's two zones in the case, total site targeting of the case alone
gives the hot utility the heat pump target left. Its cold utility is the
target's plus the heat the target let ambient air take: the air is not
written into the case, so the site's cold utility takes that heat.
"""

from __future__ import annotations

import pytest

from tests.analysis.heat_pumps.two_level_site import (
    SITE_HOT_UTILITY,
    two_level_site,
    utility_heat_pump_target,
)


def test_the_site_target_of_the_case_is_the_heat_pump_targets_net_utility():
    problem = two_level_site()
    target = utility_heat_pump_target(problem)
    assert target.hot_utility_target < 1.0 < SITE_HOT_UTILITY

    case = problem.with_heat_pump_streams(target)
    site = case.target.total_site_heat_integration()

    assert site.hot_utility_target == pytest.approx(target.hot_utility_target, abs=1e-6)
    assert site.cold_utility_target == pytest.approx(
        target.cold_utility_target + float(target.hpr_ambient_cold), abs=1e-6
    )
