"""A utility heat pump on a site is sized on the site's utility duties.

The site imports 800 kW of MPS. Its two streams read as one process would
need 400 kW. A utility heat pump works between the site's utility levels,
so at full load it delivers the 800 kW the site imports.
"""

from __future__ import annotations

import pytest

from tests.analysis.heat_pumps.two_level_site import (
    SITE_HOT_UTILITY,
    two_level_site,
    utility_heat_pump_target,
)


def test_a_full_load_utility_heat_pump_delivers_the_sites_hot_utility():
    problem = two_level_site()
    site = problem.target.total_site_heat_integration()
    assert site.hot_utility_target == pytest.approx(SITE_HOT_UTILITY)

    target = utility_heat_pump_target(problem)

    assert target.hpr_load.available == pytest.approx(SITE_HOT_UTILITY)
    assert target.hpr_load.selected == pytest.approx(SITE_HOT_UTILITY)
