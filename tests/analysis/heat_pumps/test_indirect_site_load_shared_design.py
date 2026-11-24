"""A shared-design utility heat pump is sized on the site's duties too.

Solved as one machine over every period
(``HPR_MULTIPERIOD_OPTIMIZATION_ENABLED``), the utility heat pump of the
two-level site still delivers the 800 kW the site imports at full load,
not the 400 kW its streams would need read as one process.
"""

from __future__ import annotations

import pytest

from tests.analysis.heat_pumps.two_level_site import (
    SITE_HOT_UTILITY,
    two_level_site,
    utility_heat_pump_target,
)


def test_a_shared_design_utility_heat_pump_delivers_the_sites_hot_utility():
    problem = two_level_site()

    target = utility_heat_pump_target(
        problem, options={"HPR_MULTIPERIOD_OPTIMIZATION_ENABLED": True}
    )

    assert target.hpr_shared_design
    condensers = [stream for _key, stream in target.hpr_hot_streams.items()]
    assert sum(float(s.heat_flow.value) for s in condensers) == pytest.approx(
        SITE_HOT_UTILITY
    )
