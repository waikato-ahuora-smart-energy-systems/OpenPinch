"""A utility heat pump targeted on one process is refused.

A heat pump on the utility system joins a site as process zones beside
the site's processes. Targeted on one process zone it would have to be
added inside that process, which holds unit operations, not processes.
"""

from __future__ import annotations

import pytest

from tests.analysis.heat_pumps.two_level_site import (
    two_level_site,
    utility_heat_pump_target,
)


def test_a_utility_heat_pump_targeted_on_a_process_zone_is_refused():
    problem = two_level_site()
    target = utility_heat_pump_target(problem, zone="B")

    with pytest.raises(ValueError, match="process zone 'two level site/B'.*site"):
        problem.with_heat_pump_streams(target)
