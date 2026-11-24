"""``zone=`` does not apply to a heat pump on the utility system.

For a heat pump on the process ``zone=`` names the process zone that
takes its streams. A utility heat pump gets zones of its own under the
zone it was targeted on, so a ``zone=`` would be ignored; it is refused
so the caller is not left thinking it placed the streams.
"""

from __future__ import annotations

import pytest

from tests.analysis.heat_pumps.two_level_site import (
    two_level_site,
    utility_heat_pump_target,
)


def test_a_zone_given_with_a_utility_heat_pump_target_is_refused():
    problem = two_level_site()
    target = utility_heat_pump_target(problem)

    with pytest.raises(ValueError, match="zone= .*utility system"):
        problem.with_heat_pump_streams(target, zone="A")
