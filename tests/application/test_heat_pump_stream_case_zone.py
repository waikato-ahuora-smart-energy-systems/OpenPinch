"""The heat pump's streams go in one process zone, named when it is not clear.

A target solved on a process zone puts its streams there. A target solved
on a site with one process zone puts them in that zone. On a site with
several, the caller says which.
"""

from __future__ import annotations

import pytest

from tests.application.heat_pump_stream_cases import (
    full_load_target,
    two_zone_problem,
)


def test_a_site_with_two_process_zones_needs_the_zone_named():
    problem = two_zone_problem()
    target = full_load_target(problem)

    with pytest.raises(ValueError, match="zone="):
        problem.with_heat_pump_streams(target)
