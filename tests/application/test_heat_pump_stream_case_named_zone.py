"""A named process zone takes the heat pump's streams.

The zone is given by its name or by its full address; either way the new
streams carry the address, as every stream record does.
"""

from __future__ import annotations

import pytest

from tests.application.heat_pump_stream_cases import (
    full_load_target,
    heat_pump_zones,
    two_zone_problem,
)


@pytest.mark.parametrize("zone", ["B", "Site/B"])
def test_the_named_zone_takes_the_heat_pump_streams(zone):
    problem = two_zone_problem()
    target = full_load_target(problem)

    case = problem.with_heat_pump_streams(target, zone=zone)

    assert heat_pump_zones(case) == {"Site/B"}
