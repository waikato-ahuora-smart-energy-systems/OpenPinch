"""A zone the heat pump target does not cover is refused by name."""

from __future__ import annotations

import pytest

from tests.application.heat_pump_stream_cases import (
    full_load_target,
    two_zone_problem,
)


def test_a_zone_the_target_does_not_cover_is_refused():
    problem = two_zone_problem()
    target = full_load_target(problem)

    with pytest.raises(ValueError, match="no process zone 'C'.*'Site/A', 'Site/B'"):
        problem.with_heat_pump_streams(target, zone="C")
