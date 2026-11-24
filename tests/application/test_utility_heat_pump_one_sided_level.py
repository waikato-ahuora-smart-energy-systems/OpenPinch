"""A case that would not target as the heat pump did is refused.

A utility heat pump's condenser delivers into the level the site buys
heat at. As a process zone of the site its condenser has to raise that
level, which the case's utilities allow only when the level takes heat
as well as gives it (type ``Both``). Here MPS is a hot utility alone:
the condenser would raise LPS instead and the site would still import
its 800 kW of MPS. The new case would look complete and target wrongly,
so it is refused.
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from tests.analysis.heat_pumps.two_level_site import (
    two_level_site_source,
    utility_heat_pump_target,
)


def test_a_level_the_condenser_cannot_raise_is_refused():
    source = two_level_site_source()
    mps = next(u for u in source["utilities"] if u["name"] == "MPS")
    mps["type"] = "Hot"
    problem = PinchProblem(source=source, project_name="one sided")
    target = utility_heat_pump_target(problem)

    with pytest.raises(ValueError, match="800.*type Both"):
        problem.with_heat_pump_streams(target)
