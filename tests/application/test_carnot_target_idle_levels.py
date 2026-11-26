"""A Carnot target says when it leaves a requested level idle.

Asked for three condenser levels, a search can stop at a design that uses
two and leaves the third without duty, a level short of the best design for
three. The target reports success either way, so it warns.
"""

from __future__ import annotations

import json

import pytest

from OpenPinch import PinchProblem
from OpenPinch.resources import read_sample_case


def _basic_pinch() -> PinchProblem:
    return PinchProblem(
        json.loads(read_sample_case("basic_pinch.json")), project_name="basic"
    )


def test_a_carnot_target_warns_of_requested_levels_left_idle():
    # One evaluation: every condenser at the hottest temperature, so the
    # last takes all 750 kW and two carry nothing; one evaporator is idle.
    with pytest.warns(UserWarning, match="2 of its 3 condenser levels") as caught:
        target = _basic_pinch().target.carnot_heat_pump(
            load_fraction=1.0,
            condensers=3,
            evaporators=2,
            maximum_restarts=1,
            maximum_iterations=1,
            maximum_evaluations=1,
        )

    assert sum(duty > 1e-6 for duty in target.hpr_details.Q_cond) == 1
    (warning,) = [w for w in caught if "condenser levels" in str(w.message)]
    assert "1 of its 2 evaporator levels" in str(warning.message)
