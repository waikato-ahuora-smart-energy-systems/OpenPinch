"""Heat pump streams come only from this problem's own, current target.

A target solved on another problem, or on this one before its inputs
changed, describes a heat pump for different streams. Its condenser and
evaporator would not close this case's cascade, so the builder refuses it
as ``residual_utility`` does.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import OpenPinch
from OpenPinch import PinchProblem

BASIC_PINCH = (
    Path(OpenPinch.__file__).parent / "tutorials/sample_cases/basic_pinch.json"
)


def _problem() -> PinchProblem:
    return PinchProblem(
        source=json.loads(BASIC_PINCH.read_text(encoding="utf-8")),
        project_name="basic",
    )


def test_a_target_solved_on_another_problem_is_refused():
    target = _problem().target.carnot_heat_pump(
        load_fraction=1.0, condensers=1, evaporators=1, maximum_restarts=1
    )

    with pytest.raises(ValueError, match="foreign, stale or modified"):
        _problem().with_heat_pump_streams(target)
