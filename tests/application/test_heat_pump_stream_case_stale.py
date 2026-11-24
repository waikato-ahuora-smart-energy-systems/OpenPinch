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
from tests.application.heat_pump_stream_cases import full_load_target

BASIC_PINCH = (
    Path(OpenPinch.__file__).parent / "tutorials/sample_cases/basic_pinch.json"
)


def _problem() -> PinchProblem:
    return PinchProblem(
        source=json.loads(BASIC_PINCH.read_text(encoding="utf-8")),
        project_name="basic",
    )


def test_a_target_solved_on_another_problem_is_refused():
    target = full_load_target(_problem())

    with pytest.raises(ValueError, match="foreign, stale or modified"):
        _problem().with_heat_pump_streams(target)
