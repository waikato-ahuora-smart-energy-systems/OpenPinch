"""Only a heat pump placed on the process becomes process streams.

A utility heat pump lifts heat between utility levels: its condensers and
evaporators meet the utility system, not the process streams, so adding
them to a process zone would let a design match them to streams the
target never offered.
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


def test_a_utility_heat_pump_target_is_refused():
    problem = PinchProblem(
        source=json.loads(BASIC_PINCH.read_text(encoding="utf-8")),
        project_name="basic",
    )
    target = problem.target.carnot_heat_pump(
        is_utility_heat_pump=True, load_fraction=1.0, maximum_restarts=1
    )

    with pytest.raises(ValueError, match="placed on the process"):
        problem.with_heat_pump_streams(target)
