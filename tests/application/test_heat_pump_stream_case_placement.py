"""Only a heat pump target becomes streams of a case.

A heat pump on the process is added to a process zone and one on the
utility system as zones of its own (``test_utility_heat_pump_zones.py``).
A refrigeration target is neither: its cold side serves the process
below ambient and its hot side rejects to air or cooling water, which
the stream builder does not write.

Until utility heat pumps were built as zones, this file held their
refusal.
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


def test_a_refrigeration_target_is_refused():
    problem = PinchProblem(
        source=json.loads(BASIC_PINCH.read_text(encoding="utf-8")),
        project_name="basic",
    )
    # A small search, as ``full_load_target`` takes: the refusal does not
    # depend on how good the target is.
    target = problem.target.carnot_refrigeration(
        load_fraction=1.0,
        maximum_restarts=0,
        maximum_evaluations=1,
    )

    with pytest.raises(ValueError, match="takes a heat pump target.*Refrigeration"):
        problem.with_heat_pump_streams(target)
