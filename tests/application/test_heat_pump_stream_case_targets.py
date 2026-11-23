"""The case with a heat pump's streams targets at the heat pump target.

The heat pump target leaves the process a residual hot utility and takes
its evaporator duty out of the cooling. With the two streams in the case
at ``HPR_DT_CONT``, ordinary direct heat integration targeting gives the
same two numbers, so a design held to its targets is held to the heat
pump's.
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


def test_direct_targets_of_the_new_case_are_the_heat_pump_targets_utilities():
    problem = PinchProblem(
        source=json.loads(BASIC_PINCH.read_text(encoding="utf-8")),
        project_name="basic",
    )
    base = problem.target.direct_heat_integration()
    # A small search leaves 35 kW of hot utility where the full one leaves
    # none; the case must reproduce the target it is given either way.
    target = full_load_target(problem)
    evaporator = sum(
        float(stream.heat_flow.value)
        for _key, stream in target.hpr_cold_streams.items()
    )

    direct = problem.with_heat_pump_streams(target).target.direct_heat_integration()

    # 750 kW of steam and 1000 kW of cooling without the heat pump.
    assert float(base.hot_utility_target) == pytest.approx(750.0, abs=1e-3)
    assert float(direct.hot_utility_target) == pytest.approx(
        float(target.hot_utility_target), abs=0.01
    )
    assert float(direct.cold_utility_target) == pytest.approx(
        float(base.cold_utility_target) - evaporator, abs=0.01
    )
