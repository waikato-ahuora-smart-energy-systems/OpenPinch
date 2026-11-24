"""A heat pump's streams keep its approach under a zone dt_cont multiplier.

The target stands the heat pump ``HPR_DT_CONT`` off the process as given
(its effective ``HPR_DT_CONT`` is unscaled); only the process streams'
contributions carry the zone's multiplier. The case's streams inherit
that multiplier, so they are written with ``HPR_DT_CONT`` over it, as the
retrofit's heat pump case writes them: at a multiplier of 2 and 1 K,
0.5 K, which the zone scales back to 1 K.
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


def test_the_heat_pump_streams_carry_hpr_dt_cont_over_the_multiplier():
    problem = PinchProblem(
        source=json.loads(BASIC_PINCH.read_text(encoding="utf-8")),
        project_name="basic",
    )
    problem.set_dt_cont_multiplier(2.0)
    target = full_load_target(problem)
    approach = float(target.provenance.effective_settings["HPR_DT_CONT"])

    case = problem.with_heat_pump_streams(target)

    written = {
        record["name"]: record["dt_cont"]["value"]
        for record in case.to_problem_json()["streams"]
        if (record["heat_pump"] or {}).get("role") in ("condenser", "evaporator")
    }
    assert written == {
        "HP_H1": pytest.approx(approach / 2.0),
        "HP_C1": pytest.approx(approach / 2.0),
    }
