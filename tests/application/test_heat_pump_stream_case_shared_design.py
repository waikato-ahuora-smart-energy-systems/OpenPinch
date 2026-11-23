"""A shared-design heat pump target gives streams that vary by period.

Solved with ``HPR_MULTIPERIOD_OPTIMIZATION_ENABLED``, a target on a
problem with several periods holds one machine's condenser and evaporator
in every period: hotter and larger at peak, smaller at turndown. The new
case writes each stream once, with a value per period, as the problem's
own streams are written.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import OpenPinch
from OpenPinch import PinchProblem
from tests.application.heat_pump_stream_cases import full_load_target

MULTIPERIOD = (
    Path(OpenPinch.__file__).parent
    / "tutorials/sample_cases/crude_preheat_train_multiperiod.json"
)


def test_each_heat_pump_stream_carries_its_value_in_every_period():
    problem = PinchProblem(
        source=json.loads(MULTIPERIOD.read_text(encoding="utf-8")),
        project_name="periods",
    )
    target = full_load_target(
        problem, options={"HPR_MULTIPERIOD_OPTIMIZATION_ENABLED": True}
    )
    outputs = target.hpr_details.period_outputs

    case = problem.with_heat_pump_streams(target)

    records = {
        record["name"]: record
        for record in case.to_problem_json()["streams"]
        if record["heat_pump"] is not None
    }
    assert list(problem.period_ids) == ["turndown", "base", "peak"]
    for name, group in (("HP_H1", "hpr_hot_streams"), ("HP_C1", "hpr_cold_streams")):
        solved = [dict(outputs[period][group].items())[name] for period in outputs]
        assert records[name]["heat_flow"]["values"] == pytest.approx(
            [float(stream.heat_flow.value) for stream in solved]
        )
        assert records[name]["t_supply"]["values"] == pytest.approx(
            [float(stream.supply_temperature.value) for stream in solved]
        )
    # The condenser is larger at peak than at turndown.
    duties = records["HP_H1"]["heat_flow"]["values"]
    assert duties[0] < duties[1] < duties[2]
