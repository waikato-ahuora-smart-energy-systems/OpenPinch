"""A heat pump design case on several periods prices the machine over them.

The record of the crude preheat train's heat pump weighs its work by the
problem's period weights (0.2 turndown, 0.6 base, 0.2 peak) and takes its
capacity from the peak, not from the one period the target's own fields
happen to hold.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import OpenPinch
from OpenPinch import PinchProblem
from OpenPinch.application.heat_pump_design import heat_pump_design_case

MULTIPERIOD = (
    Path(OpenPinch.__file__).parent
    / "tutorials/sample_cases/crude_preheat_train_multiperiod.json"
)


def test_the_record_weighs_work_by_period_and_takes_the_peak_capacity():
    problem = PinchProblem(
        source=json.loads(MULTIPERIOD.read_text(encoding="utf-8")),
        project_name="periods",
    )

    run = heat_pump_design_case(
        problem,
        1.0,
        levels={"condensers": 1, "evaporators": 1},
        search={"restarts": 0, "evaluations": 1},
        cost_per_kw=200.0,
    )

    outputs = run.target.hpr_details.period_outputs
    work = {period: float(output["w_net"]) for period, output in outputs.items()}
    assert run.record.work == pytest.approx(
        0.2 * work["turndown"] + 0.6 * work["base"] + 0.2 * work["peak"]
    )
    peak = next(s for _key, s in outputs["peak"]["hpr_hot_streams"].items())
    assert run.record.condenser_duty == pytest.approx(float(peak.heat_flow.value))
