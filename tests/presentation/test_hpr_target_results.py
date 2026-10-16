"""HPR summary values carried by the reported target rows."""

import pytest

from OpenPinch import PinchProblem


def _carnot_heat_pump(problem):
    return problem.target.carnot_heat_pump(
        is_utility_heat_pump=False,
        is_cascade_cycle=True,
        load_fraction=0.25,
        condensers=1,
        evaporators=1,
        maximum_restarts=1,
        maximum_iterations=20,
        maximum_evaluations=50,
    )


def test_hpr_row_reports_selected_and_achieved_load_and_objective():
    problem = PinchProblem("basic_pinch.json")
    heat_pump = _carnot_heat_pump(problem)

    (row,) = [t for t in problem.results.targets if t.hpr_cycle is not None]

    assert row.hpr_selected_load.value == pytest.approx(
        heat_pump.hpr_load.selected, rel=1e-12
    )
    assert row.hpr_selected_load.unit == "kW"
    assert row.hpr_achieved_load.value == pytest.approx(
        heat_pump.hpr_load.achieved, rel=1e-12
    )
    assert row.hpr_objective == pytest.approx(
        float(heat_pump.hpr_details.obj), rel=1e-12
    )
