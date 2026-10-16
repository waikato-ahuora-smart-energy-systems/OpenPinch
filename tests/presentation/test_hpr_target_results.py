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


def test_shared_design_hpr_row_reports_its_design_and_periods():
    problem = PinchProblem("crude_preheat_train_multiperiod.json")
    heat_pump = problem.target.carnot_heat_pump(
        period_id="base",
        load_fraction=0.25,
        condensers=1,
        evaporators=1,
        maximum_restarts=1,
        maximum_iterations=20,
        maximum_evaluations=50,
        options={"HPR_MULTIPERIOD_OPTIMIZATION_ENABLED": True},
    )
    details = heat_pump.hpr_details

    (row,) = [t for t in problem.results.targets if t.hpr_cycle is not None]

    assert row.hpr_design_vector == pytest.approx(
        [float(value) for value in details.design_vector], rel=1e-12
    )
    assert row.hpr_period_ids == ("turndown", "base", "peak")
    assert row.hpr_period_weights == pytest.approx((0.2, 0.6, 0.2), rel=1e-12)
    assert row.hpr_period_success == {"turndown": True, "base": True, "peak": True}
    assert row.hpr_weighted_objective == pytest.approx(
        float(details.weighted_output["obj"]), rel=1e-12
    )


def test_single_period_hpr_row_has_no_shared_design_values():
    problem = PinchProblem("basic_pinch.json")
    _carnot_heat_pump(problem)

    (row,) = [t for t in problem.results.targets if t.hpr_cycle is not None]

    assert row.hpr_design_vector is None
    assert row.hpr_period_ids is None
    assert row.hpr_period_weights is None
    assert row.hpr_period_success is None
    assert row.hpr_weighted_objective is None
