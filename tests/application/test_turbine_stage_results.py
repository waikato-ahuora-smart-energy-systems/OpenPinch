"""Turbine stage records reach the cogeneration target and its result rows."""

import pytest

from OpenPinch import PinchProblem
from OpenPinch.analysis.thermodynamics.water import psat_T
from OpenPinch.contracts.reporting import TargetResults
from OpenPinch.contracts.turbine import TurbineSolveResult


def _level(name, kind, temperature):
    return {
        "name": name,
        "type": kind,
        "t_supply": {"value": temperature, "unit": "degC"},
        "t_target": {"value": temperature - 0.5, "unit": "degC"},
        "dt_cont": {"value": 1, "unit": "degC"},
        "price": {"value": 10, "unit": "$/MWh"},
        "htc": {"value": 1, "unit": "kW/m^2/degC"},
        "heat_flow": None,
    }


def _stream(name, t_supply, t_target, heat_flow):
    return {
        "zone": "Plant",
        "name": name,
        "t_supply": {"value": t_supply, "unit": "degC"},
        "t_target": {"value": t_target, "unit": "degC"},
        "heat_flow": heat_flow,
        "dt_cont": {"value": 5, "unit": "degC"},
        "htc": {"value": 1, "unit": "kW/m^2/degC"},
    }


def _problem(*, cold_duty, hot_levels=(("LP", 90.0),), options=None):
    return PinchProblem(
        {
            "streams": [
                _stream("Cold", 40, 70, cold_duty),
                _stream("Hot", 35, 30, {"value": 100, "unit": "kW"}),
            ],
            "utilities": [
                *(_level(name, "Hot", t) for name, t in hot_levels),
                _level("CU", "Cold", 10.1),
            ],
            "options": options or {},
        }
    )


def _turbine_rows(rows):
    return [row for row in rows if row.turbine_result is not None]


@pytest.fixture(scope="module")
def solved():
    problem = _problem(cold_duty={"value": 6000, "unit": "kW"})
    target = problem.target.cogeneration()
    return problem, target


def test_target_keeps_the_validated_solve_result(solved):
    _, target = solved

    result = target.turbine_result

    assert isinstance(result, TurbineSolveResult)
    assert result.mode == "above_pinch"
    assert result.converged is target.turbine_converged
    assert result.total_work == pytest.approx(target.work_target, rel=1e-12)
    assert len(result.stages) == 1


def test_stage_records_are_in_the_documented_units(solved):
    _, target = solved
    result = target.turbine_result
    stage = result.stages[0]

    # kW
    assert stage.process_duty == pytest.approx(6000.0, rel=1e-6)
    assert stage.work_actual == pytest.approx(target.work_target, rel=1e-9)
    # degC and bar
    assert stage.temperature == pytest.approx(89.5, abs=1e-9)
    assert stage.pressure_out == pytest.approx(psat_T(89.5), rel=1e-9)
    assert stage.pressure_in == pytest.approx(result.inlet_pressure, rel=1e-9)
    # kg/s times kJ/kg gives kW
    assert stage.mass_flow_extracted * (
        stage.enthalpy_out - stage.condensate_enthalpy
    ) == pytest.approx(stage.process_duty, rel=1e-5)


def test_result_row_carries_the_record_as_a_copy(solved):
    problem, target = solved

    row = _turbine_rows(problem.results.targets)[0]

    assert row.turbine_result == target.turbine_result
    assert row.turbine_result is not target.turbine_result
    assert row.turbine_result.stages[0] is not target.turbine_result.stages[0]


def test_record_survives_a_json_round_trip(solved):
    problem, _ = solved
    row = _turbine_rows(problem.results.targets)[0]

    restored = TargetResults.model_validate_json(row.model_dump_json())

    assert restored.turbine_result == row.turbine_result


def test_rows_without_a_turbine_run_have_no_record(solved):
    problem, _ = solved

    rows = problem.results.targets

    assert all(
        (row.turbine_result is None) == (row.turbine_converged is None) for row in rows
    )


def test_cogeneration_target_can_be_the_base_of_another(solved):
    problem, target = solved

    again = problem.target.sun_smith_cogeneration(base_target=target)

    assert again.turbine_result.turbine_model == "Sun & Smith (2015)"
    assert target.turbine_result.turbine_model != "Sun & Smith (2015)"


def test_each_period_row_has_its_own_record_and_the_average_has_none():
    problem = _problem(
        cold_duty={"values": [5000.0, 6000.0], "unit": "kW"},
        options={
            "PROBLEM_PERIOD_IDS": ["low", "high"],
            "PROBLEM_PERIOD_WEIGHTS": [0.5, 0.5],
        },
    )
    problem.target.all_periods.cogeneration()

    report = problem.report(include_periods=True, include_weighted_average=True)

    rows = {
        row.period_id: row
        for row in report.targets
        if row.turbine_converged is not None
    }
    assert rows["low"].turbine_result.stages[0].process_duty == pytest.approx(
        5000.0, rel=1e-6
    )
    assert rows["high"].turbine_result.stages[0].process_duty == pytest.approx(
        6000.0, rel=1e-6
    )
    assert rows["weighted_average"].turbine_result is None
