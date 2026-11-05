"""Turbine convergence and diagnostics reach the cogeneration target and reports."""

import warnings

import pytest

from OpenPinch import PinchProblem
from OpenPinch.analysis.power import service, steam_turbine
from OpenPinch.contracts.reporting import TargetResults
from OpenPinch.domain.enums import TurbineModel
from OpenPinch.domain.value import Value
from OpenPinch.presentation.reporting.workbook import build_summary_dataframe

# The small Varbanov model switches from back-pressure to condensing on a wet exit.
# With a 2 bar, 185 degC inlet and one level at 85 degC the condensing branch does
# less work at the switch, so near 14280 kW neither branch is self-consistent and
# the extraction mass balance alternates. The gap is about 45 kW wide.
ALTERNATING_INLET = {"POWER_TURB_P_IN": 2.0, "POWER_TURB_T_IN": 185.0}
NON_CONVERGING_DUTY = 14280.0
CONVERGING_DUTY = 12000.0
# Work target in kW at NON_CONVERGING_DUTY: the last iteration, pinned so the
# number stays reported.
NON_CONVERGING_WORK = 1172.9464728328599
# With the default 90 bar, 450 degC inlet and one level at 89.5 degC, the small
# model is in its band at this duty and out of it at the higher one, and the
# large model the other way round.
SMALL_BAND_DUTY = 6000.0
LARGE_BAND_DUTY = 9000.0
# The auto-switching model this replaced alternated across 7200 to 7700 kW here.
OLD_GAP_DUTIES = (7200.0, 7500.0, 7700.0)
SMALL = "varbanov_small_turbine_cogeneration"
LARGE = "varbanov_large_turbine_cogeneration"


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


def _utility(name, kind, t_supply, t_target):
    return {
        "name": name,
        "type": kind,
        "t_supply": {"value": t_supply, "unit": "degC"},
        "t_target": {"value": t_target, "unit": "degC"},
        "dt_cont": {"value": 1, "unit": "degC"},
        "price": {"value": 10, "unit": "$/MWh"},
        "htc": {"value": 1, "unit": "kW/m^2/degC"},
        "heat_flow": None,
    }


def _problem(heat_flow, options=None, level=(90, 89.5), cold_target=70):
    return PinchProblem(
        {
            "streams": [
                _stream("Cold", 40, cold_target, heat_flow),
                _stream("Hot", 35, 30, {"value": 100, "unit": "kW"}),
            ],
            "utilities": [
                _utility("LP", "Hot", *level),
                _utility("CU", "Cold", 10, 10.1),
            ],
            "options": options or {},
        }
    )


def _alternating_problem(heat_flow, options=None):
    return _problem(
        heat_flow,
        options={**ALTERNATING_INLET, **(options or {})},
        level=(85.5, 85),
        cold_target=75,
    )


def _single_period_problem(duty):
    problem = _alternating_problem({"value": duty, "unit": "kW"})
    problem.target.direct_heat_integration()
    return problem


def _default_inlet_problem(duty):
    problem = _problem({"value": duty, "unit": "kW"})
    problem.target.direct_heat_integration()
    return problem


def _multiperiod_problem():
    return _alternating_problem(
        {"values": [CONVERGING_DUTY, NON_CONVERGING_DUTY], "unit": "kW"},
        options={
            "PROBLEM_PERIOD_IDS": ["low", "high"],
            "PROBLEM_PERIOD_WEIGHTS": [0.5, 0.5],
        },
    )


def _turbine_row(rows):
    return next(row for row in rows if row.turbine_converged is not None)


def test_non_converged_mass_balance_warns_and_reaches_the_target():
    problem = _single_period_problem(NON_CONVERGING_DUTY)

    with pytest.warns(UserWarning, match="mass balance did not converge"):
        target = problem.target.varbanov_small_turbine_cogeneration()

    assert target.turbine_converged is False
    assert len(target.turbine_diagnostics) == 1
    assert "did not converge in 50 iterations" in target.turbine_diagnostics[0]


def test_non_converged_run_keeps_its_numeric_target():
    problem = _single_period_problem(NON_CONVERGING_DUTY)

    with pytest.warns(UserWarning, match="mass balance did not converge"):
        target = problem.target.varbanov_small_turbine_cogeneration()

    assert target.work_target == pytest.approx(NON_CONVERGING_WORK, rel=1e-6)


def test_non_converged_run_is_flagged_on_the_result_and_the_report():
    problem = _single_period_problem(NON_CONVERGING_DUTY)

    with pytest.warns(UserWarning, match="mass balance did not converge"):
        target = problem.target.varbanov_small_turbine_cogeneration()

    row = _turbine_row(problem.results.targets)
    assert row.turbine_converged is False
    assert row.turbine_diagnostics == target.turbine_diagnostics

    report = problem.report()
    assert len(report.warnings) == 1
    assert row.scope in report.warnings[0]
    assert target.turbine_diagnostics[0] in report.warnings[0]


def test_converged_run_is_silent_and_marked_converged():
    problem = _single_period_problem(CONVERGING_DUTY)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        target = problem.target.varbanov_small_turbine_cogeneration()
        report = problem.report()

    assert target.turbine_converged is True
    assert target.turbine_diagnostics == []
    assert report.warnings == []


def test_targets_without_a_turbine_run_have_no_turbine_status():
    problem = _single_period_problem(CONVERGING_DUTY)

    for row in problem.results.targets:
        assert row.turbine_converged is None
        assert row.turbine_diagnostics == []


def test_solver_diagnostic_on_a_converged_run_warns_and_is_carried(monkeypatch):
    # The above-Pinch path has no clamp of its own, and the sink clamp is on the
    # below-Pinch path that no target reaches, so this stubs the solver's result.
    message = "Sink temperature -5 degC is outside water's saturation range."
    original = steam_turbine.MultiStageSteamTurbine.solve

    def solve_with_diagnostic(self, *args, **kwargs):
        total_work, details = original(self, *args, **kwargs)
        return total_work, {**details, "diagnostics": [message]}

    monkeypatch.setattr(
        steam_turbine.MultiStageSteamTurbine, "solve", solve_with_diagnostic
    )
    problem = _single_period_problem(CONVERGING_DUTY)

    with pytest.warns(UserWarning, match="outside water's saturation range"):
        target = problem.target.varbanov_small_turbine_cogeneration()

    assert target.turbine_converged is True
    assert target.turbine_diagnostics == [message]


def test_multiperiod_status_stays_on_its_period_and_the_average_flags_it():
    problem = _multiperiod_problem()
    with pytest.warns(UserWarning, match="period_id 'high'.*did not converge"):
        problem.target.all_periods.varbanov_small_turbine_cogeneration()

    report = problem.report(include_periods=True, include_weighted_average=True)

    rows = {
        row.period_id: row
        for row in report.targets
        if row.turbine_converged is not None
    }
    assert rows["low"].turbine_converged is True
    assert rows["low"].turbine_diagnostics == []
    assert rows["high"].turbine_converged is False
    assert "did not converge" in rows["high"].turbine_diagnostics[0]
    weighted = rows["weighted_average"]
    assert weighted.turbine_converged is False
    assert weighted.turbine_diagnostics == [
        f"Period high: {rows['high'].turbine_diagnostics[0]}"
    ]


def test_turbine_status_survives_a_json_round_trip():
    problem = _single_period_problem(NON_CONVERGING_DUTY)
    with pytest.warns(UserWarning, match="mass balance did not converge"):
        problem.target.varbanov_small_turbine_cogeneration()
    row = _turbine_row(problem.results.targets)

    restored = TargetResults.model_validate_json(row.model_dump_json())

    assert restored.turbine_converged is False
    assert restored.turbine_diagnostics == row.turbine_diagnostics
    assert isinstance(restored.work_target, Value)


def test_excel_summary_flags_a_non_converged_run():
    problem = _single_period_problem(NON_CONVERGING_DUTY)
    with pytest.warns(UserWarning, match="mass balance did not converge"):
        problem.target.varbanov_small_turbine_cogeneration()
    row = _turbine_row(problem.results.targets)

    summary = build_summary_dataframe([row])

    assert not summary.iloc[0]["Turbine Converged"]
    assert summary.iloc[0]["Turbine Converged"] is not None


def test_a_reused_target_with_no_viable_stage_drops_the_earlier_turbine_status(
    monkeypatch,
):
    problem = _single_period_problem(NON_CONVERGING_DUTY)
    with pytest.warns(UserWarning, match="mass balance did not converge"):
        target = problem.target.varbanov_small_turbine_cogeneration()
    monkeypatch.setattr(service, "_preprocess_utilities", lambda *_a, **_k: None)

    result = service.get_power_cogeneration_above_pinch(target)

    assert result.turbine_converged is None
    assert result.turbine_diagnostics == []


@pytest.mark.parametrize("method", [SMALL, LARGE])
@pytest.mark.parametrize("duty", OLD_GAP_DUTIES)
def test_the_old_alternating_gap_now_converges(method, duty):
    problem = _default_inlet_problem(duty)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        target = getattr(problem.target, method)()

    assert target.turbine_converged is True


@pytest.mark.parametrize(
    ("method", "duty"), [(SMALL, SMALL_BAND_DUTY), (LARGE, LARGE_BAND_DUTY)]
)
def test_a_stage_inside_the_band_is_silent(method, duty):
    problem = _default_inlet_problem(duty)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        target = getattr(problem.target, method)()
        report = problem.report()

    assert target.turbine_diagnostics == []
    assert report.warnings == []


@pytest.mark.parametrize(
    ("method", "duty", "band", "other", "work"),
    [
        (SMALL, LARGE_BAND_DUTY, "at or below 2000 kW", LARGE, 2802.285),
        (LARGE, SMALL_BAND_DUTY, "above 2000 kW", SMALL, 1169.3),
    ],
)
def test_out_of_range_use_warns_once_and_still_reports_the_number(
    method, duty, band, other, work
):
    problem = _default_inlet_problem(duty)

    with pytest.warns(UserWarning, match="outside its range"):
        target = getattr(problem.target, method)()

    assert target.turbine_converged is True
    assert target.work_target == pytest.approx(work, abs=0.1)
    assert len(target.turbine_diagnostics) == 1
    message = target.turbine_diagnostics[0]
    assert band in message
    assert f"{target.work_target:.0f} kW" in message
    assert other in message

    report = problem.report()
    assert len(report.warnings) == 1
    assert message in report.warnings[0]


def test_the_auto_switching_model_is_gone():
    problem = _default_inlet_problem(SMALL_BAND_DUTY)

    assert not hasattr(problem.target, "varbanov_cogeneration")
    assert not hasattr(TurbineModel, "VARBANOV")
