"""Focused tests for shared HeatExchangerNetworkLabel base-model helper methods."""

from __future__ import annotations

import logging
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import OpenPinch.analysis.heat_exchanger_networks.models.base as base
from OpenPinch.analysis.heat_exchanger_networks.models._base import alpha as base_alpha
from OpenPinch.analysis.heat_exchanger_networks.models._base import (
    execution as base_execution,
)
from OpenPinch.analysis.heat_exchanger_networks.models._base.approach import (
    _cold_utility_inlet_approach_temperature,
    _cold_utility_outlet_approach_temperature,
    _hot_utility_inlet_approach_temperature,
    _hot_utility_outlet_approach_temperature,
    _recovery_approach_temperature,
    _set_minimum_approach_temperatures,
    _weighted_state_average,
)
from OpenPinch.analysis.heat_exchanger_networks.models._base.area import (
    _post_process_lmtd,
)
from OpenPinch.analysis.heat_exchanger_networks.models._base.execution import (
    _set_value,
    _solver_value,
)
from OpenPinch.analysis.heat_exchanger_networks.models._base.parameters import (
    _normalise_state_arrays,
)
from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.pyomo_model import PyomoModel


class _Harness(base.BaseHeatExchangerNetworkModel):
    def setup(self) -> None:
        pass

    def set_preprocessing(self) -> None:
        pass

    def set_stage_wise_superstructure(self) -> None:
        pass

    def set_obj(self) -> None:
        pass

    def get_post_process(self) -> None:
        self.post_processed = True


class _PyomoLikeVariable:
    """A Pyomo variable's value interface; a fixed one is a parameter."""

    def __init__(self, *, value, lb=None, ub=None, fixed=False):
        self.value = value
        self.lb = lb
        self.ub = ub
        self.fixed = fixed

    def is_variable_type(self):
        return True

    def fix(self, value):
        self.value = value
        self.fixed = True

    def set_value(self, value, skip_validation=False):
        self.value = value


class _FakeModel:
    def __init__(self, *, solve_status=1, objective=1.0, fail_solve=False):
        self.options = SimpleNamespace(
            IMODE=None,
            SOLVER=None,
            SOLVESTATUS=solve_status,
            objfcnval=objective,
        )
        self._path = "fake-path"
        self.fail_solve = fail_solve
        self.intermediates = []
        self.variables = []
        self.equations = []
        self.solved = False

    def Intermediate(self, value):
        self.intermediates.append(value)
        return value

    def Var(self, *, value, ub, lb, name):
        self.variables.append({"value": value, "ub": ub, "lb": lb, "name": name})
        return value

    def Equation(self, expression):
        self.equations.append(expression)
        return expression

    def solve(self, disp=False):
        if self.fail_solve:
            raise RuntimeError("fake solve failure")
        self.solved = True


def _model() -> _Harness:
    model = object.__new__(_Harness)
    model.name = "base-helper"
    model.solver = "couenne"
    model.tol = 1e-6
    model.dTmin = 10.0
    model.min_dqda = 0.1
    model.post_processed = False
    model.mSuccess = 0
    return model


def _solver_arrays(**overrides):
    arrays = {
        "period_ids": ["base", "peak"],
        "period_weights": [1.0, 3.0],
        "T_h_in_period": [[200.0], [210.0]],
        "T_h_out_period": [[100.0], [110.0]],
        "f_h_period": [[1.0], [1.5]],
        "htc_h_period": [[1.0], [1.0]],
        "h_cost_period": [[0.0], [0.0]],
        "T_h_cont_period": [[4.0], [5.0]],
        "T_c_in_period": [[50.0], [60.0]],
        "T_c_out_period": [[150.0], [160.0]],
        "f_c_period": [[1.0], [1.5]],
        "htc_c_period": [[1.0], [1.0]],
        "c_cost_period": [[0.0], [0.0]],
        "T_c_cont_period": [[6.0], [7.0]],
        "T_hu_in_period": [[250.0], [260.0]],
        "T_hu_out_period": [[230.0], [240.0]],
        "htc_hu_period": [[1.0], [1.0]],
        "hu_cost_period": [[1.0], [1.0]],
        "T_hu_cont_period": [[2.0], [3.0]],
        "T_cu_in_period": [[20.0], [25.0]],
        "T_cu_out_period": [[40.0], [45.0]],
        "htc_cu_period": [[1.0], [1.0]],
        "cu_cost_period": [[1.0], [1.0]],
        "T_cu_cont_period": [[3.0], [4.0]],
    }
    arrays.update(overrides)
    return SimpleNamespace(arrays=arrays)


def _set_success_diagnostics(model: _Harness) -> None:
    for name in (
        "TAC",
        "n_units",
        "n_recovery_units",
        "T_h",
        "T_c",
        "theta_1",
        "theta_2",
        "Q_r",
        "z",
        "LMTD_r",
        "area_r",
        "Q_r_total",
        "Q_c",
        "z_cu",
        "LMTD_cu",
        "area_cu",
        "Q_cu_total",
        "Q_h",
        "z_hu",
        "LMTD_hu",
        "area_hu",
        "Q_hu_total",
    ):
        setattr(model, name, name)


def test_constructor_configures_backend_and_runs_concrete_setup(monkeypatch):
    fake_model = _FakeModel()
    calls = []

    monkeypatch.setattr(
        base_execution.backend,
        "require_solver_backend",
        lambda solver: calls.append(("require", solver)),
    )
    monkeypatch.setattr(
        base_execution, "PyomoModel", lambda: calls.append(("create",)) or fake_model
    )
    monkeypatch.setattr(
        base_execution.backend,
        "configure_pyomo_solver",
        lambda model, solver, solver_options=None: (
            calls.append(("configure", model, solver, solver_options)) or _solver_run()
        ),
    )

    model = _Harness(
        name="constructed",
        framework="PDM",
        solver="couenne",
        solver_arrays=_solver_arrays(),
        dTmin=8.0,
        z_restriction=None,
        min_dqda=0.05,
        minimisation_goal="total cost",
        non_isothermal_model=False,
        integers=True,
        tol=1e-5,
        solver_options={"time_limit": 10},
    )

    assert model.m is fake_model
    assert model.mSuccess == 0
    assert model.solver_run.name == "fake"
    assert calls == [
        ("require", "couenne"),
        ("create",),
        ("configure", fake_model, "couenne", {"time_limit": 10}),
    ]


def test_base_model_scalar_helpers_and_value_assignment():
    model = _model()

    assert _solver_value(model, [3.5]) == pytest.approx(3.5)
    assert _solver_value(model, 4.5) == pytest.approx(4.5)

    variable = _PyomoLikeVariable(value=5.0, lb=0.0, ub=10.0)
    _set_value(model, variable, 99.0)
    assert variable.value == pytest.approx(10.0)
    _set_value(model, variable, -99.0)
    assert variable.value == pytest.approx(0.0)

    # A parameter (a fixed variable) takes the value unclamped and stays fixed.
    parameter = _PyomoLikeVariable(value=1.0, fixed=True)
    _set_value(model, parameter, 7.0)
    assert parameter.value == pytest.approx(7.0)
    assert parameter.fixed

    # Both ends positive: the true log-mean, whatever the old gate said.
    log_mean_20_10 = 10.0 / np.log(2.0)
    for allowed in (False, True):
        assert _post_process_lmtd(
            model, 20.0, 10.0, 0.5, formula_allowed=allowed
        ) == pytest.approx(0.5 * log_mean_20_10)
    # Equal ends give the arithmetic mean; an inactive match gives 0.
    assert _post_process_lmtd(
        model, 12.0, 12.0, 1.0, formula_allowed=True
    ) == pytest.approx(12.0)
    assert _post_process_lmtd(model, 20.0, 10.0, 0.0, formula_allowed=True) == 0.0
    # A crossed or touching end has no log-mean: the match gets no area, so
    # verification reports its duty without area instead of a made-up one.
    for crossed in ((0.0, 10.0), (10.0, -1.0)):
        assert _post_process_lmtd(model, *crossed, 0.5, formula_allowed=False) == 0.0


@pytest.mark.parametrize("ends", [(10.0, 50.0), (50.0, 10.0)])
def test_post_process_lmtd_uses_the_log_mean_at_the_approach_bound(ends):
    model = SimpleNamespace(tol=1e-6)

    lmtd = _post_process_lmtd(model, *ends, 1.0, formula_allowed=False)

    assert lmtd == pytest.approx(24.853, abs=1e-3)


def _alpha_state(hot_temperatures, cold_temperatures) -> _Harness:
    """One hot and one cold stream matched in every stage of a solved model."""
    model = _model()
    model.I = 1
    model.J = 1
    model.S = len(hot_temperatures) - 1
    model.non_isothermal_model = False
    model.T_h = [[[t] for t in hot_temperatures]]
    model.T_c = [[[t] for t in cold_temperatures]]
    model.z = [[[[1.0] for _ in range(model.S)]]]
    model.Q_r = [[[[10.0] for _ in range(model.S)]]]
    model.alpha = []
    return model


def test_get_alpha_values_solves_the_flow_on_system(monkeypatch):
    # P_h at stage 1 is (150 - 100) / (150 - 50) = 0.5 and P_c at stage 0 is
    # (180 - 120) / (200 - 120) = 0.75. Stage 0 has no cold flow-on and
    # stage 1 no hot flow-on, so
    #   alpha_0 = 1 - 0.5 * 0.5 * alpha_1 and alpha_1 = 1 - 0.5 * 0.75 * alpha_0,
    # giving alpha_0 = 24/29 and alpha_1 = 20/29.
    model = _alpha_state([200.0, 150.0, 100.0], [180.0, 120.0, 50.0])
    # The system is linear: it is solved directly, with no solver model.
    monkeypatch.setattr(
        base_execution,
        "PyomoModel",
        lambda: pytest.fail("built a solver model"),
    )

    alpha = model.get_alpha_values()

    assert alpha[0][0][0][0] == pytest.approx(24 / 29, abs=1e-6)
    assert alpha[0][0][1][0] == pytest.approx(20 / 29, abs=1e-6)
    assert model.alpha is alpha
    assert base_alpha.alpha_value(model, 0, 0, 1) == pytest.approx(20 / 29, abs=1e-6)


def test_get_alpha_values_returns_cached_values():
    model = _model()
    model.alpha = [[[[0.25]]]]

    assert model.get_alpha_values() == [[[[0.25]]]]


def test_a_single_stage_has_no_flow_on():
    # One stage is both the last and the first: neither gamma takes flow-on.
    model = _alpha_state([200.0, 150.0], [180.0, 120.0])

    assert model.get_alpha_values()[0][0][0][0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("hot_temperatures", "cold_temperatures"),
    [
        # P_h at stage 1 is 6 and P_c at stage 0 is 0, so alpha_0 = -2.
        pytest.param([200.0, 150.0, 90.0], [140.0, 140.0, 140.0], id="out-of-range"),
        # P_h at stage 1 and P_c at stage 0 are both 2: alpha_0 + alpha_1 = 1
        # twice over, with no unique solution.
        pytest.param([200.0, 150.0, 50.0], [250.0, 150.0, 100.0], id="singular"),
    ],
)
def test_get_alpha_values_falls_back_to_neutral_alpha(
    hot_temperatures, cold_temperatures, caplog
):
    model = _alpha_state(hot_temperatures, cold_temperatures)

    with caplog.at_level(logging.WARNING, logger=base_alpha.__name__):
        assert model.get_alpha_values() is None

    # Nothing is cached, and ranking uses neutral alpha.
    assert model.alpha == []
    assert base_alpha.alpha_value(model, 0, 0, 0) == 1.0
    assert any("neutral alpha" in record.getMessage() for record in caplog.records)


def test_blank_and_solver_array_state_helpers_validate_static_arrays():
    model = _model()
    model.set_blank_input_parameters()
    assert model.period_ids.tolist() == ["0"]
    assert model.period_weight_sum == pytest.approx(1.0)

    model.solver_arrays = _solver_arrays()
    model.get_model_parameters_from_solver_arrays()

    assert model.N_periods == 2
    assert model.period_weight_sum == pytest.approx(4.0)
    assert model.T_h_in.tolist() == [200.0]
    assert model.dT_r_period.tolist() == [[[10.0]], [[12.0]]]
    # Heaters are [utility][cold] and coolers [hot][utility].
    assert model.dT_hu.tolist() == [[8.0]]
    assert model.dT_cu.tolist() == [[7.0]]
    assert _recovery_approach_temperature(model, 0, 0, period_idx=1) == pytest.approx(
        12.0
    )
    assert _hot_utility_inlet_approach_temperature(
        model, 0, 0, period_idx=1
    ) == pytest.approx(10.0)
    assert _hot_utility_outlet_approach_temperature(
        model, 0, 0, period_idx=1, heat_duty=1.0
    ) == pytest.approx(10.0)
    assert _cold_utility_inlet_approach_temperature(
        model, 0, 0, period_idx=1
    ) == pytest.approx(9.0)
    assert _cold_utility_outlet_approach_temperature(
        model, 0, 0, period_idx=1, heat_duty=1.0
    ) == pytest.approx(9.0)
    assert _weighted_state_average(model, [10.0, 30.0]) == pytest.approx(25.0)

    fallback = _model()
    assert _recovery_approach_temperature(fallback, 0, 0) == pytest.approx(10.0)
    fallback.dT_r = np.array([[11.0]])
    assert _recovery_approach_temperature(fallback, 0, 0) == pytest.approx(11.0)


@pytest.mark.parametrize(
    ("arrays", "message"),
    [
        ({}, "period_ids is required"),
        ({"period_ids": ["0"]}, "period_weights is required"),
        (
            {"period_ids": [], "period_weights": []},
            "at least one state",
        ),
        (
            {"period_ids": ["0"], "period_weights": [1.0, 2.0]},
            "weight count",
        ),
        (
            {"period_ids": ["0"], "period_weights": [float("nan")]},
            "must be finite",
        ),
        (
            {"period_ids": ["0"], "period_weights": [0.0]},
            "positive sum",
        ),
        (
            _solver_arrays(T_h_in_period=[]).arrays,
            "T_h_in_period is required",
        ),
        (
            _solver_arrays(T_h_in_period=[1.0]).arrays,
            "must be indexed by operating period",
        ),
        (
            _solver_arrays(T_h_in_period=[[1.0]]).arrays,
            "expected 2",
        ),
    ],
)
def test_normalise_state_arrays_rejects_invalid_solver_arrays(arrays, message: str):
    model = _model()
    model.solver_arrays = SimpleNamespace(arrays=arrays)
    for name, values in arrays.items():
        setattr(model, name, np.array(values, copy=True))

    with pytest.raises(ValueError, match=message):
        _normalise_state_arrays(model)


def test_minimum_approach_temperatures_use_dTmin_when_utilities_are_absent():
    model = _model()
    model.N_periods = 1
    model.T_h_cont_period = np.array([[4.0]])
    model.T_c_cont_period = np.array([[6.0]])
    model.T_hu_cont_period = np.array([[]], dtype=object)
    model.T_cu_cont_period = np.array([[]], dtype=object)

    _set_minimum_approach_temperatures(model)

    assert model.dT_hu.tolist() == [[11.0]]
    assert model.dT_cu.tolist() == [[9.0]]


def test_match_restrictions_accept_supported_shapes_and_reject_invalid_shape():
    model = _model()
    model.I = 1
    model.J = 1
    model.S = 1
    model.NHU = 1
    model.NCU = 2
    model.tol = 0.1
    model.z_feasible = [[[1]]]
    model.z_hu_feasible = [[1]]
    model.z_cu_feasible = [[1, 1]]

    model.set_match_restrictions(None)
    assert model.z_allowed == [[[1]]]
    assert model.z_hu_allowed == [[1]]
    assert model.z_cu_allowed == [[1, 1]]

    # Utility restrictions are [stream][utility].
    model.set_match_restrictions([[[[0]]], [[0]], [[1, 0]]])
    assert model.z_allowed == [[[0]]]
    assert model.z_hu_allowed == [[0]]
    assert model.z_cu_allowed == [[1, 0]]

    model.set_match_restrictions([[[[[1.0]]]], [[[1.0]]], [[[0.0], [1.0]]]])
    assert model.z_allowed == [[[1]]]
    assert model.z_hu_allowed == [[1]]
    assert model.z_cu_allowed == [[0, 1]]

    with pytest.raises(ValueError, match="Invalid restriction type"):
        model.set_match_restrictions([[[[object()]]], None, None])


def _solver_run(*, failure_reason=None, status=1, objective_value=1.0):
    return backend.SolverRun(
        name="fake",
        status=status,
        objective_value=objective_value,
        solve_time=0.25,
        failure_reason=failure_reason,
    )


def _solve_reporting_fake_options(m, **_kwargs):
    # As solve_pyomo_model does, report the model's status and objective.
    return _solver_run(
        status=m.options.SOLVESTATUS, objective_value=m.options.objfcnval
    )


def test_optimise_handles_failure_negative_success_and_unsolved_paths(monkeypatch):
    model = _model()
    model.m = _FakeModel(solve_status=1, objective=1.0)

    monkeypatch.setattr(
        base_execution.backend,
        "solve_pyomo_model",
        lambda *args, **kwargs: _solver_run(failure_reason="missing solver"),
    )
    model.optimise(print_output=False)
    assert model.mSuccess == 0
    assert model.post_processed is False

    monkeypatch.setattr(
        base_execution.backend,
        "solve_pyomo_model",
        _solve_reporting_fake_options,
    )
    model.m = _FakeModel(solve_status=1, objective=-1.0)
    model.optimise(print_output=False)
    assert model.mSuccess == 0
    assert model.solver_run.failure_reason == "negative objective value"

    model.m = _FakeModel(solve_status=2, objective=1.0)
    model.optimise(print_output=False)
    assert model.mSuccess == 2

    model.m = _FakeModel(solve_status=1, objective=1.0)
    model.post_processed = False
    _set_success_diagnostics(model)
    model.optimise(print_output=True)
    assert model.mSuccess == 1
    assert model.post_processed is True


def test_optimise_removes_run_directory_on_success_and_failure(monkeypatch, tmp_path):
    model = _model()
    model.get_post_process = lambda: setattr(model, "post_processed", True)

    def solve(m, **_kwargs):
        # The run directory is still there while the model solves.
        assert (tmp_path / m.name).is_dir()
        return _solver_run(failure_reason=None if m.name == "solved" else "bad")

    monkeypatch.setattr(base_execution.backend, "solve_pyomo_model", solve)
    for name in ("solved", "failed"):
        (tmp_path / name).mkdir()
        model.m = _FakeModel(solve_status=1, objective=1.0)
        model.m.name = name
        model.m._path = str(tmp_path / name)
        model.post_processed = False

        model.optimise(print_output=False)

        assert model.post_processed is (name == "solved")
        assert not (tmp_path / name).exists()

    def broken_solve(m, **_kwargs):
        raise RuntimeError("solver crashed")

    (tmp_path / "crashed").mkdir()
    model.m = _FakeModel()
    model.m._path = str(tmp_path / "crashed")
    monkeypatch.setattr(base_execution.backend, "solve_pyomo_model", broken_solve)
    with pytest.raises(RuntimeError, match="solver crashed"):
        model.optimise(print_output=False)
    assert not (tmp_path / "crashed").exists()


def test_cleanup_model_run_directory_is_idempotent_and_allows_a_resolve(
    monkeypatch, tmp_path
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model = SimpleNamespace(_path=str(run_dir))

    backend.cleanup_model_run_directory(model)
    backend.cleanup_model_run_directory(model)
    backend.cleanup_model_run_directory(None)
    assert not run_dir.exists()

    # A model solved again after cleanup gets its run directory back.
    def factory(_name):
        assert run_dir.is_dir()
        raise RuntimeError("stop after the directory check")

    pyomo = SimpleNamespace(SolverFactory=factory)
    monkeypatch.setattr(backend, "require_synthesis_dependency", lambda *a, **k: pyomo)
    run = backend.solve_pyomo_model(model, solver_name="couenne")
    assert run.failure_reason == "stop after the directory check"
    assert run_dir.is_dir()


def _ipopt_available() -> bool:
    try:
        backend.require_solver_backend("ipopt-pyomo")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


@pytest.mark.solver
# A string condition is evaluated when the test runs, so collecting this module
# does not import the solver packages.
@pytest.mark.skipif(
    "not _ipopt_available()",
    reason="needs the ipopt binary and the pyomo package",
)
@pytest.mark.parametrize(
    ("solver_options", "termination_condition", "converged"),
    [({}, "optimal", True), ({"max_iter": 1}, "maxIterations", False)],
)
def test_pyomo_solve_records_whether_the_solver_converged(
    solver_options, termination_condition: str, converged: bool
):
    model = PyomoModel()
    try:
        x = model.Var(value=5.0)
        y = model.Var(value=-3.0)
        model.Equation(x**2 + y**2 >= 2)
        model.Minimize((x - 1) ** 4 + (y - 2) ** 4 + x * y)
        backend.configure_pyomo_solver(
            model, "ipopt-pyomo", solver_options=solver_options
        )
        run = backend.solve_pyomo_model(model, solver_name="ipopt-pyomo")
    finally:
        backend.cleanup_model_run_directory(model)

    # An iteration limit still passes, but is not converged.
    assert run.status == 1
    assert run.failure_reason is None
    assert run.termination_condition == termination_condition
    assert run.converged is converged


def test_optimise_warns_when_the_solver_stopped_before_converging(monkeypatch, caplog):
    model = _model()
    model.m = _FakeModel(solve_status=1, objective=1.0)
    model.get_post_process = lambda: None
    monkeypatch.setattr(
        base_execution.backend,
        "solve_pyomo_model",
        lambda *args, **kwargs: replace(
            _solver_run(), termination_condition="maxIterations"
        ),
    )

    with caplog.at_level("INFO", logger=base_execution.logger.name):
        model.optimise(print_output=False)

    # The solution is kept, but not reported as a converged success.
    assert model.mSuccess == 1
    assert "[Success]" not in caplog.text
    assert "maxIterations" in caplog.text


def test_optimise_fails_a_passing_solve_that_has_no_objective_value(monkeypatch):
    # A passing status with nothing loaded has no objective to judge; it
    # raised TypeError and lost the whole task.
    model = _model()
    model.m = _FakeModel(solve_status=1, objective=1.0)
    model.get_post_process = lambda: None
    monkeypatch.setattr(
        base_execution.backend,
        "solve_pyomo_model",
        lambda *args, **kwargs: _solver_run(objective_value=None),
    )

    model.optimise(print_output=False)

    assert model.mSuccess == 0
    assert model.solver_run.failure_reason == "solver returned no objective value"


def test_output_to_cmd_line_logs_successful_solve(caplog):
    model = _model()
    model.mSuccess = 0
    model.output_to_cmd_line()

    model.mSuccess = 1
    model.m = _FakeModel(objective=12.0)
    model.solver_run = _solver_run(objective_value=12.0)
    _set_success_diagnostics(model)

    with caplog.at_level("INFO", logger=base_execution.logger.name):
        model.output_to_cmd_line()

    assert "Successful Solve.Path" in caplog.text
    assert "Q_hu total" in caplog.text


@pytest.mark.parametrize("inactive_by", ["pinch decomposition", "zero duty"])
def test_utility_approach_equations_skip_streams_inactive_in_a_period(inactive_by):
    from OpenPinch.analysis.heat_exchanger_networks.models._base.approach import (
        _set_multiperiod_utility_approach_equations,
    )

    model = _model()
    model.m = _FakeModel()
    model.N_periods = 2
    model.period_ids = ["base", "peak"]
    model.I = model.J = model.S = model.NHU = model.NCU = 1
    model.z_hu_allowed = model.z_cu_allowed = [[1]]
    model.z_hu = model.z_cu = [[1]]
    model.T_hu_in_period = model.T_hu_out_period = np.array([[250.0], [250.0]])
    model.T_cu_in_period = np.array([[20.0], [20.0]])
    model.T_cu_out_period = np.array([[40.0], [40.0]])
    model.T_hu_cont_period = model.T_cu_cont_period = np.array([[2.0], [2.0]])
    model.T_h_cont_period = model.T_c_cont_period = np.array([[3.0], [3.0]])
    # The pinch decomposition clips H1 to 0 K in the peak period, where it is
    # absent from this side of the pinch.
    model.T_h_in_period = np.array([[200.0], [0.0]])
    model.T_h_out_period = np.array([[100.0], [0.0]])
    model.T_c_in_period = np.array([[50.0], [50.0]])
    model.T_c_out_period = np.array([[150.0], [150.0]])
    model.T_h_by_period = [[[200.0, 120.0]], [[0.0, 0.0]]]
    model.T_c_by_period = [[[140.0, 50.0]], [[140.0, 50.0]]]
    if inactive_by == "pinch decomposition":
        model.z_i_active_period = [[1], [0]]
        model.z_j_active_period = [[1], [1]]
    else:
        # A stagewise model has no activity flags: H1 has no duty in peak.
        model.Qtot_sh_period = np.array([[100.0], [0.0]])
        model.Qtot_sc_period = np.array([[100.0], [100.0]])
    model.Q_h_by_period = model.Q_c_by_period = [[[0.0]], [[0.0]]]

    _set_multiperiod_utility_approach_equations(model)

    # Both terminals of the heater in both periods, the cooler in base only.
    assert len(model.m.equations) == 6
    assert all(model.m.equations)
