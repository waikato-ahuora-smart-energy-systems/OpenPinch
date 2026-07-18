"""Focused tests for shared HeatExchangerNetworkLabel base-model helper methods."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

import OpenPinch.analysis.heat_exchanger_networks.models.base as base
from OpenPinch.analysis.heat_exchanger_networks.models._base import alpha as base_alpha
from OpenPinch.analysis.heat_exchanger_networks.models._base import (
    execution as base_execution,
)
from OpenPinch.analysis.heat_exchanger_networks.solver import backend


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


class GKVariable:
    def __init__(self, *, lower=0.0, upper=10.0):
        self.lower = lower
        self.upper = upper
        self.VALUE = SimpleNamespace(value=None)

    def __getitem__(self, index):
        if index == 0:
            return self.VALUE.value if self.VALUE.value is not None else 0.0
        raise IndexError(index)


class GKParameter:
    def __init__(self):
        self.VALUE = SimpleNamespace(value=None)

    def __getitem__(self, index):
        if index == 0:
            return self.VALUE.value if self.VALUE.value is not None else 0.0
        raise IndexError(index)


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
    model.solver = "apopt"
    model.tol = 1e-6
    model.dTmin = 10.0
    model.min_dqda = 0.1
    model.post_processed = False
    model.mSuccess = 0
    return model


def _configure_equation_state(model: _Harness, *, non_isothermal: bool) -> None:
    model.I = 1
    model.J = 1
    model.S = 3
    model.non_isothermal_model = non_isothermal
    model.m = _FakeModel()
    model.T_h = [[200.0, 180.0, 160.0, 140.0]]
    model.T_c = [[50.0, 70.0, 90.0, 110.0]]
    model.T_h_out_x = [[[190.0, 170.0, 150.0]]]
    model.T_c_out_y = [[[60.0, 80.0, 100.0]]]
    model.z = [[[1.0, 0.0, 1.0]]]
    model.z_allowed = [[[1, 0, 1]]]
    model.Q_r = [[[10.0, 0.0, 5.0]]]
    model.theta_1 = [[[20.0, 20.0, 20.0]]]
    model.theta_2 = [[[10.0, 10.0, 10.0]]]
    model.U_r = [[1.0]]


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


def _configure_postoptimisation_state(model: _Harness, *, non_isothermal: bool) -> None:
    model.I = 1
    model.J = 1
    model.S = 3
    model.non_isothermal_model = non_isothermal
    model.T_h = [[[200.0], [180.0], [80.0], [70.0]]]
    model.T_c = [[[50.0], [70.0], [90.0], [110.0]]]
    model.T_h_out_x = [[[[190.0], [170.0], [75.0]]]]
    model.T_c_out_y = [[[[60.0], [80.0], [95.0]]]]
    model.z = [[[[1.0], [0.0], [1.0]]]]
    model.Q_r = [[[[10.0], [0.0], [5.0]]]]
    model.alpha = []


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
        "create_gekko_model",
        lambda remote=False: calls.append(("create", remote)) or fake_model,
    )
    monkeypatch.setattr(
        base_execution.backend,
        "configure_gekko_solver",
        lambda model, solver, solver_options=None: (
            calls.append(("configure", model, solver, solver_options)) or _solver_run()
        ),
    )

    model = _Harness(
        name="constructed",
        framework="PDM",
        solver="apopt",
        solver_arrays=_solver_arrays(),
        dTmin=8.0,
        z_restriction=None,
        min_dqda=0.05,
        minimisation_goal="total cost",
        non_isothermal_model=False,
        integers=True,
        tol=1e-5,
        solver_options=["MAX_ITER=10"],
    )

    assert model.m is fake_model
    assert model.mSuccess == 0
    assert model.solver_run.name == "fake"
    assert calls == [
        ("create", False),
        ("configure", fake_model, "apopt", ["MAX_ITER=10"]),
    ]


def test_base_model_scalar_helpers_and_value_assignment():
    model = _model()

    assert model._solver_value([3.5]) == pytest.approx(3.5)
    assert model._solver_value(4.5) == pytest.approx(4.5)

    variable = GKVariable(lower=0.0, upper=10.0)
    model._set_value(variable, 99.0)
    assert variable.VALUE.value == pytest.approx(10.0)
    model._set_value(variable, -99.0, brackets=True)
    assert variable.VALUE.value == [0.0]

    parameter = GKParameter()
    model._set_value(parameter, 7.0, brackets=True)
    assert parameter.VALUE.value == [7.0]

    assert model._post_process_lmtd(
        20.0,
        10.0,
        0.5,
        formula_allowed=False,
    ) == pytest.approx(10.0)
    assert model._post_process_lmtd(
        20.0,
        10.0,
        0.5,
        formula_allowed=False,
        fallback_delta=8.0,
    ) == pytest.approx(4.0)
    assert (
        model._post_process_lmtd(
            20.0,
            10.0,
            0.5,
            formula_allowed=True,
        )
        > 0.0
    )


def test_get_alpha_values_uses_cached_values_and_swallows_solver_failure(monkeypatch):
    model = _model()
    model.alpha = [1.0]
    assert model.get_alpha_values() == [1.0]

    failing_model = _FakeModel(fail_solve=True)
    model.alpha = []
    calls = []

    monkeypatch.setattr(
        base_alpha.backend, "create_gekko_model", lambda remote=False: failing_model
    )
    monkeypatch.setattr(
        base_alpha.backend,
        "suppress_gekko_numpy_array_copy_deprecation",
        nullcontext,
    )
    monkeypatch.setattr(
        model,
        "set_alpha_dqda_equations",
        lambda **kwargs: calls.append(kwargs),
    )

    assert model.get_alpha_values() == []
    assert calls == [{"m": failing_model, "postoptimisation": True}]


@pytest.mark.parametrize("non_isothermal", [False, True])
def test_set_alpha_dqda_equations_builds_fake_model_equations(non_isothermal: bool):
    model = _model()
    _configure_equation_state(model, non_isothermal=non_isothermal)

    model.set_alpha_dqda_equations(postoptimisation=False)

    assert len(model.alpha) == 1
    assert len(model.gamma_h_eqn) == 3
    assert len(model.gamma_c_eqn) == 3
    assert len(model.alpha_eqn) == 2
    assert model.alpha_dQ_dA_eqn[1] is None
    assert model.m.intermediates
    assert model.m.equations


@pytest.mark.parametrize("non_isothermal", [False, True])
def test_set_alpha_dqda_equations_builds_postoptimisation_equations(
    non_isothermal: bool,
):
    model = _model()
    _configure_postoptimisation_state(model, non_isothermal=non_isothermal)
    fake_model = _FakeModel()

    model.set_alpha_dqda_equations(m=fake_model, postoptimisation=True)

    assert len(model.alpha) == 1
    assert len(model.gamma_h_eqn) == 3
    assert len(model.gamma_c_eqn) == 3
    assert len(model.alpha_eqn) == 3
    assert len(fake_model.variables) == 9
    assert model.P_h[0][0][2] == 0.0
    assert model.P_c[0][0][2] == 0.0
    assert model.beta_h[0][0][1] == 0.0
    assert model.beta_c[0][0][1] == 0.0
    assert model.z_i[0][1] == 0.0
    assert model.z_j[0][1] == 0.0


def test_set_alpha_dqda_equations_requires_model_for_postoptimisation():
    model = _model()

    with pytest.raises(ValueError, match="require a model"):
        model.set_alpha_dqda_equations(postoptimisation=True)


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
    assert model.dT_hu.tolist() == [8.0]
    assert model.dT_cu.tolist() == [7.0]
    assert model._recovery_approach_temperature(0, 0, period_idx=1) == pytest.approx(
        12.0
    )
    assert model._hot_utility_inlet_approach_temperature(
        0, period_idx=1
    ) == pytest.approx(10.0)
    assert model._hot_utility_outlet_approach_temperature(
        0, period_idx=1, heat_duty=1.0
    ) == pytest.approx(10.0)
    assert model._cold_utility_inlet_approach_temperature(
        0, period_idx=1
    ) == pytest.approx(9.0)
    assert model._cold_utility_outlet_approach_temperature(
        0, period_idx=1, heat_duty=1.0
    ) == pytest.approx(9.0)
    assert model._weighted_state_average([10.0, 30.0]) == pytest.approx(25.0)

    fallback = _model()
    assert fallback._recovery_approach_temperature(0, 0) == pytest.approx(10.0)
    fallback.dT_r = np.array([[11.0]])
    assert fallback._recovery_approach_temperature(0, 0) == pytest.approx(11.0)


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
        model._normalise_state_arrays()


def test_minimum_approach_temperatures_use_dTmin_when_utilities_are_absent():
    model = _model()
    model.N_periods = 1
    model.T_h_cont_period = np.array([[4.0]])
    model.T_c_cont_period = np.array([[6.0]])
    model.T_hu_cont_period = np.array([[]], dtype=object)
    model.T_cu_cont_period = np.array([[]], dtype=object)

    model._set_minimum_approach_temperatures()

    assert model.dT_hu.tolist() == [11.0]
    assert model.dT_cu.tolist() == [9.0]


def test_match_restrictions_accept_supported_shapes_and_reject_invalid_shape():
    model = _model()
    model.I = 1
    model.J = 1
    model.S = 1
    model.tol = 0.1
    model.z_feasible = [[[1]]]
    model.z_hu_feasible = [1]
    model.z_cu_feasible = [1]

    model.set_match_restrictions(None)
    assert model.z_allowed == [[[1]]]
    assert model.z_hu_allowed == [1]
    assert model.z_cu_allowed == [1]

    model.set_match_restrictions([[[[0]]], [0], [1]])
    assert model.z_allowed == [[[0]]]
    assert model.z_hu_allowed == [0]
    assert model.z_cu_allowed == [1]

    model.set_match_restrictions([[[[[1.0]]]], [[1.0]], [[0.0]]])
    assert model.z_allowed == [[[1]]]
    assert model.z_hu_allowed == [1]
    assert model.z_cu_allowed == [0]

    model.set_match_restrictions([[[[GKVariable()]]], [GKParameter()], [GKParameter()]])
    assert model.z_allowed == [[[0]]]

    with pytest.raises(ValueError, match="Invalid restriction type"):
        model.set_match_restrictions([[[[object()]]], None, None])


def _solver_run(*, failure_reason=None):
    return backend.SolverRun(
        name="fake",
        status=1,
        objective_value=1.0,
        solve_time=0.25,
        failure_reason=failure_reason,
    )


def test_optimise_handles_failure_negative_success_and_unsolved_paths(monkeypatch):
    model = _model()
    model.m = _FakeModel(solve_status=1, objective=1.0)

    monkeypatch.setattr(
        base_execution.backend,
        "solve_gekko_model",
        lambda *args, **kwargs: _solver_run(failure_reason="missing solver"),
    )
    model.optimise(print_output=False)
    assert model.mSuccess == 0
    assert model.post_processed is False

    monkeypatch.setattr(
        base_execution.backend,
        "solve_gekko_model",
        lambda *args, **kwargs: _solver_run(),
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


def test_output_to_cmd_line_logs_successful_solve(caplog):
    model = _model()
    model.mSuccess = 0
    model.output_to_cmd_line()

    model.mSuccess = 1
    model.m = _FakeModel(objective=12.0)
    _set_success_diagnostics(model)

    with caplog.at_level("INFO", logger=base_execution.logger.name):
        model.output_to_cmd_line()

    assert "Successful Solve.Path" in caplog.text
    assert "Q_hu total" in caplog.text
