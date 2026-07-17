"""Focused StageWise helper tests backed by static edge-case fixtures."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

import OpenPinch.analysis.heat_exchanger_networks.models.stagewise as stagewise
from OpenPinch.analysis.heat_exchanger_networks.models._stagewise import (
    evolution as stagewise_evolution,
)
from OpenPinch.analysis.heat_exchanger_networks.models._stagewise import (
    verification as stagewise_verification,
)
from OpenPinch.analysis.heat_exchanger_networks.models.stagewise import (
    StageWiseModel,
)
from tests.support.paths import FIXTURES_ROOT

FIXTURE_PATH = FIXTURES_ROOT / "stagewise_helper_cases.json"


class _ObjectiveModel:
    def __init__(self):
        self.minimised = []
        self.maximised = []
        self.intermediates = []
        self.equations = []

    def sum(self, values):
        return sum(values)

    def Minimize(self, expression):
        self.minimised.append(expression)

    def Maximize(self, expression):
        self.maximised.append(expression)

    def Intermediate(self, expression, *, name: str | None = None):
        self.intermediates.append((name, expression))
        return expression

    def Equation(self, expression):
        self.equations.append(expression)
        return expression


class _VerifiedCandidate:
    def __init__(self, *, m_success: int, valid: bool = True, has_verify: bool = True):
        self.mSuccess = m_success
        self.name = "candidate"
        self._valid = valid
        if has_verify:
            self.verify = lambda: (self._valid, [] if self._valid else ["invalid"])


class _FloatOnly:
    def __init__(self, value: float):
        self.value = value

    def __float__(self):
        return self.value


def test_stagewise_setup_preserves_solver_construction_order():
    model = StageWiseModel.__new__(StageWiseModel)
    model.framework = "TDM"
    calls = []
    ordered_steps = (
        "set_blank_input_parameters",
        "get_model_parameters_from_solver_arrays",
        "set_preprocessing",
        "set_match_restrictions",
        "set_stage_wise_superstructure",
        "set_dqda_equations",
        "set_obj",
    )
    for method_name in ordered_steps:
        if method_name == "set_match_restrictions":
            setattr(
                model,
                method_name,
                lambda restrictions, name=method_name: calls.append(
                    (name, restrictions)
                ),
            )
        else:
            setattr(
                model,
                method_name,
                lambda name=method_name: calls.append(name),
            )
    model.z_restriction = ["allowed"]

    model.setup()

    assert calls == [
        "set_blank_input_parameters",
        "get_model_parameters_from_solver_arrays",
        "set_preprocessing",
        ("set_match_restrictions", ["allowed"]),
        "set_stage_wise_superstructure",
        "set_dqda_equations",
        "set_obj",
    ]


class _IndexedValue:
    def __init__(self, value: float):
        self.value = value

    def __getitem__(self, index):
        if index == 0:
            return self.value
        raise IndexError(index)


class _SourceVariable:
    def __init__(self, value: float):
        self.VALUE = _IndexedValue(value)


class GKVariable:
    def __init__(self):
        self.lower = -1e9
        self.upper = 1e9
        self.VALUE = SimpleNamespace(value=None)

    def __getitem__(self, index):
        if index == 0:
            return self.VALUE.value
        raise IndexError(index)


def _variables(shape: tuple[int, ...]):
    if not shape:
        return GKVariable()
    return [_variables(shape[1:]) for _ in range(shape[0])]


def _source_variables(values):
    if isinstance(values, list):
        return [_source_variables(value) for value in values]
    return _SourceVariable(float(values))


def _case(name: str) -> SimpleNamespace:
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))[name]
    case = SimpleNamespace(**data)
    if hasattr(case, "period_weights"):
        case._weighted_numeric_average = lambda values: float(
            sum(
                float(case.period_weights[n]) * float(values[n])
                for n in range(case.N_periods)
            )
            / case.period_weight_sum
        )
    return case


def _single_state_objective_case(goal: str) -> StageWiseModel:
    model = StageWiseModel.__new__(StageWiseModel)
    model.I = 1
    model.J = 1
    model.S = 1
    model.N_periods = 1
    model.minimisation_goal = goal
    model.m = _ObjectiveModel()
    model.Q_h = [10.0]
    model.Q_c = [5.0]
    model.Q_r = [[[7.0]]]
    model.hu_cost = [2.0]
    model.cu_cost = [3.0]
    model.HU_target = 2.0
    model.hu_unit_cost = [2.0]
    model.cu_unit_cost = [3.0]
    model.unit_cost = [4.0]
    model.z_hu = [1]
    model.z_cu = [1]
    model.z = [[[1]]]
    model.z_allowed = [[[1]]]
    model.A_coeff = [1.0]
    model.A_exp = [1.0]
    model.U_r = [[1.0]]
    model.theta_1 = [[[10.0]]]
    model.theta_2 = [[[20.0]]]
    model.hu_coeff = [1.0]
    model.hu_exp = [1.0]
    model.U_hu = [1.0]
    model.T_hu_in = [200.0]
    model.T_hu_out = [180.0]
    model.T_c_out = [50.0]
    model.T_c = [[40.0, 60.0]]
    model.cu_coeff = [1.0]
    model.cu_exp = [1.0]
    model.U_cu = [1.0]
    model.T_h = [[100.0, 80.0]]
    model.T_h_out = [60.0]
    model.T_cu_in = [20.0]
    model.T_cu_out = [30.0]
    model.hu_cost_total = 10.0
    model.cu_cost_total = 20.0
    model.recovery_area_cost_filtered = [[30.0]]
    model.hu_area_cost_total = 40.0
    model.cu_area_cost_total = 50.0
    return model


def _multi_state_objective_case(goal: str) -> StageWiseModel:
    model = StageWiseModel.__new__(StageWiseModel)
    model.I = 1
    model.J = 1
    model.S = 1
    model.N_periods = 2
    model.period_weights = [1.0, 3.0]
    model.period_weight_sum = 4.0
    model.minimisation_goal = goal
    model.m = _ObjectiveModel()
    model.Q_h_by_period = [[10.0], [20.0]]
    model.Q_c_by_period = [[5.0], [15.0]]
    model.Q_r_by_period = [[[[7.0]]], [[[9.0]]]]
    model.hu_cost_period = [[2.0], [4.0]]
    model.cu_cost_period = [[3.0], [6.0]]
    model.HU_target = 2.0
    return model


def _single_period_warm_start_model(*, stages: int = 2) -> StageWiseModel:
    model = StageWiseModel.__new__(StageWiseModel)
    model.I = 1
    model.J = 1
    model.S = stages
    model.K = stages + 1
    model.dTmin = 7.0
    model.non_isothermal_model = True
    model.z_allowed = [[[1 if index < stages - 1 else 0 for index in range(stages)]]]
    model.z_cu_allowed = [0]
    model.z_hu_allowed = [0]
    model.Q_r = _variables((1, 1, stages))
    model.z = _variables((1, 1, stages))
    model.theta_1 = _variables((1, 1, stages))
    model.theta_2 = _variables((1, 1, stages))
    model.T_h = _variables((1, stages + 1))
    model.T_c = _variables((1, stages + 1))
    model.Q_c = _variables((1,))
    model.z_cu = _variables((1,))
    model.Q_h = _variables((1,))
    model.z_hu = _variables((1,))
    model.X = _variables((1, 1, stages))
    model.Y = _variables((1, 1, stages))
    model.T_h_out_x = _variables((1, 1, stages))
    model.T_c_out_y = _variables((1, 1, stages))
    return model


def _single_period_source(*, non_isothermal: bool, q_values: list[float]):
    stages = len(q_values)
    return SimpleNamespace(
        non_isothermal_model=non_isothermal,
        Q_r=_source_variables([[q_values]]),
        z=[[[[1] for _ in range(stages)]]],
        theta_1=_source_variables([[[10.0 + index for index in range(stages)]]]),
        theta_2=_source_variables([[[20.0 + index for index in range(stages)]]]),
        T_h=_source_variables([[100.0 - 5.0 * index for index in range(stages + 1)]]),
        T_c=_source_variables([[40.0 + 5.0 * index for index in range(stages + 1)]]),
        Q_c=_source_variables([5.0]),
        z_cu=[[1]],
        Q_h=_source_variables([6.0]),
        z_hu=[[1]],
        X=_source_variables([[[0.25 + 0.1 * index for index in range(stages)]]]),
        Y=_source_variables([[[0.35 + 0.1 * index for index in range(stages)]]]),
        T_h_out_x=_source_variables(
            [[[80.0 - 5.0 * index for index in range(stages)]]]
        ),
        T_c_out_y=_source_variables(
            [[[60.0 + 5.0 * index for index in range(stages)]]]
        ),
    )


def _multi_period_warm_start_model() -> StageWiseModel:
    model = _single_period_warm_start_model(stages=2)
    model.N_periods = 2
    model.Q_r_by_period = _variables((2, 1, 1, 2))
    model.theta_1_by_period = _variables((2, 1, 1, 2))
    model.theta_2_by_period = _variables((2, 1, 1, 2))
    model.T_h_by_period = _variables((2, 1, 3))
    model.T_c_by_period = _variables((2, 1, 3))
    model.Q_c_by_period = _variables((2, 1))
    model.Q_h_by_period = _variables((2, 1))
    model.area_r_shared = _variables((1, 1, 2))
    model.area_hu_shared = _variables((1,))
    model.area_cu_shared = _variables((1,))
    model.X_by_period = _variables((2, 1, 1, 2))
    model.Y_by_period = _variables((2, 1, 1, 2))
    model.T_h_out_x_by_period = _variables((2, 1, 1, 2))
    model.T_c_out_y_by_period = _variables((2, 1, 1, 2))
    return model


def _asymmetric_warm_start_model(*, multiperiod: bool) -> StageWiseModel:
    model = StageWiseModel.__new__(StageWiseModel)
    model.I = 2
    model.J = 2
    model.S = 1
    model.K = 2
    model.dTmin = 7.0
    model.non_isothermal_model = True
    model.z_allowed = [[[1], [1]], [[1], [1]]]
    model.z_cu_allowed = [0, 0]
    model.z_hu_allowed = [0, 0]
    model.z = _variables((2, 2, 1))
    model.z_cu = _variables((2,))
    model.z_hu = _variables((2,))
    if not multiperiod:
        model.Q_r = _variables((2, 2, 1))
        model.theta_1 = _variables((2, 2, 1))
        model.theta_2 = _variables((2, 2, 1))
        model.T_h = _variables((2, 2))
        model.T_c = _variables((2, 2))
        model.Q_c = _variables((2,))
        model.Q_h = _variables((2,))
        model.X = _variables((2, 2, 1))
        model.Y = _variables((2, 2, 1))
        model.T_h_out_x = _variables((2, 2, 1))
        model.T_c_out_y = _variables((2, 2, 1))
        return model

    model.N_periods = 2
    model.Q_r_by_period = _variables((2, 2, 2, 1))
    model.theta_1_by_period = _variables((2, 2, 2, 1))
    model.theta_2_by_period = _variables((2, 2, 2, 1))
    model.T_h_by_period = _variables((2, 2, 2))
    model.T_c_by_period = _variables((2, 2, 2))
    model.Q_c_by_period = _variables((2, 2))
    model.Q_h_by_period = _variables((2, 2))
    model.X_by_period = _variables((2, 2, 2, 1))
    model.Y_by_period = _variables((2, 2, 2, 1))
    model.T_h_out_x_by_period = _variables((2, 2, 2, 1))
    model.T_c_out_y_by_period = _variables((2, 2, 2, 1))
    return model


def _asymmetric_isothermal_source() -> SimpleNamespace:
    duties = [[[10.0], [30.0]], [[20.0], [40.0]]]
    return SimpleNamespace(
        non_isothermal_model=False,
        Q_r=_source_variables(duties),
        z=[[[[1]], [[1]]], [[[1]], [[1]]]],
        theta_1=_source_variables([[[20.0], [20.0]], [[20.0], [20.0]]]),
        theta_2=_source_variables([[[15.0], [15.0]], [[15.0], [15.0]]]),
        T_h=_source_variables([[200.0, 150.0], [190.0, 140.0]]),
        T_c=_source_variables([[50.0, 100.0], [60.0, 110.0]]),
        Q_c=_source_variables([0.0, 0.0]),
        z_cu=[[0], [0]],
        Q_h=_source_variables([0.0, 0.0]),
        z_hu=[[0], [0]],
    )


def _source_area_case(*, model_area: float | None = None) -> SimpleNamespace:
    expected = 2.0 * (10.0 / ((10.0 * 20.0 * (10.0 + 20.0) / 2.0 + 1e-3) ** (1 / 3)))
    return SimpleNamespace(
        I=1,
        J=1,
        S=1,
        z_allowed=[[[1]]],
        Q_r=[[[[10.0]]]],
        U_r=[[1.0]],
        theta_1=[[[[10.0]]]],
        theta_2=[[[[20.0]]]],
        A_exp=[1.0],
        A_coeff=[2.0],
        recovery_area_cost_filtered=[[expected if model_area is None else model_area]],
    )


def _shared_area_case() -> SimpleNamespace:
    case = SimpleNamespace(
        I=1,
        J=1,
        S=1,
        N_periods=1,
        Q_r_by_period=[[[[10.0]]]],
        Q_h_by_period=[[10.0]],
        Q_c_by_period=[[10.0]],
        U_r_period=[[[1.0]]],
        U_hu_period=[[1.0]],
        U_cu_period=[[1.0]],
        theta_1_by_period=[[[[10.0]]]],
        theta_2_by_period=[[[[20.0]]]],
        T_hu_in_period=[[200.0]],
        T_hu_out_period=[[180.0]],
        T_c_out_period=[[50.0]],
        T_c_by_period=[[[40.0, 60.0]]],
        T_h_by_period=[[[100.0, 80.0]]],
        T_h_out_period=[[60.0]],
        T_cu_in_period=[[20.0]],
        T_cu_out_period=[[30.0]],
        area_r_shared=[[[100.0]]],
        area_hu_shared=[100.0],
        area_cu_shared=[100.0],
        A_coeff=[2.0],
        A_exp=[1.0],
        hu_coeff=[3.0],
        hu_exp=[1.0],
        cu_coeff=[4.0],
        cu_exp=[1.0],
        recovery_area_cost_total=200.0,
        hu_area_cost_total=300.0,
        cu_area_cost_total=400.0,
    )
    case._utility_solved_outlet_temperature = (
        lambda side, period_idx, match_index, heat_duty: (
            case.T_hu_out_period[period_idx][0]
            if side == "hot"
            else case.T_cu_out_period[period_idx][0]
        )
    )
    return case


@pytest.mark.parametrize(
    ("case_name", "expected"),
    [
        ("source_temperature_valid", True),
        ("source_temperature_crossed", False),
        ("source_non_isothermal_valid", True),
        ("source_non_isothermal_inconsistent", False),
        ("state_temperature_valid", True),
        ("state_temperature_crossed", False),
        ("state_non_isothermal_valid", True),
    ],
)
def test_stagewise_temperature_verification_from_static_cases(case_name, expected):
    assert stagewise_verification._check_temperatures(_case(case_name)) is expected


def test_stagewise_utility_cost_verification_handles_source_and_period_cases():
    assert (
        stagewise_verification._check_utility_costs(_case("utility_cost_source"))
        is True
    )
    assert (
        stagewise_verification._check_utility_costs(_case("utility_cost_state")) is True
    )

    bad_source = _case("utility_cost_source")
    bad_source.hu_cost_total = 999.0
    assert stagewise_verification._check_utility_costs(bad_source) is False

    bad_state = _case("utility_cost_state")
    bad_state.cu_cost_total = 999.0
    assert stagewise_verification._check_utility_costs(bad_state) is False


def test_stagewise_area_cost_verification_handles_source_and_shared_cases():
    assert stagewise_verification._check_area_costs(_source_area_case()) is True
    assert (
        stagewise_verification._check_area_costs(_source_area_case(model_area=0.0))
        is False
    )

    assert stagewise_verification._check_area_costs(_shared_area_case()) is True

    failed_recovery = _shared_area_case()
    failed_recovery.area_r_shared = [[[0.0]]]
    assert stagewise_verification._check_area_costs(failed_recovery) is False

    failed_hot_utility = _shared_area_case()
    failed_hot_utility.area_hu_shared = [0.0]
    assert stagewise_verification._check_area_costs(failed_hot_utility) is False

    failed_cold_utility = _shared_area_case()
    failed_cold_utility.area_cu_shared = [0.0]
    assert stagewise_verification._check_area_costs(failed_cold_utility) is False

    failed_total = _shared_area_case()
    failed_total.recovery_area_cost_total = 999.0
    assert stagewise_verification._check_area_costs(failed_total) is False


@pytest.mark.parametrize(
    ("goal", "expected", "method"),
    [
        ("cold utility", 12.5, "minimised"),
        ("utility costs", 136.25, "minimised"),
        ("heat recovery", 8.5, "maximised"),
        ("dQ/dA obj", 15.5, "minimised"),
    ],
)
def test_stagewise_multiperiod_objective_selection(goal, expected, method):
    model = _multi_state_objective_case(goal)

    model.set_obj()

    recorded = getattr(model.m, method)
    assert recorded == [pytest.approx(expected)]


@pytest.mark.parametrize(
    ("goal", "expected", "method"),
    [
        ("cold utility", 5.0, "minimised"),
        ("utility costs", 35.0, "minimised"),
        ("heat recovery", 7.0, "maximised"),
        ("dQ/dA obj", 8.0, "minimised"),
    ],
)
def test_stagewise_single_state_objective_selection(goal, expected, method):
    model = _single_state_objective_case(goal)

    model.set_obj()

    recorded = getattr(model.m, method)
    assert recorded == [pytest.approx(expected)]


def test_stagewise_single_state_total_cost_objective_records_minimisation():
    model = _single_state_objective_case("total cost")

    model.set_obj()

    assert len(model.m.minimised) == 1
    assert model.utility_unit_cost_total == pytest.approx(5.0)
    assert model.recovery_unit_cost[0][0] == pytest.approx(4.0)


def test_stagewise_single_period_dqda_equations_use_recovery_stage_grid():
    model = StageWiseModel.__new__(StageWiseModel)
    model.I = 1
    model.J = 1
    model.S = 2
    model.N_periods = 1
    model.m = _ObjectiveModel()
    model.min_dqda = 0.1
    model.T_h = [[100.0, 90.0, 80.0]]
    model.T_c = [[30.0, 40.0, 50.0]]
    model.theta_1 = [[[2.0, 3.0]]]
    model.theta_2 = [[[4.0, 5.0]]]
    model.U_r = [[0.5]]
    model.z_allowed = [[[1, 0]]]
    model.M_ij = [[10.0]]
    model.z = [[[1.0, 0.0]]]

    model.set_dqda_equations()

    assert model.min_dqda_int[0][0][0] == pytest.approx(2.0)
    assert model.min_dqda_int[0][0][1] is None
    assert model.min_dQ_dA_eqn[1] is None
    assert model.m.intermediates[0][0] == "dqda_calc_H0_to_C0_at_S0"


def test_stagewise_multiperiod_dqda_equations_use_period_recovery_stage_grid():
    model = StageWiseModel.__new__(StageWiseModel)
    model.I = 1
    model.J = 1
    model.S = 2
    model.N_periods = 2
    model.m = _ObjectiveModel()
    model.min_dqda = 0.1
    model.T_h_by_period = [[[100.0, 90.0, 80.0]], [[120.0, 100.0, 80.0]]]
    model.T_c_by_period = [[[30.0, 40.0, 50.0]], [[35.0, 45.0, 55.0]]]
    model.theta_1_by_period = [[[[2.0, 3.0]]], [[[4.0, 5.0]]]]
    model.theta_2_by_period = [[[[4.0, 5.0]]], [[[6.0, 7.0]]]]
    model.U_r_period = [[[0.5]], [[0.25]]]
    model.z_allowed = [[[1, 0]]]
    model.M_ij_period = [[[10.0]], [[20.0]]]
    model.z = [[[1.0, 0.0]]]

    model.set_dqda_equations()

    assert model.min_dqda_int[0][0][0][0] == pytest.approx(2.0)
    assert model.min_dqda_int[0][0][0][1] is None
    assert model.min_dqda_int[1][0][0][0] == pytest.approx(1.5)
    assert model.min_dqda_int[1][0][0][1] is None
    assert model.min_dQ_dA_eqn[1] is None
    assert model.min_dQ_dA_eqn[3] is None
    assert [name for name, _ in model.m.intermediates] == [
        "dqda_calc_H0_to_C0_at_S0_period0",
        "dqda_calc_H0_to_C0_at_S0_period1",
    ]


def test_stagewise_single_period_initial_values_cover_inactive_utilities_and_noniso():
    model = _single_period_warm_start_model(stages=2)
    source = _single_period_source(non_isothermal=True, q_values=[10.0, 0.0])

    model.set_initial_values_for_variables(source)

    assert model.Q_r[0][0][0].VALUE.value == pytest.approx(10.0)
    assert model.Q_r[0][0][1].VALUE.value == pytest.approx(0.0)
    assert model.theta_1[0][0][1].VALUE.value == pytest.approx(model.dTmin)
    assert model.Q_c[0].VALUE.value == pytest.approx(0.0)
    assert model.z_cu[0].VALUE.value == 0
    assert model.Q_h[0].VALUE.value == pytest.approx(0.0)
    assert model.z_hu[0].VALUE.value == 0
    assert model.X[0][0][0].VALUE.value == pytest.approx(0.25)
    assert model.Y[0][0][0].VALUE.value == pytest.approx(0.35)
    assert model.X[0][0][1].VALUE.value == pytest.approx(0.0)
    assert model.Y[0][0][1].VALUE.value == pytest.approx(0.0)
    assert model.T_h_out_x[0][0][1].VALUE.value == pytest.approx(90.0)
    assert model.T_c_out_y[0][0][1].VALUE.value == pytest.approx(45.0)


def test_stagewise_single_period_initial_values_derive_noniso_split_from_iso_source():
    model = _single_period_warm_start_model(stages=3)
    source = _single_period_source(non_isothermal=False, q_values=[10.0, 0.0, 0.0])

    model.set_initial_values_for_variables(source)

    assert model.X[0][0][0].VALUE.value == pytest.approx(1.0)
    assert model.Y[0][0][0].VALUE.value == pytest.approx(1.0)
    assert model.X[0][0][1].VALUE.value == pytest.approx(0.0)
    assert model.Y[0][0][1].VALUE.value == pytest.approx(0.0)
    assert model.X[0][0][2].VALUE.value == pytest.approx(0.0)
    assert model.Y[0][0][2].VALUE.value == pytest.approx(0.0)
    assert model.T_h_out_x[0][0][2].VALUE.value == pytest.approx(85.0)
    assert model.T_c_out_y[0][0][2].VALUE.value == pytest.approx(50.0)


def test_stagewise_multiperiod_initial_values_use_source_fallback_and_shared_area():
    model = _multi_period_warm_start_model()
    source = _single_period_source(non_isothermal=True, q_values=[10.0, 0.0])
    source.area_r_shared = [[[11.0, 12.0]]]
    source.area_hu_shared = [13.0]
    source.area_cu_shared = [14.0]

    model.set_initial_values_for_variables(source)

    assert model.Q_r_by_period[0][0][0][0].VALUE.value == pytest.approx(10.0)
    assert model.Q_r_by_period[1][0][0][0].VALUE.value == pytest.approx(10.0)
    assert model.Q_r_by_period[0][0][0][1].VALUE.value == pytest.approx(0.0)
    assert model.theta_1_by_period[0][0][0][1].VALUE.value == pytest.approx(model.dTmin)
    assert model.Q_c_by_period[0][0].VALUE.value == pytest.approx(0.0)
    assert model.Q_h_by_period[0][0].VALUE.value == pytest.approx(0.0)
    assert model.z[0][0][0].VALUE.value == 1
    assert model.z_cu[0].VALUE.value == 1
    assert model.z_hu[0].VALUE.value == 1
    assert model.area_r_shared[0][0][0].VALUE.value == pytest.approx(11.0)
    assert model.area_hu_shared[0].VALUE.value == pytest.approx(13.0)
    assert model.area_cu_shared[0].VALUE.value == pytest.approx(14.0)
    assert model.X_by_period[1][0][0][0].VALUE.value == pytest.approx(0.25)
    assert model.Y_by_period[1][0][0][0].VALUE.value == pytest.approx(0.35)
    assert model.T_h_out_x_by_period[1][0][0][0].VALUE.value == pytest.approx(80.0)
    assert model.T_c_out_y_by_period[1][0][0][0].VALUE.value == pytest.approx(60.0)


@pytest.mark.parametrize("multiperiod", [False, True])
def test_warm_start_split_normalization_uses_fixed_stream_totals(multiperiod: bool):
    model = _asymmetric_warm_start_model(multiperiod=multiperiod)
    model.set_initial_values_for_variables(_asymmetric_isothermal_source())

    x = model.X_by_period[1] if multiperiod else model.X
    y = model.Y_by_period[1] if multiperiod else model.Y
    assert x[0][0][0].VALUE.value == pytest.approx(10.0 / 40.0)
    assert x[0][1][0].VALUE.value == pytest.approx(30.0 / 40.0)
    assert x[1][0][0].VALUE.value == pytest.approx(20.0 / 60.0)
    assert x[1][1][0].VALUE.value == pytest.approx(40.0 / 60.0)
    assert y[0][0][0].VALUE.value == pytest.approx(10.0 / 30.0)
    assert y[0][1][0].VALUE.value == pytest.approx(20.0 / 30.0)
    assert y[1][0][0].VALUE.value == pytest.approx(30.0 / 70.0)
    assert y[1][1][0].VALUE.value == pytest.approx(40.0 / 70.0)


def test_stagewise_candidate_helpers_cover_invalid_and_verified_paths():
    assert stagewise_evolution._count_allowed_matches([[[1, [1], 0, [0]]]]) == 2
    assert stagewise_evolution._is_usable_evolution_candidate(None) is False
    assert (
        stagewise_evolution._is_usable_evolution_candidate(
            _VerifiedCandidate(m_success=0)
        )
        is False
    )
    assert (
        stagewise_evolution._is_usable_evolution_candidate(
            _VerifiedCandidate(m_success=1, has_verify=False)
        )
        is True
    )
    assert (
        stagewise_evolution._is_usable_evolution_candidate(
            _VerifiedCandidate(m_success=1, valid=False)
        )
        is False
    )
    assert (
        stagewise_evolution._is_usable_evolution_candidate(
            _VerifiedCandidate(m_success=1, valid=True)
        )
        is True
    )


def test_stagewise_value_coercion_accepts_solver_like_shapes():
    assert stagewise_verification._value(1) == 1.0
    assert stagewise_verification._value([2.5]) == 2.5
    assert stagewise_verification._value(SimpleNamespace(VALUE=[3.5])) == 3.5
    assert stagewise_verification._value(SimpleNamespace(value=[4.5])) == 4.5
    assert (
        stagewise_verification._value(SimpleNamespace(VALUE=SimpleNamespace(value=5.5)))
        == 5.5
    )
    assert (
        stagewise_verification._value(
            SimpleNamespace(VALUE=SimpleNamespace(value=_FloatOnly(6.5)))
        )
        == 6.5
    )
    assert math.isclose(stagewise_verification._value(_FloatOnly(7.5)), 7.5)


def _evolution_model(*, tac: float = 100.0, success: int = 1) -> StageWiseModel:
    model = StageWiseModel.__new__(StageWiseModel)
    model.name = "root"
    model.TAC = tac
    model.mSuccess = success
    model.I = 1
    model.J = 1
    model.S = 1
    model.m = SimpleNamespace(cleanup=lambda: setattr(model, "_cleaned", True))
    return model


def test_stagewise_evolution_skips_unsuccessful_and_empty_frontiers():
    failed = _evolution_model(success=0)
    assert failed.get_net_benefit_evolution(print_output=False) is failed

    empty = _evolution_model()
    empty._evolution_candidate_specs = lambda *args, **kwargs: []

    assert (
        empty.get_net_benefit_evolution(
            print_output=False,
            n_ad_branches=2,
        )
        is empty
    )
    assert empty._cleaned is True


def test_stagewise_evolution_prunes_unusable_and_stale_candidates():
    model = _evolution_model(tac=100.0)
    spec = stagewise._EvolutionCandidateSpec(
        kind="plus",
        unit=0,
        branch_index=0,
        rank=1,
        prev_case=model,
        position=(0, 0, 0),
        z_allowed=[[[1]]],
        signature=((0, 0, 0),),
    )
    unusable = _VerifiedCandidate(m_success=0)
    stale = _VerifiedCandidate(m_success=1)
    stale.TAC = 150.0
    stale.name = "stale"

    model._evolution_candidate_specs = lambda *args, **kwargs: [spec, spec]
    model._solve_evolution_candidates = lambda *args, **kwargs: [
        (spec, unusable),
        (spec, stale),
    ]

    assert (
        model.get_net_benefit_evolution(
            print_output=False,
            n_ad_branches=2,
            no_improvement_patience=1,
        )
        is model
    )
    assert model._cleaned is True


def test_stagewise_source_evolution_and_candidate_selection_edges():
    model = _evolution_model(tac=100.0)
    model.get_n_minus_one_evolution = lambda **kwargs: None
    model.get_n_plus_one_evolution = lambda **kwargs: None

    assert (
        model._get_source_net_benefit_evolution(
            print_output=False,
            max_depth=1,
        )
        is model
    )
    assert model._select_source_best_candidate(model, None, None) is None

    minus = SimpleNamespace(mSuccess=1, TAC=80.0)
    plus = SimpleNamespace(mSuccess=0, TAC=70.0)
    assert model._select_source_best_candidate(model, minus, plus) is minus
    assert model._select_source_best_candidate(model, plus, minus) is minus


def test_stagewise_candidate_spec_duplicate_and_parallel_solve_edges():
    model = _evolution_model()

    assert (
        model._evolution_candidate_spec(
            kind="plus",
            unit=0,
            branch_index=0,
            rank=1,
            prev_case=model,
            position=(0, 0),
            z_value=1,
            seen_signatures=set(),
        )
        is None
    )

    model._z_allowed_with_candidate = lambda *args, **kwargs: [[[1]]]
    model._topology_signature_from_z = lambda z_values: ((0, 0, 0),)
    assert (
        model._evolution_candidate_spec(
            kind="plus",
            unit=0,
            branch_index=0,
            rank=1,
            prev_case=model,
            position=(0, 0, 0),
            z_value=1,
            seen_signatures={((0, 0, 0),)},
        )
        is None
    )

    first = stagewise._EvolutionCandidateSpec(
        kind="plus",
        unit=0,
        branch_index=0,
        rank=1,
        prev_case=model,
        position=(0, 0, 0),
        z_allowed=[[[1]]],
        signature=((0, 0, 0),),
    )
    second = stagewise._EvolutionCandidateSpec(
        kind="minus",
        unit=0,
        branch_index=0,
        rank=2,
        prev_case=model,
        position=(0, 0, 0),
        z_allowed=[[[0]]],
        signature=(),
    )

    def solve_candidate(spec, print_output):
        if spec.kind == "plus":
            raise RuntimeError("failed branch")
        return SimpleNamespace(mSuccess=1, TAC=90.0)

    model._solve_evolution_candidate = solve_candidate
    solved = model._solve_evolution_candidates(
        [first, second],
        print_output=False,
        max_parallel=2,
    )

    assert solved == [(second, solved[0][1])]


def test_stagewise_select_best_candidate_update_and_binary_value_edges(monkeypatch):
    model = _evolution_model()
    plus = _VerifiedCandidate(m_success=1)
    plus.TAC = 90.0
    minus = _VerifiedCandidate(m_success=0)

    assert model._select_best_candidate(model, minus, plus) is plus

    target = StageWiseModel.__new__(StageWiseModel)
    target.I = 1
    target.J = 1
    target.S = 1
    calls = []
    target.set_initial_values_for_variables = lambda best, brackets: calls.append(
        ("initial", brackets)
    )
    target.get_post_process = lambda: calls.append(("post", None))
    best = SimpleNamespace(
        verify=lambda: (True, []),
        alpha=[[[[0.1]]]],
        z_allowed=[[[1]]],
        hu_cost_total=1.0,
        cu_cost_total=2.0,
        recovery_area_cost_filtered=3.0,
        recovery_area_cost_total=4.0,
        capital_cost_total=5.0,
        weighted_operating_cost=6.0,
        area_r_shared=7.0,
        area_hu_shared=8.0,
        area_cu_shared=9.0,
        hu_area_cost_total=10.0,
        cu_area_cost_total=11.0,
    )

    target._update_with_best_model(best)

    assert target.alpha == [[[[0.1]]]]
    assert target.z_allowed == [[[1]]]
    assert calls == [("initial", True), ("post", None)]

    z_values = [[[[0]]]]
    target._set_recovery_binary_value(z_values, (0, 0, 0), 1)
    assert z_values[0][0][0] == [1]

    class ValueHolder:
        def __init__(self):
            self.VALUE = SimpleNamespace(value=0)

    z_values = [[[ValueHolder()]]]
    target._set_recovery_binary_value(z_values, (0, 0, 0), 1)
    assert z_values[0][0][0].VALUE.value == 1

    class NoMutation:
        def __setitem__(self, key, value):
            raise TypeError("immutable")

    z_values = [[[NoMutation()]]]
    target._set_recovery_binary_value(z_values, (0, 0, 0), 1)
    assert z_values[0][0][0] == 1


def test_stagewise_post_process_and_verify_failure_edges(monkeypatch):
    model = StageWiseModel.__new__(StageWiseModel)
    model.mSuccess = 0
    assert model.get_post_process() is None

    verify_model = StageWiseModel.__new__(StageWiseModel)
    verify_model.minimisation_goal = "total cost"
    monkeypatch.setattr(
        stagewise_verification, "_check_temperatures", lambda case: False
    )
    monkeypatch.setattr(
        stagewise_verification, "_check_utility_costs", lambda case: False
    )
    monkeypatch.setattr(stagewise_verification, "_check_area_costs", lambda case: False)

    assert verify_model.verify() == (False, ["temperature", "cost", "area"])
