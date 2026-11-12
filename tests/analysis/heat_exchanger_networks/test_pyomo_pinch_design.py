"""The pinch-design model on native Pyomo, against its GEKKO conversion.

Couenne and Ipopt solve the PDM as a native Pyomo model. While GEKKO remains
for the stage-wise models, the same equations built through GEKKO's Pyomo
conversion are the reference: both must reach the same solution.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from OpenPinch.analysis.heat_exchanger_networks.models._base import (
    approach as base_approach,
)
from OpenPinch.analysis.heat_exchanger_networks.models._base import (
    execution as base_execution,
)
from OpenPinch.analysis.heat_exchanger_networks.models.pinch_decomposition import (
    PinchDecompModel,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    problem_to_solver_arrays,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
    require_solver_binary,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.pinch_design_decomposition import (
    build_pinch_design_decomposition,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.pyomo_model import (
    PyomoModel,
)
from OpenPinch.application.problem import PinchProblem
from tests.analysis.heat_exchanger_networks.test_multi_utility import (
    _two_cold_utility_problem,
)
from tests.support.paths import REPOSITORY_ROOT

FOUR_STREAM_JSON = (
    REPOSITORY_ROOT
    / "tests/fixtures/openhens/Four-stream-Yee-and-Grossmann-1990-1.json"
)


def _couenne_available() -> bool:
    try:
        require_solver_binary("couenne", purpose="Pyomo PDM parity")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


class _GekkoPinchDecompModel(PinchDecompModel):
    """The PDM built through GEKKO and its Pyomo conversion, as before."""

    def setup_model(self) -> None:
        base_execution.setup_model(self)


def _four_stream_problem() -> PinchProblem:
    payload = json.loads(FOUR_STREAM_JSON.read_text(encoding="utf-8"))
    return PinchProblem(source=payload, project_name="Four-stream")


def _pdm(model_cls, problem, *, dTmin: float, side: str, solver: str = "couenne"):
    model = model_cls(
        name=f"{side} pinch {dTmin}",
        framework="PDM",
        solver=solver,
        solver_arrays=problem_to_solver_arrays(problem, dTmin),
        dTmin=dTmin,
        z_restriction=None,
        min_dqda=0.0,
        minimisation_goal="hot utility" if side == "above" else "cold utility",
        non_isothermal_model=False,
        integers=True,
        tol=1e-3,
        pinch_loc=side,
        pinch_decomposition=build_pinch_design_decomposition(
            problem, dTmin, pinch_location=side
        ),
        stage_selection="automated",
    )
    model.optimise(False)
    return model


@pytest.mark.solver
# A string condition is evaluated when the test runs, so collecting this module
# does not import the solver packages.
@pytest.mark.skipif("not _couenne_available()", reason="needs the couenne binary")
def test_couenne_pdm_is_a_native_pyomo_model_with_one_objective():
    import pyomo.environ as pyo

    model = _pdm(PinchDecompModel, _four_stream_problem(), dTmin=14.0, side="above")

    assert isinstance(model.m, PyomoModel)
    assert len(list(model.m.model.component_data_objects(pyo.Objective))) == 1
    assert model.mSuccess == 1


def _solution(model) -> dict[str, float]:
    return {
        "success": model.mSuccess,
        "objective": model.solver_run.objective_value,
        "recovery": model.Q_r_total,
        **{
            f"hot utility {u}": duty
            for u, duty in enumerate(model.Q_hu_by_utility_by_period[0])
        },
        **{
            f"cold utility {u}": duty
            for u, duty in enumerate(model.Q_cu_by_utility_by_period[0])
        },
    }


@pytest.mark.solver
@pytest.mark.skipif("not _couenne_available()", reason="needs the couenne binary")
@pytest.mark.parametrize(
    "case, dTmin, side",
    [
        ("four-stream", 14.0, "above"),
        ("four-stream", 14.0, "below"),
        ("four-stream", 26.0, "above"),
        ("four-stream", 26.0, "below"),
        # Two cold utilities below the pinch: utility cost picks glycol.
        ("two cold utilities", 4.0, "below"),
    ],
)
def test_pyomo_pdm_matches_the_gekko_pdm(case: str, dTmin: float, side: str):
    pytest.importorskip("gekko")
    problem = (
        _four_stream_problem()
        if case == "four-stream"
        else _two_cold_utility_problem(5.0, glycol_first=False)
    )

    pyomo_model = _pdm(PinchDecompModel, problem, dTmin=dTmin, side=side)
    gekko_model = _pdm(_GekkoPinchDecompModel, problem, dTmin=dTmin, side=side)

    assert isinstance(gekko_model.m, PyomoModel) is False
    assert pyomo_model.mSuccess == 1
    assert _solution(pyomo_model) == pytest.approx(_solution(gekko_model), abs=1e-3)


@pytest.mark.synthesis
def test_relaxed_pyomo_utility_match_with_a_short_fixed_end_carries_no_duty():
    # A relaxed model fixes its utility binaries at 1, as GEKKO parameters
    # before. A heater or cooler whose fixed end misses its approach must then
    # carry no duty, not add a constraint no solution can meet.
    pyo = pytest.importorskip("pyomo.environ")
    model = SimpleNamespace(m=PyomoModel(), tol=1e-6)
    duty = model.m.Var(value=5.0, lb=0.0, ub=10.0)

    base_approach._set_terminal_approach_equations(
        model,
        match=model.m.Param(value=1),
        duty=duty,
        inlet_shortfall=5.0,
        outlet_margin=duty,
        big_m=100.0,
    )

    (constraint,) = model.m.model.component_data_objects(pyo.Constraint)
    assert constraint.equality
    assert constraint.body is duty
    assert pyo.value(constraint.upper) == pytest.approx(0.0)


def _amalgamated_values(model_cls, problem, dTmin: float) -> dict[str, list[float]]:
    from OpenPinch.analysis.heat_exchanger_networks.models._stagewise.verification import (
        _value,
    )

    above = _pdm(model_cls, problem, dTmin=dTmin, side="above")
    below = _pdm(model_cls, problem, dTmin=dTmin, side="below")
    case = above.amalgamate_networks(below_case=below, above_case=above)
    try:
        return {
            "recovery": [
                _value(duty)
                for hot in case.Q_r_by_period[0]
                for cold in hot
                for duty in cold
            ],
            "hot temperatures": [
                _value(t) for row in case.T_h_by_period[0] for t in row
            ],
            "cold temperatures": [
                _value(t) for row in case.T_c_by_period[0] for t in row
            ],
            "heaters": [_value(q) for row in case.Q_h_by_period[0] for q in row],
            "coolers": [_value(q) for row in case.Q_c_by_period[0] for q in row],
        }
    finally:
        for model in (above, below, case):
            model.cleanup()


@pytest.mark.solver
@pytest.mark.skipif("not _couenne_available()", reason="needs the couenne binary")
def test_amalgamated_pyomo_pdm_matches_the_gekko_amalgamation():
    pytest.importorskip("gekko")
    problem = _four_stream_problem()

    pyomo_values = _amalgamated_values(PinchDecompModel, problem, 14.0)
    gekko_values = _amalgamated_values(_GekkoPinchDecompModel, problem, 14.0)

    for name, values in gekko_values.items():
        assert pyomo_values[name] == pytest.approx(values, abs=1e-3), name


@pytest.mark.synthesis
def test_set_value_changes_pyomo_parameters_and_clamps_variables():
    # The segmented-stream solve loop switches its interval selectors, which
    # are parameters, between solves.
    pytest.importorskip("pyomo.environ")
    model = PyomoModel()
    selector = model.Param(value=1.0)
    duty = model.Var(value=1.0, lb=0.0, ub=10.0)

    base_execution._set_value(None, selector, 0.0)
    base_execution._set_value(None, duty, 12.0)

    assert selector.fixed
    assert selector.value == pytest.approx(0.0)
    assert duty.value == pytest.approx(10.0)


def _highs_available() -> bool:
    try:
        from OpenPinch.analysis.heat_exchanger_networks.solver import backend

        backend.require_solver_backend("highs")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


def _built_pdm(problem, *, dTmin: float, side: str, goal: str | None = None):
    return PinchDecompModel(
        name=f"{side} pinch {dTmin}",
        framework="PDM",
        solver="highs",
        solver_arrays=problem_to_solver_arrays(problem, dTmin),
        dTmin=dTmin,
        z_restriction=None,
        min_dqda=0.0,
        minimisation_goal=goal
        or ("hot utility" if side == "above" else "cold utility"),
        non_isothermal_model=False,
        integers=True,
        tol=1e-3,
        pinch_loc=side,
        pinch_decomposition=build_pinch_design_decomposition(
            problem, dTmin, pinch_location=side
        ),
        stage_selection="automated",
    )


@pytest.mark.synthesis
@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
@pytest.mark.parametrize("side", ["above", "below"])
def test_pdm_logic_constraints_are_big_m_so_the_model_is_linear(side: str):
    # Duties are switched by their binaries through Q <= Q_max * z rather than
    # Q * (1 - z) == 0, so a MILP solver can take the model.
    pyo = pytest.importorskip("pyomo.environ")
    from pyomo.core.expr import polynomial_degree

    model = _built_pdm(_four_stream_problem(), dTmin=14.0, side=side)

    degrees = {
        polynomial_degree(constraint.body)
        for constraint in model.m.model.component_data_objects(
            pyo.Constraint, active=True
        )
    }
    assert degrees <= {0, 1}
    assert polynomial_degree(model.m.objective) <= 1


@pytest.mark.synthesis
def test_solver_value_reads_pyomo_variables_and_parameters():
    # The relaxed segment loop reads solved heat coordinates between solves.
    pytest.importorskip("pyomo.environ")
    model = PyomoModel()

    assert base_execution._solver_value(None, model.Var(value=4.0)) == 4.0
    assert base_execution._solver_value(None, model.Param(value=2.5)) == 2.5


@pytest.mark.synthesis
@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_a_nonlinear_highs_pdm_is_solved_with_couenne(monkeypatch):
    # HiGHS takes only linear models; one that is not linear (segmented
    # streams that keep products, for example) goes to Couenne instead.
    from OpenPinch.analysis.heat_exchanger_networks.solver import backend

    configured = []

    def configure(model, solver_name, *, solver_options=None):
        configured.append((solver_name, solver_options))
        return backend.SolverRun(name=solver_name, extension="pyomo")

    monkeypatch.setattr(PyomoModel, "is_linear", lambda self: False)
    monkeypatch.setattr(backend, "configure_pyomo_solver", configure)

    model = _built_pdm(_four_stream_problem(), dTmin=14.0, side="above")

    assert model.solver == "couenne"
    # HiGHS options do not apply to Couenne.
    assert configured == [("highs", None), ("couenne", None)]


@pytest.mark.synthesis
@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_a_linear_highs_pdm_stays_on_highs():
    model = _built_pdm(_four_stream_problem(), dTmin=14.0, side="above")

    assert model.solver == "highs"
    assert model.m.is_linear()
