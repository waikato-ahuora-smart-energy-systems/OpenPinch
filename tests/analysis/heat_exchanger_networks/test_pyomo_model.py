"""Native Pyomo model builder used by the pinch-design models."""

from __future__ import annotations

from pathlib import Path

import pytest

from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.pyomo_model import (
    PyomoModel,
)

pyo = pytest.importorskip("pyomo.environ")

pytestmark = pytest.mark.synthesis


def test_var_has_its_bounds_start_value_and_domain():
    model = PyomoModel()

    duty = model.Var(value=2.0, lb=0.0, ub=10.0, name="Q")
    binary = model.Var(value=1, lb=0, ub=1, integer=True, name="z")

    assert duty.value == pytest.approx(2.0)
    assert (duty.lb, duty.ub) == (0.0, 10.0)
    assert duty.domain is pyo.Reals
    assert binary.domain is pyo.Integers
    assert (binary.lb, binary.ub) == (0, 1)


def test_param_is_a_fixed_value_that_can_be_changed():
    model = PyomoModel()

    selector = model.Param(value=1.0, name="active")
    selector.fix(0.0)

    assert selector.fixed
    assert pyo.value(selector) == pytest.approx(0.0)


def test_equations_become_constraints_of_the_model():
    model = PyomoModel()
    x = model.Var(value=0.0, lb=0.0, ub=5.0)
    y = model.Var(value=0.0, lb=0.0, ub=5.0)

    model.Equation(x + y >= 4)
    model.Equations([x <= 3, y <= 3])

    constraints = list(model.model.component_data_objects(pyo.Constraint))
    assert len(constraints) == 3


def test_model_has_exactly_one_objective():
    model = PyomoModel()
    x = model.Var(value=0.0, lb=0.0, ub=5.0)

    model.Maximize(x)

    objectives = list(model.model.component_data_objects(pyo.Objective))
    assert len(objectives) == 1
    assert objectives[0].sense == pyo.maximize
    # Solvers reading a model with two objectives silently use only the
    # first, so a second one is an error rather than a dropped term.
    with pytest.raises(ValueError, match="one objective"):
        model.Minimize(x)


def test_sum_of_many_terms_builds_without_recursion(caplog):
    model = PyomoModel()
    terms = [model.Var(value=1.0) for _ in range(5000)]

    with caplog.at_level("WARNING", logger="pyomo"):
        total = model.sum(terms)
    assert "deprecated" not in caplog.text
    model.Equation(total >= 0)

    assert pyo.value(total) == pytest.approx(5000.0)


def test_intermediate_is_the_expression_itself():
    model = PyomoModel()
    x = model.Var(value=3.0)

    doubled = model.Intermediate(2 * x)

    assert pyo.value(doubled) == pytest.approx(6.0)


def _solver_available(solver_name: str) -> bool:
    try:
        backend.require_solver_backend(solver_name)
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


def _bounded_model(lower_sum: float) -> tuple[PyomoModel, object, object]:
    model = PyomoModel()
    x = model.Var(value=1.0, lb=0.0, ub=10.0)
    y = model.Var(value=1.0, lb=0.0, ub=10.0)
    model.Equation(x + y >= lower_sum)
    model.Minimize(x + 3 * y)
    return model, x, y


@pytest.mark.solver
# A string condition is evaluated when the test runs, so collecting this module
# does not import the solver packages.
@pytest.mark.skipif(
    "not _solver_available('ipopt-pyomo')", reason="needs the ipopt binary"
)
def test_pyomo_solve_loads_values_and_records_the_run():
    model, x, y = _bounded_model(4.0)

    backend.configure_pyomo_solver(model, "ipopt-pyomo")
    run = backend.solve_pyomo_model(model, solver_name="ipopt-pyomo")

    assert run.failure_reason is None
    assert run.status == 1
    assert run.termination_condition == "optimal"
    assert run.converged
    assert run.objective_value == pytest.approx(4.0, abs=1e-6)
    assert x.value == pytest.approx(4.0, abs=1e-6)
    assert y.value == pytest.approx(0.0, abs=1e-6)


@pytest.mark.solver
@pytest.mark.skipif(
    "not _solver_available('ipopt-pyomo')", reason="needs the ipopt binary"
)
def test_pyomo_solve_reports_an_infeasible_model_as_failed():
    model, _x, _y = _bounded_model(40.0)

    backend.configure_pyomo_solver(model, "ipopt-pyomo")
    run = backend.solve_pyomo_model(model, solver_name="ipopt-pyomo")

    assert run.status == 0
    assert "infeasible" in run.failure_reason


@pytest.mark.solver
@pytest.mark.skipif("not _solver_available('couenne')", reason="needs couenne")
def test_pyomo_solver_options_go_to_an_option_file_removed_on_cleanup():
    model, x, _y = _bounded_model(4.0)

    run = backend.configure_pyomo_solver(
        model, "couenne", solver_options={"max_iter": 50}
    )
    solved = backend.solve_pyomo_model(model, solver_name="couenne")
    option_file = Path(run.option_file)

    assert option_file.name == "couenne.opt"
    assert option_file.read_text(encoding="utf-8") == "max_iter 50\n"
    assert solved.failure_reason is None
    assert solved.option_file == run.option_file
    assert x.value == pytest.approx(4.0, abs=1e-6)
    backend.cleanup_model_run_directory(model)
    assert not option_file.parent.exists()


@pytest.mark.parametrize("termination", ["noSolution", "unbounded"])
def test_pyomo_solve_without_a_solution_is_failed(monkeypatch, termination):
    # A warning that carries no solution must not be read as a solved network:
    # the variables would still hold their start values.
    model, _x, _y = _bounded_model(4.0)
    results = type(
        "Results",
        (),
        {"solver": type("Solver", (), {"status": "warning"})()},
    )()
    results.solver.termination_condition = termination

    class _Solver:
        def solve(self, *_args, **_kwargs):
            return results

    monkeypatch.setattr(pyo, "SolverFactory", lambda _name: _Solver())

    run = backend.solve_pyomo_model(model, solver_name="couenne")

    assert run.status == 0
    assert run.objective_value is None
    assert termination in run.failure_reason


def _highs_available() -> bool:
    return _solver_available("highs")


def _binary_model() -> tuple[PyomoModel, object, object]:
    # Covering a demand of 3 needs the unit switched on.
    model = PyomoModel()
    duty = model.Var(value=0.0, lb=0.0, ub=10.0)
    unit = model.Var(value=0, lb=0, ub=1, integer=True)
    model.Equation(duty >= 3.0)
    model.Equation(duty <= 10.0 * unit)
    model.Minimize(duty + 0.5 * unit)
    return model, duty, unit


def test_highs_needs_highspy_and_no_solver_binary(monkeypatch):
    def missing_binary(binary_name: str, *, purpose: str | None = None) -> str:
        raise MissingSynthesisSolverError(f"{binary_name} is missing")

    monkeypatch.setattr(backend, "require_solver_binary", missing_binary)
    monkeypatch.setattr(
        backend,
        "require_synthesis_dependency",
        lambda import_name, **_kwargs: (
            pytest.fail(import_name)
            if import_name == "highspy"
            else pytest.importorskip(import_name)
        ),
    )

    with pytest.raises(pytest.fail.Exception, match="highspy"):
        backend.require_solver_backend("highs")


@pytest.mark.skipif("not _highs_available()", reason="needs highspy")
def test_highs_solves_a_milp_with_options_and_no_option_file():
    model, duty, unit = _binary_model()

    run = backend.configure_pyomo_solver(
        model, "highs", solver_options={"time_limit": 30.0}
    )
    solved = backend.solve_pyomo_model(model, solver_name="highs")

    assert run.option_file is None
    assert model._path is None
    assert solved.failure_reason is None
    assert solved.status == 1
    assert solved.termination_condition == "optimal"
    assert solved.solver_options == {"time_limit": 30.0}
    assert solved.objective_value == pytest.approx(3.5)
    assert (duty.value, unit.value) == pytest.approx((3.0, 1.0))


def test_is_linear_checks_constraints_and_objective():
    model = PyomoModel()
    x = model.Var(value=1.0, lb=0.0, ub=5.0)
    z = model.Var(value=1, lb=0, ub=1, integer=True)
    fixed = model.Param(value=2.0)
    model.Equation(x <= 5.0 * z)
    # A fixed parameter times a variable is still linear.
    model.Equation(fixed * x >= 1.0)
    model.Minimize(x)

    assert model.is_linear()

    model.Equation(x * (1 - z) == 0.0)

    assert not model.is_linear()


@pytest.mark.parametrize(
    ("user_options", "expected"),
    [
        (None, {"bound_push": 1e-6, "bound_frac": 1e-6}),
        (
            {"bound_push": 1e-2, "max_iter": 50},
            {"bound_push": 1e-2, "bound_frac": 1e-6, "max_iter": 50},
        ),
    ],
)
def test_ipopt_pyomo_starts_close_to_its_bounds_by_default(
    monkeypatch, user_options, expected
):
    # Ipopt's default push of 1e-2 is relative to the bound, about 3.5 K at
    # 350 K, which moves warm-started temperatures off their solved values.
    monkeypatch.setattr(backend, "require_solver_backend", lambda _name: None)
    monkeypatch.setattr(backend, "_require_pyomo_solver_available", lambda *_args: None)
    model, _x, _y = _bounded_model(4.0)

    run = backend.configure_pyomo_solver(
        model, "ipopt-pyomo", solver_options=user_options
    )

    assert run.solver_options == expected
    # Options go on Ipopt's command line, so parallel evolution solves need
    # no option file and no working-directory lock.
    assert run.option_file is None
    assert model._path is None
