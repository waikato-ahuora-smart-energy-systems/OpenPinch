"""A Couenne solve that reaches its time limit fails instead of hanging."""

from __future__ import annotations

import json
import random
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from OpenPinch.analysis.heat_exchanger_networks.execution.executor import (
    LocalSynthesisExecutor,
    _solve_built_task,
)
from OpenPinch.analysis.heat_exchanger_networks.execution.settings import (
    workflow_settings_from_problem,
)
from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.targeting.pinch_design_method import (
    build_pinch_design_method_tasks,
)
from OpenPinch.application.problem import PinchProblem
from OpenPinch.contracts.synthesis.task import HeatExchangerNetworkSynthesisTask
from tests.support.paths import FIXTURES_ROOT


class _Model:
    def __init__(self, path: Path, *, seconds: float = 0.0) -> None:
        self.options = SimpleNamespace(
            SOLVER=None, SOLVER_EXTENSION=None, SOLVESTATUS=None, objfcnval=None
        )
        self.solver_options: list[str] = []
        self._path = str(path)
        self._seconds = seconds

    def solve(self, *, disp: bool, debug: int) -> None:
        del disp, debug
        time.sleep(self._seconds)
        # GEKKO reports an interrupted Pyomo solve as a successful one.
        self.options.SOLVESTATUS = 1
        self.options.objfcnval = 10.0


@pytest.fixture
def present_solver(monkeypatch) -> None:
    class _Factory:
        def available(self, exception_flag: bool = False) -> bool:
            del exception_flag
            return True

    def dependency(import_name: str, **_kwargs):
        if import_name == "pyomo.environ":
            return SimpleNamespace(SolverFactory=lambda _name: _Factory())
        return object()

    monkeypatch.setattr(
        backend, "require_solver_binary", lambda name, **_kwargs: f"/bin/{name}"
    )
    monkeypatch.setattr(backend, "require_synthesis_dependency", dependency)


def _solve(model: _Model, *, solver_options=None) -> backend.SolverRun:
    backend.configure_gekko_solver(model, "couenne", solver_options=solver_options)
    return backend.solve_gekko_model(model, solver_name="couenne")


def test_couenne_gets_the_default_time_limit(present_solver, tmp_path) -> None:
    run = backend.configure_gekko_solver(_Model(tmp_path / "run"), "couenne")

    limit = backend.DEFAULT_COUENNE_OPTIONS["time_limit"]
    assert run.solver_options == {"time_limit": limit}
    assert Path(run.option_file).read_text(encoding="utf-8") == (
        f"time_limit {limit}\n"
    )


def test_configured_options_keep_the_default_time_limit(
    present_solver, tmp_path
) -> None:
    run = backend.configure_gekko_solver(
        _Model(tmp_path / "run"), "couenne", solver_options={"node_limit": 50}
    )

    assert run.solver_options == {
        "time_limit": backend.DEFAULT_COUENNE_OPTIONS["time_limit"],
        "node_limit": 50,
    }


def test_configured_time_limit_replaces_the_default(present_solver, tmp_path) -> None:
    run = backend.configure_gekko_solver(
        _Model(tmp_path / "run"), "couenne", solver_options={"time_limit": 5}
    )

    assert run.solver_options == {"time_limit": 5}


def test_ipopt_has_no_default_option_file(present_solver, tmp_path) -> None:
    run = backend.configure_gekko_solver(_Model(tmp_path / "run"), "ipopt-pyomo")

    assert run.option_file is None


def test_solve_reaching_the_time_limit_fails_despite_a_returned_solution(
    present_solver, tmp_path
) -> None:
    run = _solve(
        _Model(tmp_path / "run", seconds=0.02), solver_options={"time_limit": 0.01}
    )

    assert run.status == 1
    assert run.failure_reason is not None
    assert "time limit of 0.01 s" in run.failure_reason


def test_solve_inside_the_time_limit_succeeds(present_solver, tmp_path) -> None:
    run = _solve(_Model(tmp_path / "run"), solver_options={"time_limit": 3600})

    assert run.failure_reason is None


def test_ipopt_is_not_failed_by_a_time_limit_option(present_solver, tmp_path) -> None:
    model = _Model(tmp_path / "run", seconds=0.02)
    backend.configure_gekko_solver(
        model, "ipopt-pyomo", solver_options={"max_wall_time": 0.01}
    )

    run = backend.solve_gekko_model(model, solver_name="ipopt-pyomo")

    assert run.failure_reason is None


def test_waiting_for_the_working_directory_is_not_solve_time(
    present_solver, tmp_path
) -> None:
    model = _Model(tmp_path / "run")
    backend.configure_gekko_solver(model, "couenne", solver_options={"time_limit": 0.1})
    held = threading.Event()
    release = threading.Event()

    def hold_lock() -> None:
        with backend._WORKING_DIRECTORY_LOCK:
            held.set()
            release.wait(timeout=5)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    held.wait(timeout=5)
    threading.Timer(0.2, release.set).start()
    run = backend.solve_gekko_model(model, solver_name="couenne")
    holder.join()

    assert run.failure_reason is None
    assert run.solve_time < 0.1


def test_failed_task_reports_the_reason_from_the_solver_run() -> None:
    task = HeatExchangerNetworkSynthesisTask(
        run_id="time-limit",
        method="thermal_derivative_method",
        approach_temperature=10.0,
        derivative_threshold=0.5,
        stage_count=2,
    )

    class _TimedOutProblem:
        def get_solution(self, **_kwargs):
            return SimpleNamespace(
                mSuccess=0,
                solver_run=SimpleNamespace(failure_reason="time limit reached"),
            )

    _task, outcome, _problem = _solve_built_task(
        (task, _TimedOutProblem(), False, None)
    )

    assert outcome.status == "failed"
    assert outcome.error == "time limit reached"


@pytest.mark.solver
def test_couenne_binary_stops_at_the_limit_in_its_option_file(
    monkeypatch, tmp_path
) -> None:
    # Couenne silently ignores the same limit under another name, such as
    # bonmin.time_limit, and this model then runs for well over 40 s.
    pyomo = pytest.importorskip("pyomo.environ")
    backend.require_solver_binary("couenne", purpose="the time limit test")
    size = 30
    coefficients = random.Random(1)
    quadratic = [
        [coefficients.uniform(-1, 1) for _ in range(size)] for _ in range(size)
    ]
    model = pyomo.ConcreteModel()
    model.x = pyomo.Var(range(size), bounds=(0.1, 10), initialize=1)
    model.y = pyomo.Var(range(size), domain=pyomo.Binary, initialize=0)
    model.objective = pyomo.Objective(
        expr=sum(
            quadratic[i][j] * model.x[i] * model.x[j]
            for i in range(size)
            for j in range(size)
        )
        + sum(pyomo.log(model.x[i] + model.y[i]) for i in range(size))
    )
    model.demand = pyomo.Constraint(
        expr=sum(model.x[i] * model.y[i] for i in range(size)) >= 5
    )
    model.link = pyomo.Constraint(
        range(size), rule=lambda m, i: m.x[i] <= 10 * m.y[i] + 2
    )
    backend._write_solver_option_file(
        SimpleNamespace(_path=str(tmp_path)),
        backend.COUENNE_OPTION_FILE,
        {"time_limit": 1},
    )
    monkeypatch.chdir(tmp_path)

    results = pyomo.SolverFactory("couenne").solve(model, load_solutions=False)

    assert "interrupted on limit" in str(results.solver.message)


@pytest.mark.solver
def test_short_limit_on_a_tiny_case_fails_tasks_with_the_reason(tmp_path) -> None:
    fixture = FIXTURES_ROOT / "openhens/Four-stream-Yee-and-Grossmann-1990-1.json"
    case = json.loads(fixture.read_text(encoding="utf-8"))
    case["options"]["HENS_SOLVER_OPTIONS_PDM"] = {"time_limit": 0.001}
    source = tmp_path / "case.json"
    source.write_text(json.dumps(case), encoding="utf-8")
    problem = PinchProblem(source=source)
    problem.target.all_heat_integration()
    tasks = build_pinch_design_method_tasks(workflow_settings_from_problem(problem))

    outcomes = LocalSynthesisExecutor(print_output=False).execute(
        tasks[:1], problem=problem, parent_outcomes={}, max_parallel=1
    )

    assert [outcome.status for outcome in outcomes] == ["failed"]
    assert "time limit of 0.001 s" in outcomes[0].error
