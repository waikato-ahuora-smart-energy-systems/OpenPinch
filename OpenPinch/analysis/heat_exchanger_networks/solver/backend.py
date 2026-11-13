"""Lazy solver backend helpers for migrated heat exchanger network equation models."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ....domain.solver_options import validate_solver_options
from .dependencies import (
    MissingSynthesisSolverError,
    require_solver_binary,
    require_synthesis_dependency,
)

PYOMO_SOLVERS = frozenset({"couenne", "ipopt-pyomo", "highs", "scip", "mindtpy"})
# Solvers that keep integer variables integer; the others relax them.
INTEGER_SOLVERS = frozenset({"couenne", "highs", "scip", "mindtpy"})
# Executables a solver needs. MindtPy's outer approximation solves its NLP
# subproblems with Ipopt and its MILP masters with HiGHS.
PYOMO_SOLVER_BINARIES = {
    "couenne": "couenne",
    "ipopt-pyomo": "ipopt",
    "mindtpy": "ipopt",
}
# Python packages a solver needs, checked before any executable.
PYOMO_SOLVER_MODULES = {"highs": "highspy", "scip": "pyscipopt", "mindtpy": "highspy"}
# Solvers that take their options at solve time (in process, or on the
# command line) rather than from an option file in the working directory.
SOLVE_TIME_OPTION_SOLVERS = frozenset({"highs", "ipopt-pyomo", "scip", "mindtpy"})
# Solvers that can start a MIP from the model's current values.
WARM_START_SOLVERS = frozenset({"highs", "scip"})
# Each integer solver's wall-time limit option, in seconds.
TIME_LIMIT_OPTIONS = {
    "highs": "time_limit",
    "couenne": "time_limit",
    "scip": "limits/time",
    "mindtpy": "time_limit",
}
# Seconds allowed for a PDM side's fewest-units re-solve, which can find a
# sparse network long before it proves the fewest units.
FEWEST_UNITS_TIME_LIMIT = 60.0
# Seconds allowed for a PDM side's first, minimum-utility solve, unless the
# user sets a limit: with a utility choice it may not close its gap quickly.
# A HiGHS solve stopped there keeps its incumbent, as unconverged.
PDM_TIME_LIMIT = 600.0
PYOMO_SOLVER_FACTORY_NAMES = {
    "couenne": "couenne",
    "ipopt-pyomo": "ipopt",
    "highs": "appsi_highs",
    "scip": "scip_direct",
    "mindtpy": "mindtpy",
}
COUENNE_OPTION_FILE = "couenne.opt"
IPOPT_OPTION_FILE = "ipopt.opt"
DEFAULT_COUENNE_OPTIONS: dict[str, Any] = {}
# Ipopt pushes starting values off their bounds by 1e-2 relative to the bound
# (about 3.5 K at 350 K), which undoes warm starts from solved networks.
DEFAULT_IPOPT_PYOMO_OPTIONS: dict[str, Any] = {
    "bound_push": 1e-6,
    "bound_frac": 1e-6,
}
# Defaults under the given options: SCIP prints nothing, and MindtPy runs
# outer approximation with HiGHS masters and Ipopt subproblems.
DEFAULT_SOLVER_OPTIONS: dict[str, dict[str, Any]] = {
    "ipopt-pyomo": DEFAULT_IPOPT_PYOMO_OPTIONS,
    "scip": {"display/verblevel": 0},
    "mindtpy": {"strategy": "OA", "mip_solver": "appsi_highs", "nlp_solver": "ipopt"},
}
# Pyomo termination conditions of a converged solve. A solve that stops at
# an iteration or time limit, for example, still passes, but is not converged.
CONVERGED_TERMINATION_CONDITIONS = frozenset(
    {"optimal", "locallyOptimal", "globallyOptimal", "feasible"}
)
# Solver statuses and termination conditions a Pyomo solve accepts. The
# others, including "unbounded" and "noSolution", leave no solution to read.
PASSING_SOLVER_STATUSES = frozenset({"ok", "warning"})
PASSING_TERMINATION_CONDITIONS = CONVERGED_TERMINATION_CONDITIONS | frozenset(
    {
        "unknown",
        "maxTimeLimit",
        "maxIterations",
        "minFunctionValue",
        "minStepLength",
        "maxEvaluations",
        "other",
        "intermediateNonInteger",
    }
)


@dataclass(frozen=True)
class SolverRun:
    """Serializable metadata for one local heat exchanger network solver attempt."""

    name: str
    extension: str | int | None = None
    status: int | None = None
    objective_value: float | None = None
    solve_time: float | None = None
    failure_reason: str | None = None
    solver_options: dict[str, Any] | None = None
    option_file: str | None = None
    termination_condition: str | None = None

    @property
    def converged(self) -> bool:
        """False when the solver stopped before converging, for example at its
        iteration limit. Solves without a recorded condition count as converged."""

        return (
            self.termination_condition is None
            or self.termination_condition in CONVERGED_TERMINATION_CONDITIONS
        )


def cleanup_model_run_directory(model: Any) -> None:
    """Remove a model's temporary run directory, once.

    Solved values stay on the model's variables, so this is safe once the
    results have been read. Models without a run directory, repeat calls and
    models whose directory is already gone are no-ops.
    """

    path = getattr(model, "_path", None)
    if isinstance(path, (str, os.PathLike)) and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
        model._openpinch_cleaned_up = True


def require_solver_backend(solver_name: str) -> None:
    """Validate optional Python packages and executables for one solver name."""

    if solver_name in PYOMO_SOLVERS:
        if solver_name in PYOMO_SOLVER_MODULES:
            require_synthesis_dependency(
                PYOMO_SOLVER_MODULES[solver_name],
                purpose=f"{solver_name} heat exchanger network synthesis solves",
            )
        if solver_name in PYOMO_SOLVER_BINARIES:
            require_solver_binary(
                PYOMO_SOLVER_BINARIES[solver_name],
                purpose=f"{solver_name} heat exchanger network synthesis solves",
            )
        require_synthesis_dependency(
            "pyomo.environ",
            package="pyomo",
            purpose=f"{solver_name} solver factory access",
        )
        return

    raise MissingSynthesisSolverError(
        "Unsupported heat exchanger network synthesis solver "
        f"{solver_name!r}. Supported solvers "
        f"are: {', '.join(sorted(PYOMO_SOLVERS))}."
    )


def configure_pyomo_solver(
    model: Any,
    solver_name: str,
    *,
    solver_options: Mapping[str, Any] | Sequence[str] | None = None,
) -> SolverRun:
    """Check a Pyomo solver and write its option file for a ``PyomoModel``.

    Couenne reads options from ``couenne.opt`` in the working directory, so
    its options go to a file in a run directory created for the model and
    removed by ``cleanup_model_run_directory``. HiGHS and SCIP (in process),
    Ipopt (on its command line) and MindtPy (as solve arguments) take their
    options at solve time, with no file, over ``DEFAULT_SOLVER_OPTIONS``.
    """

    if solver_name not in PYOMO_SOLVERS:
        raise MissingSynthesisSolverError(
            f"{solver_name!r} cannot solve a native Pyomo model. Supported "
            f"solvers are: {', '.join(sorted(PYOMO_SOLVERS))}."
        )
    require_solver_backend(solver_name)
    effective_options = _merge_solver_options(
        DEFAULT_SOLVER_OPTIONS.get(solver_name, {}), solver_options
    )
    option_file = None
    if solver_name in SOLVE_TIME_OPTION_SOLVERS:
        model._openpinch_solver_options = effective_options or None
    elif effective_options:
        if model._path is None:
            model._path = tempfile.mkdtemp(prefix="openpinch-pyomo-")
        option_file = _write_solver_option_file(
            model,
            COUENNE_OPTION_FILE if solver_name == "couenne" else IPOPT_OPTION_FILE,
            effective_options,
        )
    _require_pyomo_solver_available(
        solver_name, PYOMO_SOLVER_FACTORY_NAMES[solver_name]
    )
    return SolverRun(
        name=solver_name,
        extension="pyomo",
        solver_options=effective_options or None,
        option_file=option_file,
    )


def _require_pyomo_solver_available(solver_name: str, executable: str) -> None:
    pyomo = require_synthesis_dependency(
        "pyomo.environ",
        package="pyomo",
        purpose=f"{solver_name} solver factory availability checks",
    )
    if not pyomo.SolverFactory(executable).available(exception_flag=False):
        raise MissingSynthesisSolverError(
            f"The {executable!r} Pyomo solver is not available. Install the "
            "solver binary, confirm it is on PATH, and rerun the heat "
            "exchanger network synthesis solver tests."
        )


def _has_incumbent(results: Any, model: Any) -> bool:
    """Whether a solve that stopped early loaded a feasible solution."""

    if len(getattr(results, "solution", ())) > 0:
        return True
    problem = getattr(results, "problem", None)
    minimising = getattr(model.objective, "sense", 1) == 1
    bound = getattr(problem, "upper_bound" if minimising else "lower_bound", None)
    return bound is not None and math.isfinite(float(bound))


def solve_pyomo_model(
    model: Any,
    *,
    solver_name: str,
    disp: bool = False,
) -> SolverRun:
    """Solve a ``PyomoModel`` and normalise the result into a ``SolverRun``.

    Values are loaded onto the model's variables. A solver status outside
    ``PASSING_SOLVER_STATUSES`` or a termination condition outside
    ``PASSING_TERMINATION_CONDITIONS`` becomes the failure reason, with
    status 0.
    """

    pyomo = require_synthesis_dependency(
        "pyomo.environ",
        package="pyomo",
        purpose=f"{solver_name} heat exchanger network synthesis solves",
    )
    _restore_run_directory(model)
    start = time.time()
    status = 0
    objective_value = None
    termination_condition = None
    failure_reason = None
    try:
        with _solver_working_directory(model, "pyomo"):
            solver = pyomo.SolverFactory(PYOMO_SOLVER_FACTORY_NAMES[solver_name])
            options = getattr(model, "_openpinch_solver_options", None) or {}
            if solver_name == "mindtpy":
                # MindtPy takes its configuration as solve arguments.
                results = solver.solve(model.model, tee=disp, **options)
            else:
                if solver_name in SOLVE_TIME_OPTION_SOLVERS:
                    solver.options.update(options)
                warm_start = (
                    {"warmstart": True}
                    if getattr(model, "warm_start", False)
                    and solver_name in WARM_START_SOLVERS
                    else {}
                )
                results = solver.solve(model.model, tee=disp, **warm_start)
        solver_status = str(results.solver.status)
        termination_condition = str(results.solver.termination_condition)
        stopped_with_solution = (
            # HiGHS reports a time-limited solve as "aborted", even with an
            # incumbent loaded.
            solver_status == "aborted"
            and termination_condition in PASSING_TERMINATION_CONDITIONS
            and _has_incumbent(results, model)
        )
        if solver_status not in PASSING_SOLVER_STATUSES and not stopped_with_solution:
            failure_reason = f"Solver exited with status: {solver_status}"
        elif termination_condition not in PASSING_TERMINATION_CONDITIONS:
            failure_reason = f"Solver termination condition: {termination_condition}"
        else:
            status = 1
            objective_value = _as_float_or_none(
                pyomo.value(model.objective, exception=False)
            )
    except Exception as exc:  # Pyomo and solver exceptions become task metadata.
        failure_reason = str(exc) or exc.__class__.__name__

    return SolverRun(
        name=solver_name,
        extension="pyomo",
        status=status,
        objective_value=objective_value,
        solve_time=time.time() - start,
        failure_reason=failure_reason,
        solver_options=getattr(model, "_openpinch_solver_options", None),
        option_file=getattr(model, "_openpinch_solver_option_file", None),
        termination_condition=termination_condition,
    )


def _as_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _merge_solver_options(
    defaults: Mapping[str, Any],
    overrides: Mapping[str, Any] | Sequence[str] | None,
) -> dict[str, Any]:
    """Return solver options with user-provided values taking precedence."""

    merged = dict(defaults)
    merged.update(_normalise_solver_options(overrides))
    return merged


def _normalise_solver_options(
    solver_options: Mapping[str, Any] | Sequence[str] | None,
) -> dict[str, Any]:
    """Normalise mapping or ``["name value"]`` options into insertion order."""

    if solver_options is None:
        return {}
    if isinstance(solver_options, Mapping):
        return validate_solver_options(
            {
                str(name): value
                for name, value in solver_options.items()
                if value is not None
            }
        )
    if isinstance(solver_options, (str, bytes)) or not isinstance(
        solver_options,
        Sequence,
    ):
        raise ValueError("Solver options must be a mapping or a list of strings.")

    normalised: dict[str, Any] = {}
    for raw_option in solver_options:
        option = str(raw_option).strip()
        if not option or option.startswith("#"):
            continue
        name, sep, value = option.partition(" ")
        normalised[name] = value.strip() if sep else ""
    return validate_solver_options(normalised)


def _format_solver_option_lines(options: Mapping[str, Any]) -> list[str]:
    return [
        f"{name} {_format_solver_option_value(value)}"
        for name, value in validate_solver_options(options).items()
    ]


def _format_solver_option_value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _write_solver_option_file(
    model: Any,
    filename: str,
    options: Mapping[str, Any],
) -> str:
    model_path = getattr(model, "_path", None)
    if model_path is None:
        raise RuntimeError(
            f"Cannot write {filename}: the model does not expose a run path."
        )
    directory = Path(model_path)
    directory.mkdir(parents=True, exist_ok=True)
    option_path = directory / filename
    option_path.write_text(
        "\n".join(_format_solver_option_lines(options)) + "\n",
        encoding="utf-8",
    )
    model._openpinch_solver_options = dict(options)
    model._openpinch_solver_option_file = str(option_path)
    return str(option_path)


def _restore_run_directory(model: Any) -> None:
    """Recreate a cleaned-up run directory so a model can be solved again."""

    if not getattr(model, "_openpinch_cleaned_up", False):
        return
    path = model._path
    if os.path.isdir(path):
        return
    os.makedirs(path, exist_ok=True)
    options = getattr(model, "_openpinch_solver_options", None)
    option_file = getattr(model, "_openpinch_solver_option_file", None)
    if options and option_file is not None:
        _write_solver_option_file(model, Path(option_file).name, options)


# The working directory is process-wide. Pyomo solvers read their option file
# from it, so solves that need it are serialised: parallel evolution threads
# would otherwise change it under one another.
_WORKING_DIRECTORY_LOCK = threading.RLock()


@contextmanager
def _solver_working_directory(model: Any, extension: str | int | None):
    option_file = getattr(model, "_openpinch_solver_option_file", None)
    if extension != "pyomo" or option_file is None:
        yield
        return

    with _WORKING_DIRECTORY_LOCK:
        previous = Path.cwd()
        os.chdir(Path(option_file).parent)
        try:
            yield
        finally:
            os.chdir(previous)
