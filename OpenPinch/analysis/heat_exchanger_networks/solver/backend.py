"""Lazy solver backend helpers for migrated heat exchanger network equation models."""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
import warnings
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

PYOMO_SOLVERS = frozenset({"couenne", "ipopt-pyomo", "highs"})
GEKKO_SOLVERS = frozenset({"ipopt-GEKKO", "apopt"})
# Solvers that keep integer variables integer; the others relax them.
INTEGER_SOLVERS = frozenset({"apopt", "couenne", "highs"})
PYOMO_SOLVER_BINARIES = {
    "couenne": "couenne",
    "ipopt-pyomo": "ipopt",
}
# Solvers run in-process through a Python package rather than an executable.
PYOMO_SOLVER_MODULES = {"highs": "highspy"}
PYOMO_SOLVER_FACTORY_NAMES = {
    "couenne": "couenne",
    "ipopt-pyomo": "ipopt",
    "highs": "appsi_highs",
}
COUENNE_OPTION_FILE = "couenne.opt"
IPOPT_OPTION_FILE = "ipopt.opt"
DEFAULT_COUENNE_OPTIONS: dict[str, Any] = {}
DEFAULT_IPOPT_GEKKO_OPTIONS: dict[str, Any] = {
    "tol": "1e-3",
    "acceptable_tol": "1e-2",
    "constr_viol_tol": "1e-2",
    "acceptable_constr_viol_tol": "1e-1",
    "compl_inf_tol": "1e-2",
    "max_iter": 1000,
    "print_level": 5,
}
# Pyomo termination conditions of a converged solve. GEKKO's Pyomo extension
# only warns on the others (an iteration or time limit, for example) and still
# reports the solve as successful.
CONVERGED_TERMINATION_CONDITIONS = frozenset(
    {"optimal", "locallyOptimal", "globallyOptimal", "feasible"}
)
# Solver statuses and termination conditions a native Pyomo solve accepts:
# those GEKKO's Pyomo extension accepts (the others raise there), except
# "unbounded" and "noSolution", which leave no solution to read.
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
GEKKO_NUMPY_ARRAY_COPY_DEPRECATION = (
    r"__array__ implementation doesn't accept a copy keyword.*"
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


def create_gekko_model(*, remote: bool = False) -> Any:
    """Create a GEKKO model after the optional dependency has been requested.

    The model records the Pyomo termination condition of each solve, which
    GEKKO itself only prints.
    """

    require_synthesis_dependency(
        "gekko",
        package="gekko",
        purpose="GEKKO heat exchanger network equation model construction",
    )
    return _termination_recording_class("_TerminationRecordingGEKKO")(remote=remote)


def __getattr__(name: str) -> type:
    # Pickle looks classes up by module and name, so unpickling a model in a
    # process pool worker resolves the lazily built classes here.
    if name in _TERMINATION_RECORDING_CLASS_NAMES:
        return _termination_recording_class(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_TERMINATION_RECORDING_CLASS_NAMES = frozenset(
    {"_TerminationRecordingGEKKO", "_TerminationRecordingPyomoConverter"}
)
_TERMINATION_RECORDING_CLASS_LOCK = threading.Lock()


def _termination_recording_class(name: str) -> type:
    """Return a GEKKO subclass published as a module attribute of this module.

    The classes are built on first use, so importing this module does not
    import gekko, and published once, so models pickle by reference to them.
    """

    with _TERMINATION_RECORDING_CLASS_LOCK:
        if name not in globals():
            _define_termination_recording_classes()
    return globals()[name]


def _define_termination_recording_classes() -> None:
    from gekko import GEKKO
    from gekko.gk_solver_extension_pyomo import PyomoConverter

    class _TerminationRecordingPyomoConverter(PyomoConverter):
        def handle_status(self, status_dict, status, status_string) -> None:
            if isinstance(status, self.TerminationCondition):
                self._gekko_model._openpinch_termination_condition = str(status)
            super().handle_status(status_dict, status, status_string)

    class _TerminationRecordingGEKKO(GEKKO):
        def solver_extension_pyomo(self, disp=True):
            self.solve_with_converter(_TerminationRecordingPyomoConverter, disp)

    for cls in (_TerminationRecordingPyomoConverter, _TerminationRecordingGEKKO):
        cls.__qualname__ = cls.__name__
        globals()[cls.__name__] = cls


def cleanup_gekko_model(model: Any) -> None:
    """Remove a GEKKO model's temporary run directory, once.

    Solved values stay on the model's variables, so this is safe once the
    results have been read. Repeat calls and models whose directory is
    already gone are no-ops (GEKKO's own ``cleanup`` prints when the
    directory is missing).
    """

    if model is None:
        return
    path = getattr(model, "_path", None)
    if isinstance(path, (str, os.PathLike)):
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
            model._openpinch_cleaned_up = True
        return
    cleanup = getattr(model, "cleanup", None)
    if callable(cleanup):
        cleanup()


def require_solver_backend(solver_name: str) -> None:
    """Validate optional Python packages and executables for one solver name."""

    if solver_name in PYOMO_SOLVERS:
        if solver_name in PYOMO_SOLVER_MODULES:
            require_synthesis_dependency(
                PYOMO_SOLVER_MODULES[solver_name],
                purpose=f"{solver_name} heat exchanger network synthesis solves",
            )
        else:
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

    if solver_name in GEKKO_SOLVERS:
        require_synthesis_dependency(
            "gekko",
            package="gekko",
            purpose=f"{solver_name} heat exchanger network synthesis solves",
        )
        return

    raise MissingSynthesisSolverError(
        "Unsupported heat exchanger network synthesis solver "
        f"{solver_name!r}. Supported solvers "
        f"are: {', '.join(sorted(PYOMO_SOLVERS | GEKKO_SOLVERS))}."
    )


def configure_gekko_solver(
    model: Any,
    solver_name: str,
    *,
    solver_options: Mapping[str, Any] | Sequence[str] | None = None,
) -> SolverRun:
    """Apply GEKKO/Pyomo solver settings without eager imports."""

    require_solver_backend(solver_name)

    if solver_name in PYOMO_SOLVERS:
        extension: str | int | None = "pyomo"
    elif solver_name in GEKKO_SOLVERS:
        extension = 0
    else:
        extension = getattr(model.options, "SOLVER_EXTENSION", None)

    model.options.SOLVER_EXTENSION = extension
    model.options.SOLVER = solver_name.split("-")[0]
    effective_options: dict[str, Any] = {}
    option_file = None

    if solver_name == "ipopt-GEKKO":
        effective_options = _merge_solver_options(
            DEFAULT_IPOPT_GEKKO_OPTIONS,
            solver_options,
        )
        model.solver_options = _format_solver_option_lines(effective_options)

    if solver_name == "apopt":
        model.options.MAX_ITER = 1000
        model.options.RTOL = 1e-2
        model.options.OTOL = 1e-2
        effective_options = _normalise_solver_options(solver_options)
        if effective_options:
            model.solver_options = _format_solver_option_lines(effective_options)

    if model.options.SOLVER_EXTENSION == "pyomo":
        if solver_name == "couenne":
            effective_options = _normalise_solver_options(solver_options)
            if effective_options:
                option_file = _write_solver_option_file(
                    model,
                    COUENNE_OPTION_FILE,
                    effective_options,
                )
        else:
            effective_options = _normalise_solver_options(solver_options)
            if effective_options:
                option_file = _write_solver_option_file(
                    model,
                    IPOPT_OPTION_FILE,
                    effective_options,
                )
        _require_pyomo_solver_available(solver_name, model.options.SOLVER)

    return SolverRun(
        name=solver_name,
        extension=extension,
        solver_options=effective_options or None,
        option_file=option_file,
    )


def solve_gekko_model(
    model: Any,
    *,
    solver_name: str,
    disp: bool = False,
    debug: int = 0,
) -> SolverRun:
    """Run a configured GEKKO model and normalize solver-specific failures."""

    extension = getattr(model.options, "SOLVER_EXTENSION", None)
    _restore_run_directory(model)
    model._openpinch_termination_condition = None
    start = time.time()
    failure_reason = None
    try:
        with _solver_working_directory(model, extension):
            with suppress_gekko_numpy_array_copy_deprecation():
                model.solve(disp=disp, debug=debug)
    except Exception as exc:  # GEKKO/Pyomo exceptions become task metadata.
        failure_reason = str(exc) or exc.__class__.__name__
    solve_time = time.time() - start

    status = getattr(model.options, "SOLVESTATUS", None)
    objective_value = getattr(model.options, "objfcnval", None)
    if failure_reason is None and status != 1:
        failure_reason = f"solver status {status}"

    return SolverRun(
        name=solver_name,
        extension=extension,
        status=status,
        objective_value=_as_float_or_none(objective_value),
        solve_time=solve_time,
        failure_reason=failure_reason,
        solver_options=getattr(model, "_openpinch_solver_options", None),
        option_file=getattr(model, "_openpinch_solver_option_file", None),
        termination_condition=model._openpinch_termination_condition,
    )


def configure_pyomo_solver(
    model: Any,
    solver_name: str,
    *,
    solver_options: Mapping[str, Any] | Sequence[str] | None = None,
) -> SolverRun:
    """Check a Pyomo solver and write its option file for a ``PyomoModel``.

    Couenne and Ipopt read options from ``couenne.opt`` or ``ipopt.opt`` in
    the working directory, so options go to a file in a run directory created
    for the model and removed by ``cleanup_gekko_model``. HiGHS runs in
    process and takes its options at solve time, with no file.
    """

    if solver_name not in PYOMO_SOLVERS:
        raise MissingSynthesisSolverError(
            f"{solver_name!r} cannot solve a native Pyomo model. Supported "
            f"solvers are: {', '.join(sorted(PYOMO_SOLVERS))}."
        )
    require_solver_backend(solver_name)
    effective_options = _normalise_solver_options(solver_options)
    option_file = None
    if solver_name in PYOMO_SOLVER_MODULES:
        # In-process solvers take their options at solve time.
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


def solve_pyomo_model(
    model: Any,
    *,
    solver_name: str,
    disp: bool = False,
) -> SolverRun:
    """Solve a ``PyomoModel`` and normalise the result like a GEKKO solve.

    Values are loaded onto the model's variables. A solver status or
    termination condition GEKKO's Pyomo extension would reject becomes the
    failure reason, with status 0.
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
            if solver_name in PYOMO_SOLVER_MODULES:
                solver.options.update(
                    getattr(model, "_openpinch_solver_options", None) or {}
                )
            results = solver.solve(model.model, tee=disp)
        solver_status = str(results.solver.status)
        termination_condition = str(results.solver.termination_condition)
        if solver_status not in PASSING_SOLVER_STATUSES:
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


@contextmanager
def suppress_gekko_numpy_array_copy_deprecation():
    """Suppress GEKKO's NumPy 2 array-copy deprecation during solver writes."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=GEKKO_NUMPY_ARRAY_COPY_DEPRECATION,
            category=DeprecationWarning,
        )
        yield


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
            f"Cannot write {filename}: GEKKO model does not expose a run path."
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
