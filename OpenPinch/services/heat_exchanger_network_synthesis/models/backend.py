"""Lazy solver backend helpers for migrated HEN equation models."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._dependencies import (
    MissingSynthesisSolverError,
    require_solver_binary,
    require_synthesis_dependency,
)

PYOMO_SOLVERS = frozenset({"couenne", "ipopt-pyomo"})
GEKKO_SOLVERS = frozenset({"ipopt-GEKKO", "apopt"})
PYOMO_SOLVER_BINARIES = {
    "couenne": "couenne",
    "ipopt-pyomo": "ipopt",
}
COUENNE_OPTION_FILE = "couenne.opt"
IPOPT_OPTION_FILE = "ipopt.opt"
DEFAULT_COUENNE_OPTIONS: dict[str, Any] = {
    "problem_print_level": 3,
    "node_limit": 2000,
    "feas_tolerance": 0.01,
    "allowable_gap": 0.01,
    "allowable_fraction_gap": 0.1,
    "delete_redundant": "yes",
}
DEFAULT_IPOPT_GEKKO_OPTIONS: dict[str, Any] = {
    "tol": "1e-3",
    "acceptable_tol": "1e-2",
    "constr_viol_tol": "1e-2",
    "acceptable_constr_viol_tol": "1e-1",
    "compl_inf_tol": "1e-2",
    "max_iter": 1000,
    "print_level": 5,
}


@dataclass(frozen=True)
class SolverRun:
    """Serializable metadata for one local HEN solver attempt."""

    name: str
    extension: str | int | None = None
    status: int | None = None
    objective_value: float | None = None
    solve_time: float | None = None
    failure_reason: str | None = None
    solver_options: dict[str, Any] | None = None
    option_file: str | None = None


def create_gekko_model(*, remote: bool = False) -> Any:
    """Create a GEKKO model after the optional dependency has been requested."""

    gekko = require_synthesis_dependency(
        "gekko",
        package="gekko",
        purpose="GEKKO HEN equation model construction",
    )
    return gekko.GEKKO(remote=remote)


def require_solver_backend(solver_name: str) -> None:
    """Validate optional Python packages and executables for one solver name."""

    if solver_name in PYOMO_SOLVERS:
        binary_name = PYOMO_SOLVER_BINARIES[solver_name]
        require_solver_binary(
            binary_name,
            purpose=f"{solver_name} HEN synthesis solves",
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
            purpose=f"{solver_name} HEN synthesis solves",
        )
        return

    raise MissingSynthesisSolverError(
        f"Unsupported HEN synthesis solver {solver_name!r}. Supported solvers "
        f"are: {', '.join(sorted(PYOMO_SOLVERS | GEKKO_SOLVERS))}."
    )


def configure_gekko_solver(
    model: Any,
    solver_name: str,
    *,
    solver_options: Mapping[str, Any] | Sequence[str] | None = None,
) -> SolverRun:
    """Apply the legacy GEKKO/Pyomo solver settings without eager imports."""

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
            effective_options = _merge_solver_options(
                DEFAULT_COUENNE_OPTIONS,
                solver_options,
            )
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
        pyomo = require_synthesis_dependency(
            "pyomo.environ",
            package="pyomo",
            purpose=f"{solver_name} solver factory availability checks",
        )
        solver_factory = pyomo.SolverFactory(model.options.SOLVER)
        try:
            available = solver_factory.available(exception_flag=False)
        except TypeError:
            available = solver_factory.available(False)
        if not available:
            raise MissingSynthesisSolverError(
                f"The {model.options.SOLVER!r} Pyomo solver is not available. "
                "Install the solver binary, confirm it is on PATH, and rerun "
                "the HEN synthesis solver tests."
            )

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
    start = time.time()
    failure_reason = None
    try:
        with _solver_working_directory(model, extension):
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
        return {
            str(name): value
            for name, value in solver_options.items()
            if value is not None
        }
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
    return normalised


def _format_solver_option_lines(options: Mapping[str, Any]) -> list[str]:
    return [
        f"{name} {_format_solver_option_value(value)}".rstrip()
        for name, value in options.items()
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


@contextmanager
def _solver_working_directory(model: Any, extension: str | int | None):
    option_file = getattr(model, "_openpinch_solver_option_file", None)
    if extension != "pyomo" or option_file is None:
        yield
        return

    previous = Path.cwd()
    os.chdir(Path(option_file).parent)
    try:
        yield
    finally:
        os.chdir(previous)
