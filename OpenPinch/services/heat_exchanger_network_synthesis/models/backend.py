"""Lazy solver backend helpers for migrated HEN equation models."""

from __future__ import annotations

import time
from dataclasses import dataclass
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


@dataclass(frozen=True)
class SolverRun:
    """Serializable metadata for one local HEN solver attempt."""

    name: str
    extension: str | int | None = None
    status: int | None = None
    objective_value: float | None = None
    solve_time: float | None = None
    failure_reason: str | None = None


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


def configure_gekko_solver(model: Any, solver_name: str) -> SolverRun:
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

    if solver_name == "ipopt-GEKKO":
        model.solver_options = [
            "tol 1e-3",
            "acceptable_tol 1e-2",
            "constr_viol_tol 1e-2",
            "acceptable_constr_viol_tol 1e-1",
            "compl_inf_tol 1e-2",
            "max_iter 1000",
            "print_level 5",
        ]

    if solver_name == "apopt":
        model.options.MAX_ITER = 1000
        model.options.RTOL = 1e-2
        model.options.OTOL = 1e-2

    if model.options.SOLVER_EXTENSION == "pyomo":
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

    return SolverRun(name=solver_name, extension=extension)


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
    )


def _as_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
