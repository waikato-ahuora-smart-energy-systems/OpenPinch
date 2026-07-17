"""Explicit-state solver execution for base HEN models."""

from __future__ import annotations

import logging
from typing import Any

from ...solver import backend

logger = logging.getLogger(__name__)


def optimise(model, print_output: bool) -> None:
    """Solve the concrete model and extract plain result data on success."""

    total_solve_time = 0.0
    piecewise_mappings = getattr(model, "_piecewise_active_mappings", [])
    active_mapping_stable = not piecewise_mappings or getattr(model, "integers", False)
    for _attempt in range(8):
        model.solver_run = backend.solve_gekko_model(
            model.m,
            solver_name=model.solver,
            disp=False,
            debug=0,
        )
        total_solve_time += float(model.solver_run.solve_time or 0.0)
        if model.solver_run.failure_reason is not None:
            break
        active_mapping_stable = (
            True
            if not piecewise_mappings
            else not model._update_piecewise_active_segments()
        )
        if active_mapping_stable:
            break
    model.solve_time = total_solve_time

    if (
        model.solver_run.failure_reason is not None
        and piecewise_mappings
        and not getattr(model, "integers", False)
    ):
        model.solver_run = backend.SolverRun(
            name=model.solver_run.name,
            extension=model.solver_run.extension,
            status=model.solver_run.status,
            objective_value=model.solver_run.objective_value,
            solve_time=total_solve_time,
            failure_reason=(
                f"{model.solver_run.failure_reason}; segmented-stream active "
                "interval solve was unresolved, so use APOPT or Couenne"
            ),
        )

    if model.solver_run.failure_reason is None and not active_mapping_stable:
        model.solver_run = backend.SolverRun(
            name=model.solver_run.name,
            extension=model.solver_run.extension,
            status=model.solver_run.status,
            objective_value=model.solver_run.objective_value,
            solve_time=total_solve_time,
            failure_reason=(
                "piecewise active segments did not stabilise; use APOPT or "
                "Couenne for interval-disjunctive segmented-stream solving"
            ),
        )

    if model.solver_run.failure_reason is not None:
        model.mSuccess = 0
        logger.error("[Failed] [model: %s] [path: %s]", model.name, model.m._path)
    elif model.m.options.SOLVESTATUS == 1:
        if model.m.options.objfcnval + model.tol < 0:
            model.mSuccess = 0
            model.solver_run = backend.SolverRun(
                name=model.solver_run.name,
                extension=model.solver_run.extension,
                status=model.solver_run.status,
                objective_value=model.solver_run.objective_value,
                solve_time=model.solver_run.solve_time,
                failure_reason="negative objective value",
            )
            logger.error(
                "[Failed] [model: %s] [path: %s]",
                model.name,
                model.m._path,
            )
        else:
            model.mSuccess = model.m.options.SOLVESTATUS
            logger.info(
                "[Success] [model: %s] [path: %s]",
                model.name,
                model.m._path,
            )
    else:
        model.mSuccess = model.m.options.SOLVESTATUS
        logger.error(
            "[Failed] [model: %s] [path: %s] [status: %s]",
            model.name,
            model.m._path,
            model.m.options.SOLVESTATUS,
        )

    if model.mSuccess:
        model.get_post_process()
        if print_output:
            model.output_to_cmd_line()


def setup_model(model) -> None:
    """Create and configure the GEKKO model behind optional guards."""

    model.m = backend.create_gekko_model(remote=False)
    model.mSuccess: int = 0
    model.solver_run = backend.configure_gekko_solver(
        model.m,
        model.solver,
        solver_options=model.solver_options,
    )


def _solver_value(model, value: Any) -> float:
    try:
        return float(value[0])
    except TypeError, IndexError, KeyError:
        return float(value)


def _set_value(model, variable: Any, value: float, *, brackets: bool = False) -> None:
    """Assign GEKKO values while preserving source bound-clamping behavior."""

    if type(variable).__name__ == "GKVariable":
        if variable.lower is not None:
            value = max(variable.lower, value)
        if variable.upper is not None:
            value = min(variable.upper, value)
        variable.VALUE.value = [value] if brackets else value
        return
    if type(variable).__name__ == "GKParameter":
        variable.VALUE.value = [value] if brackets else value
        return


def output_to_cmd_line(model) -> None:
    """Emit the same solved-array diagnostics as the source base model."""

    if model.mSuccess != 1:
        return
    logger.info("Successful Solve.Path %s name %s", model.m._path, model.name)
    logger.info("Objective 0: %s", model.m.options.objfcnval)
    logger.info("Objective 1: %s", model.TAC)
    logger.info("Total Units: %s", model.n_units)
    logger.info("Total Recovery Units: %s", model.n_recovery_units)
    logger.info("T hot: %s", model.T_h)
    logger.info("T cold: %s", model.T_c)
    logger.info("theta 1: %s", model.theta_1)
    logger.info("theta 2: %s", model.theta_2)
    logger.info("Heat recovery Q: %s", model.Q_r)
    logger.info("Heat recovery z: %s", model.z)
    logger.info("Heat recovery LMTD: %s", model.LMTD_r)
    logger.info("Heat recovery area: %s", model.area_r)
    logger.info("Q_r total: %s", model.Q_r_total)
    logger.info("Cold utility Q: %s", model.Q_c)
    logger.info("Cold utility z: %s", model.z_cu)
    logger.info("Cold utility LMTD: %s", model.LMTD_cu)
    logger.info("Cold utility area: %s", model.area_cu)
    logger.info("Q_cu total: %s", model.Q_cu_total)
    logger.info("Hot utility Q: %s", model.Q_h)
    logger.info("Hot utility z: %s", model.z_hu)
    logger.info("Hot utility LMTD: %s", model.LMTD_hu)
    logger.info("Hot utility area: %s", model.area_hu)
    logger.info("Q_hu total: %s", model.Q_hu_total)
