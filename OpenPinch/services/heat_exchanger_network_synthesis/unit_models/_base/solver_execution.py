"""Explicit-state solver execution for base HEN models."""

from __future__ import annotations

import logging

from ...common.solver import backend

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
