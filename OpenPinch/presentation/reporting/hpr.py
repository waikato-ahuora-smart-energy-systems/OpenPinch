"""Plain-JSON summaries of heat pump and refrigeration (HPR) targets."""

from __future__ import annotations

from typing import Any

from ...contracts.hpr import HPRTargetingError


def _floats(values) -> list[float]:
    return [] if values is None else [float(value) for value in values]


def hpr_summary(target: Any, label: str | None = None) -> dict[str, Any]:
    """Return the headline results of a solved HPR target as plain JSON values.

    Args:
        target: A solved heat pump or refrigeration target, such as the result
            of ``problem.target.carnot_heat_pump(...)``.
        label: Optional name stored under ``"name"``.

    Returns:
        A dict with the backend, cycle, selected and achieved load, objective,
        period ids and weights, design vector and simulated loop count.
    """
    details = target.hpr_details
    record = details.target_simulation_record
    load = target.hpr_load
    return {
        "name": label,
        "status": "feasible",
        "backend": details.simulation_backend,
        "cycle": target.hpr_cycle,
        "selected_load": None if load is None else float(load.selected),
        "achieved_load": None if load is None else float(load.achieved),
        "objective": float(details.obj),
        "period_ids": [] if details.period_ids is None else list(details.period_ids),
        "period_weights": _floats(details.period_weights),
        "design_vector": _floats(details.design_vector),
        "loop_count": 0 if record is None else len(record.loops),
    }


def hpr_failure_summary(error: HPRTargetingError) -> dict[str, Any]:
    """Return an HPR targeting failure and its bounded diagnostics as plain JSON."""
    return {
        "status": "typed infeasible",
        "reason": str(error),
        "diagnostics": error.diagnostics.model_dump(mode="json"),
    }


__all__ = ["hpr_failure_summary", "hpr_summary"]
