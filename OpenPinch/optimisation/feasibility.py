"""Feasibility of one point against bounds and scipy-style constraints."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .errors import InvalidOptimisationProblemError

# Default slack for bounds and constraints, matching
# ``OptimisationOptions.feasibility_tolerance``.
DEFAULT_FEASIBILITY_TOLERANCE = 1.0e-6


def constraint_items(constraints) -> tuple:
    """Return the supplied constraints as a tuple of single constraints."""
    if constraints is None:
        return ()
    if isinstance(constraints, Mapping) or hasattr(constraints, "lb"):
        return (constraints,)
    if isinstance(constraints, (tuple, list)):
        return tuple(constraints)
    raise InvalidOptimisationProblemError(
        "constraints must be a scipy-compatible constraint, a mapping, "
        "or a sequence of them."
    )


def validate_constraints(constraints) -> None:
    """Raise when a constraint is not scipy-compatible, before any solve."""
    for constraint in constraint_items(constraints):
        if isinstance(constraint, Mapping):
            constraint_type = str(constraint.get("type", "")).lower()
            if constraint_type not in {"eq", "ineq"} or not callable(
                constraint.get("fun")
            ):
                raise InvalidOptimisationProblemError(
                    "constraints must use scipy-compatible 'eq' or 'ineq' mappings."
                )
            continue
        if getattr(constraint, "lb", None) is None or (
            getattr(constraint, "ub", None) is None
        ):
            raise InvalidOptimisationProblemError(
                "constraints must be scipy-compatible constraint objects or mappings."
            )
        if not hasattr(constraint, "A") and not callable(
            getattr(constraint, "fun", None)
        ):
            raise InvalidOptimisationProblemError(
                "constraint objects must provide a callable fun or a matrix A."
            )


def is_feasible(
    point: np.ndarray,
    *,
    bounds: np.ndarray | None,
    constraints,
    tolerance: float = DEFAULT_FEASIBILITY_TOLERANCE,
) -> bool:
    """Return whether one finite point satisfies bounds and constraints."""
    point = np.asarray(point, dtype=float)
    if not np.isfinite(point).all():
        return False
    if bounds is not None and (
        np.any(point < bounds[:, 0] - tolerance)
        or np.any(point > bounds[:, 1] + tolerance)
    ):
        return False
    validate_constraints(constraints)
    for constraint in constraint_items(constraints):
        if isinstance(constraint, Mapping):
            constraint_type = str(constraint.get("type", "")).lower()
            values = np.asarray(
                constraint["fun"](point, *tuple(constraint.get("args", ()))),
                dtype=float,
            )
            if not np.isfinite(values).all():
                return False
            if constraint_type == "eq" and np.any(np.abs(values) > tolerance):
                return False
            if constraint_type == "ineq" and np.any(values < -tolerance):
                return False
            continue

        if hasattr(constraint, "A"):
            values = np.asarray(constraint.A @ point, dtype=float)
        else:
            values = np.asarray(constraint.fun(point), dtype=float)
        if not np.isfinite(values).all():
            return False
        if np.any(
            values < np.asarray(constraint.lb, dtype=float) - tolerance
        ) or np.any(values > np.asarray(constraint.ub, dtype=float) + tolerance):
            return False
    return True


__all__ = [
    "DEFAULT_FEASIBILITY_TOLERANCE",
    "constraint_items",
    "is_feasible",
    "validate_constraints",
]
