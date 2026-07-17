"""Failures raised by reusable scalar optimisation."""


class OptimisationError(RuntimeError):
    """Base class for optimisation failures."""


class InvalidOptimisationProblemError(OptimisationError, ValueError):
    """The supplied bounds, starts, or options are invalid."""


class InvalidObjectiveValueError(OptimisationError, ValueError):
    """An objective result is not a finite scalar value."""


class NoOptimisationCandidatesError(OptimisationError):
    """A backend completed without producing a usable candidate."""


__all__ = [
    "InvalidObjectiveValueError",
    "InvalidOptimisationProblemError",
    "NoOptimisationCandidatesError",
    "OptimisationError",
]
