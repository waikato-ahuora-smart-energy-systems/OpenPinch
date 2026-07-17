"""Immutable inputs and outputs for reusable scalar optimisation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Mapping

ScalarObjective = Callable[..., float]


class OptimisationMethod(StrEnum):
    """Supported black-box minimisation methods."""

    DUAL_ANNEALING = "dual_annealing"
    CMA_ES = "cmaes"
    BAYESIAN = "bo"
    RBF = "rbf_surrogate"


@dataclass(frozen=True, slots=True)
class OptimisationProblem:
    """A bounded scalar minimisation problem."""

    objective: ScalarObjective
    bounds: tuple[tuple[float, float], ...]
    initial_points: tuple[tuple[float, ...], ...] = ()
    args: tuple[Any, ...] = ()
    constraints: Any = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "bounds",
            tuple((float(lower), float(upper)) for lower, upper in self.bounds),
        )
        object.__setattr__(
            self,
            "initial_points",
            tuple(
                tuple(float(value) for value in point) for point in self.initial_points
            ),
        )
        object.__setattr__(self, "args", tuple(self.args))


@dataclass(frozen=True, slots=True)
class OptimisationOptions:
    """Backend-independent execution options plus explicit backend overrides."""

    n_runs: int = 1
    maxiter: int = 300
    seed: int = 0
    maxfun: int = 1_000_000
    cluster_tol: float = 0.01
    max_minima: int | None = 4
    local_method: str = "SLSQP"
    backend_options: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any] | None = None,
    ) -> OptimisationOptions:
        """Build options from the conventional keyword mapping form."""
        if values is None:
            return cls()
        values = dict(values)
        generic_names = {
            "n_runs",
            "maxiter",
            "seed",
            "maxfun",
            "cluster_tol",
            "max_minima",
            "local_method",
        }
        generic = {
            name: values.pop(name) for name in tuple(values) if name in generic_names
        }
        return cls(
            **generic,
            backend_options=tuple(sorted(values.items())),
        )

    def to_backend_kwargs(self) -> dict[str, Any]:
        """Return a detached mapping suitable for one backend invocation."""
        return {
            "n_runs": self.n_runs,
            "maxiter": self.maxiter,
            "seed": self.seed,
            "maxfun": self.maxfun,
            "cluster_tol": self.cluster_tol,
            "max_minima": self.max_minima,
            "local_method": self.local_method,
            **dict(self.backend_options),
        }


@dataclass(frozen=True, slots=True, order=True)
class OptimisationCandidate:
    """One finite candidate ordered by objective and then coordinates."""

    objective: float
    point: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class OptimisationResult:
    """Deterministically ordered candidates returned by one method."""

    method: OptimisationMethod
    candidates: tuple[OptimisationCandidate, ...]

    @property
    def best(self) -> OptimisationCandidate:
        """Return the lowest-objective candidate."""
        if not self.candidates:
            raise LookupError("The optimisation result contains no candidates.")
        return self.candidates[0]


__all__ = [
    "OptimisationCandidate",
    "OptimisationMethod",
    "OptimisationOptions",
    "OptimisationProblem",
    "OptimisationResult",
    "ScalarObjective",
]
