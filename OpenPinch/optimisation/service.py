"""Method selection and validation for reusable scalar minimisation."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from .backends.bayesian import _get_bo_multiminima_in_parallel
from .backends.cma_es import _get_cma_multiminima_in_parallel
from .backends.dual_annealing import _get_da_multiminima_in_parallel
from .backends.protocol import OptimisationBackend
from .backends.rbf import _get_rbf_surrogate_multiminima_in_parallel
from .errors import (
    InvalidObjectiveValueError,
    InvalidOptimisationProblemError,
    NoOptimisationCandidatesError,
)
from .models import (
    OptimisationCandidate,
    OptimisationMethod,
    OptimisationOptions,
    OptimisationProblem,
    OptimisationResult,
)

_COMMON_OPTION_NAMES = frozenset(
    {
        "args",
        "constraints",
        "n_runs",
        "maxiter",
        "seed",
        "maxfun",
        "cluster_tol",
        "max_minima",
        "local_method",
    }
)
_METHOD_OPTION_NAMES = {
    OptimisationMethod.DUAL_ANNEALING: frozenset(
        {
            "initial_temp",
            "restart_temp_ratio",
            "visit",
            "accept",
        }
    ),
    OptimisationMethod.CMA_ES: frozenset(
        {"maxfevals", "popsize", "sigma0", "tolx", "tolfun"}
    ),
    OptimisationMethod.BAYESIAN: frozenset(
        {
            "maxfevals",
            "n_init",
            "acq_candidates",
            "lengthscale",
            "noise",
            "xi",
        }
    ),
    OptimisationMethod.RBF: frozenset(
        {
            "maxfevals",
            "n_init",
            "n_candidates",
            "kernel",
            "epsilon",
            "smoothing",
            "degree",
            "distance_tol",
        }
    ),
}


def run_multistart_minimisation(
    problem: OptimisationProblem,
    *,
    method: OptimisationMethod | str = OptimisationMethod.DUAL_ANNEALING,
    options: OptimisationOptions | Mapping[str, object] | None = None,
) -> OptimisationResult:
    """Run one reusable backend and return finite candidates in stable order."""
    method = _normalise_method(method)
    options = _normalise_options(options)
    bounds, initial_points = _validate_problem(problem)
    backend_kwargs = options.to_backend_kwargs()
    _validate_options(method, backend_kwargs)

    backend = _resolve_backend(method)
    points, objectives = backend(
        func=problem.objective,
        bounds=bounds,
        x0_ls=initial_points,
        args=problem.args,
        constraints=problem.constraints,
        **backend_kwargs,
    )
    candidates = _normalise_candidates(points, objectives, len(bounds))
    if not candidates:
        raise NoOptimisationCandidatesError(
            f"{method.value} completed without a finite candidate."
        )
    return OptimisationResult(method=method, candidates=candidates)


def _normalise_method(method: OptimisationMethod | str) -> OptimisationMethod:
    if isinstance(method, OptimisationMethod):
        return method
    try:
        return OptimisationMethod(str(method))
    except ValueError as exc:
        supported = ", ".join(item.value for item in OptimisationMethod)
        raise InvalidOptimisationProblemError(
            f"Unsupported optimisation method {method!r}. "
            f"Supported methods: {supported}."
        ) from exc


def _normalise_options(
    options: OptimisationOptions | Mapping[str, object] | None,
) -> OptimisationOptions:
    if options is None:
        return OptimisationOptions()
    if isinstance(options, OptimisationOptions):
        return options
    if isinstance(options, Mapping):
        return OptimisationOptions.from_mapping(options)
    raise InvalidOptimisationProblemError(
        "options must be OptimisationOptions, a mapping, or None."
    )


def _validate_problem(
    problem: OptimisationProblem,
) -> tuple[np.ndarray, np.ndarray | None]:
    if not callable(problem.objective):
        raise InvalidOptimisationProblemError("objective must be callable.")
    bounds = np.asarray(problem.bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1:] != (2,) or bounds.shape[0] == 0:
        raise InvalidOptimisationProblemError(
            "bounds must contain at least one (lower, upper) pair."
        )
    if not np.isfinite(bounds).all():
        raise InvalidOptimisationProblemError("bounds must be finite.")
    if np.any(bounds[:, 0] > bounds[:, 1]):
        raise InvalidOptimisationProblemError(
            "each lower bound must be less than or equal to its upper bound."
        )

    if not problem.initial_points:
        return bounds, None
    initial_points = np.asarray(problem.initial_points, dtype=float)
    if initial_points.ndim != 2 or initial_points.shape[1] != bounds.shape[0]:
        raise InvalidOptimisationProblemError(
            "every initial point must match the number of bounds."
        )
    if not np.isfinite(initial_points).all():
        raise InvalidOptimisationProblemError("initial points must be finite.")
    if np.any(initial_points < bounds[:, 0]) or np.any(initial_points > bounds[:, 1]):
        raise InvalidOptimisationProblemError(
            "initial points must lie within the supplied bounds."
        )
    return bounds, initial_points


def _validate_options(
    method: OptimisationMethod,
    backend_kwargs: Mapping[str, object],
) -> None:
    allowed = _COMMON_OPTION_NAMES | _METHOD_OPTION_NAMES[method]
    unknown = sorted(set(backend_kwargs) - allowed)
    if unknown:
        names = ", ".join(unknown)
        raise InvalidOptimisationProblemError(
            f"Unsupported {method.value} option(s): {names}."
        )
    if int(backend_kwargs["n_runs"]) < 1:
        raise InvalidOptimisationProblemError("n_runs must be at least one.")
    if int(backend_kwargs["maxiter"]) < 0:
        raise InvalidOptimisationProblemError("maxiter must be non-negative.")
    if int(backend_kwargs["maxfun"]) < 1:
        raise InvalidOptimisationProblemError("maxfun must be positive.")
    if float(backend_kwargs["cluster_tol"]) < 0.0:
        raise InvalidOptimisationProblemError("cluster_tol must be non-negative.")
    max_minima = backend_kwargs["max_minima"]
    if max_minima is not None and int(max_minima) < 1:
        raise InvalidOptimisationProblemError("max_minima must be positive or None.")


def _resolve_backend(method: OptimisationMethod) -> OptimisationBackend:
    if method is OptimisationMethod.DUAL_ANNEALING:
        return _get_da_multiminima_in_parallel
    if method is OptimisationMethod.CMA_ES:
        return _get_cma_multiminima_in_parallel
    if method is OptimisationMethod.BAYESIAN:
        return _get_bo_multiminima_in_parallel
    if method is OptimisationMethod.RBF:
        return _get_rbf_surrogate_multiminima_in_parallel
    raise AssertionError(f"Unhandled optimisation method: {method!r}")


def _normalise_candidates(
    points,
    objectives,
    dimension: int,
) -> tuple[OptimisationCandidate, ...]:
    points = np.asarray(points, dtype=float)
    objectives = np.asarray(objectives, dtype=float)
    if points.size == 0 and objectives.size == 0:
        return ()
    if points.ndim != 2 or points.shape[1] != dimension:
        raise InvalidObjectiveValueError(
            "The backend returned candidate points with an invalid shape."
        )
    if objectives.ndim != 1 or objectives.shape[0] != points.shape[0]:
        raise InvalidObjectiveValueError(
            "The backend returned objectives with an invalid shape."
        )

    candidates: list[OptimisationCandidate] = []
    for point, objective in zip(points, objectives, strict=True):
        objective = float(objective)
        if not math.isfinite(objective) or not np.isfinite(point).all():
            raise InvalidObjectiveValueError(
                "The backend returned a non-finite point or objective."
            )
        candidates.append(
            OptimisationCandidate(
                objective=objective,
                point=tuple(float(value) for value in point),
            )
        )
    return tuple(sorted(candidates))


__all__ = ["run_multistart_minimisation"]
