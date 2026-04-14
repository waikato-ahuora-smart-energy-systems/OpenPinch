"""Black-box optimisation entrypoint with lightweight backend dispatch."""

from typing import Callable, Optional

from ..lib.enums import BB_Minimiser
from .bb_optimisers import (
    _filter_opt_kwargs,
    _get_bo_multiminima_in_parallel,
    _get_cma_multiminima_in_parallel,
    _get_da_multiminima_in_parallel,
    _get_rbf_surrogate_multiminima_in_parallel,
    _set_objective_func,
)

_DA_ALLOWED_KEYS = frozenset(
    {
        "args",
        "constraints",
        "n_runs",
        "maxiter",
        "seed",
        "initial_temp",
        "restart_temp_ratio",
        "visit",
        "accept",
        "maxfun",
        "cluster_tol",
        "max_minima",
    }
)
_CMA_ALLOWED_KEYS = frozenset(
    {
        "args",
        "constraints",
        "n_runs",
        "maxiter",
        "seed",
        "maxfun",
        "maxfevals",
        "popsize",
        "sigma0",
        "cluster_tol",
        "max_minima",
        "local_method",
        "tolx",
        "tolfun",
    }
)
_BO_ALLOWED_KEYS = frozenset(
    {
        "args",
        "constraints",
        "n_runs",
        "maxiter",
        "seed",
        "maxfun",
        "maxfevals",
        "cluster_tol",
        "max_minima",
        "local_method",
        "n_init",
        "acq_candidates",
        "lengthscale",
        "noise",
        "xi",
    }
)
_RBF_ALLOWED_KEYS = frozenset(
    {
        "args",
        "constraints",
        "n_runs",
        "maxiter",
        "seed",
        "maxfun",
        "maxfevals",
        "cluster_tol",
        "max_minima",
        "local_method",
        "n_init",
        "n_candidates",
        "kernel",
        "epsilon",
        "smoothing",
        "degree",
        "distance_tol",
    }
)

__all__ = [
    "multiminima",
]

_DEFAULT_MULTIMINIMA_HANDLE = BB_Minimiser.DA
_MULTIMINIMA_ALLOWED_KEYS = {
    BB_Minimiser.DA: _DA_ALLOWED_KEYS,
    BB_Minimiser.CMAES: _CMA_ALLOWED_KEYS,
    BB_Minimiser.BO: _BO_ALLOWED_KEYS,
    BB_Minimiser.RBF: _RBF_ALLOWED_KEYS,
}
_MULTIMINIMA_HANDLE_ALIASES = {method.value: method for method in BB_Minimiser}
_MULTIMINIMA_HANDLE_ALIASES.update(
    {
        "da": BB_Minimiser.DA,
        "dualannealing": BB_Minimiser.DA,
        "dual-annealing": BB_Minimiser.DA,
        "dual_annealing_multiminima": BB_Minimiser.DA,
        "cma": BB_Minimiser.CMAES,
        "cma-es": BB_Minimiser.CMAES,
        "cma_es": BB_Minimiser.CMAES,
        "cmaes_multiminima": BB_Minimiser.CMAES,
        "bayes": BB_Minimiser.BO,
        "bayesian_optimization": BB_Minimiser.BO,
        "bayesian_optimisation": BB_Minimiser.BO,
        "bo_multiminima": BB_Minimiser.BO,
        "rbf": BB_Minimiser.RBF,
        "rbf-surrogate": BB_Minimiser.RBF,
        "rbf_surrogate_multiminima": BB_Minimiser.RBF,
    }
)


def _get_supported_multiminima_handles() -> tuple[BB_Minimiser, ...]:
    """Return canonical optimizer handles supported by :func:`multiminima`."""
    return tuple(_MULTIMINIMA_ALLOWED_KEYS.keys())


def _normalise_multiminima_handle(
    handle: Optional[str | BB_Minimiser],
) -> BB_Minimiser:
    """Resolve a canonical optimizer handle from a canonical name or alias."""
    if handle is None:
        return _DEFAULT_MULTIMINIMA_HANDLE
    if isinstance(handle, BB_Minimiser):
        return handle
    if not isinstance(handle, str):
        raise TypeError(
            "Optimizer handle must be a string or BB_Minimiser; "
            f"got {type(handle).__name__}."
        )

    normalised = handle.strip().lower().replace(" ", "_")
    canonical = _MULTIMINIMA_HANDLE_ALIASES.get(normalised)
    if canonical is None:
        supported = ", ".join(
            method.value for method in _get_supported_multiminima_handles()
        )
        raise ValueError(
            f"Unsupported optimiser handle {handle!r}. Supported handles: {supported}."
        )
    return canonical


def _get_multiminima_handler(
    handle: Optional[str | BB_Minimiser] = _DEFAULT_MULTIMINIMA_HANDLE,
) -> Callable:
    """Return the backend runner corresponding to ``handle``."""
    canonical_handle = _normalise_multiminima_handle(handle)
    if canonical_handle is BB_Minimiser.DA:
        return _get_da_multiminima_in_parallel
    if canonical_handle is BB_Minimiser.CMAES:
        return _get_cma_multiminima_in_parallel
    if canonical_handle is BB_Minimiser.BO:
        return _get_bo_multiminima_in_parallel
    if canonical_handle is BB_Minimiser.RBF:
        return _get_rbf_surrogate_multiminima_in_parallel

    supported = ", ".join(
        method.value for method in _get_supported_multiminima_handles()
    )
    raise ValueError(
        f"Unsupported optimiser handle {canonical_handle!r}. Supported handles: {supported}."
    )


def multiminima(
    func: Callable,
    x0_ls: Optional[tuple] = None,
    func_kwargs: Optional[dict] = {},
    bounds: Optional[tuple] = None,
    opt_kwargs: Optional[dict] = {},
    optimiser_handle: Optional[str | BB_Minimiser] = _DEFAULT_MULTIMINIMA_HANDLE,
) -> list:
    """Run a selected multi-start optimizer by handle."""
    func_kwargs = {} if func_kwargs is None else func_kwargs
    opt_kwargs = {} if opt_kwargs is None else opt_kwargs
    objective = _set_objective_func(func=func, func_kwargs=func_kwargs)
    canonical_handle = _normalise_multiminima_handle(optimiser_handle)
    handler = _get_multiminima_handler(handle=canonical_handle)
    allowed_keys = _MULTIMINIMA_ALLOWED_KEYS[canonical_handle]
    backend_kwargs = _filter_opt_kwargs(opt_kwargs, allowed_keys)
    return handler(
        func=objective,
        x0_ls=x0_ls,
        bounds=bounds,
        **backend_kwargs,
    )
