# Optimisation Backends

## Overview
Four concrete global-search backends for bounded scalar minimisation: dual annealing, CMA-ES, Bayesian optimisation and an RBF surrogate. Each runs several independent searches, pools the incumbents they record, then clusters and polishes them through `OpenPinch/optimisation/candidates.py` and `OpenPinch/optimisation/execution.py`. `OpenPinch/optimisation/service.py` selects and calls these backends; nothing else imports them directly.

## Interface
- `OptimisationBackend` (`protocol.py`): keyword-only callable protocol `(func, bounds, x0_ls, args, constraints, **options)` returning `(points, objectives)` arrays.
- `_get_da_multiminima_in_parallel` (`dual_annealing.py`): multistart `scipy.optimize.dual_annealing` backend, one entry point per method.
- `_get_cma_multiminima_in_parallel` (`cma_es.py`): pure-Python rank-mu CMA-ES backend.
- `_get_bo_multiminima_in_parallel` (`bayesian.py`): Gaussian-process expected-improvement backend.
- `_get_rbf_surrogate_multiminima_in_parallel` (`rbf.py`): `RBFInterpolator` merit-function backend.
- `run_multistart`, `collect_candidates`, `as_seed_array` (`_multistart.py`): shared plumbing; `run_multistart` runs a per-run function `n_runs` times, then clusters and polishes.
- `run_surrogate_loop`, `candidate_pool` (`_surrogate.py`): unit-box loop and candidate sampler shared by the BO and RBF backends.
- Subdirectories: none.

## Invariants
- A backend is called with keyword arguments only and returns `(points, objectives)` as arrays [protocol.py:OptimisationBackend].
- A per-run function takes the run index first and returns `(list of points, list of objectives)` [_multistart.py:RunSingle].
- Run `k` starts from row `k % len(x0_ls)` of the seed starts, and its random generator is seeded with `seed + k` [_surrogate.py:run_surrogate_loop] [cma_es.py:_run_cma_single] [dual_annealing.py:_run_da_single].
- Constraints are not used during the global search; they are passed only to the polish step [_multistart.py:run_multistart].
- A coordinate with equal lower and upper bounds stays at that value. Dual annealing anneals only the free coordinates. The surrogate and CMA-ES paths return the fixed point directly only when every coordinate is fixed; otherwise they still sample all dimensions, and fixed ones are held by clipping to their bounds [dual_annealing.py:_run_da_single] [_surrogate.py:run_surrogate_loop] [cma_es.py:_run_cma_single].
- BO, RBF and CMA-ES take their evaluation cap from `maxfevals`, falling back to `maxfun`; dual annealing passes `maxfun` straight to scipy [bayesian.py:_get_bo_multiminima_in_parallel] [rbf.py:_get_rbf_surrogate_multiminima_in_parallel] [cma_es.py:_get_cma_multiminima_in_parallel] [dual_annealing.py:_get_da_multiminima_in_parallel].
- Surrogate backends search in the unit box and map back with clipping to the original bounds [_surrogate.py:run_surrogate_loop].
- Non-finite objective values are excluded when fitting the BO and RBF models [bayesian.py:_run_bo_single] [rbf.py:_fit_rbf_surrogate_model].
- CMA-ES uses correctly rounded dot products and its own Jacobi eigendecomposition so the same inputs give the same samples on every platform; do not replace them with `numpy.linalg` calls [cma_es.py:_exact_dot] [cma_es.py:_symmetric_eigh].

## Side Effects
- Search runs execute in a `ProcessPoolExecutor` when the objective is picklable and more than one worker is useful; otherwise they run serially [_multistart.py:collect_candidates].
- Dual annealing calls `scipy.optimize.dual_annealing` with local search disabled [dual_annealing.py:_run_da_single].
- The polish step calls `scipy.optimize.minimize` and logs warnings (see the parent README) [_multistart.py:run_multistart].
- No file or network IO.

## Verification
- Run: `uv run --no-sync pytest tests/optimisation/test_backends.py`
