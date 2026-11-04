# Scalar Optimisation

## Overview
Reusable bounded scalar minimisation that returns several ranked local minima rather than one point. `service.py` validates a problem and options, calls one of four backends in [backends](./backends/README.md), then keeps only finite, in-bounds, constraint-feasible candidates. The package imports nothing from other OpenPinch packages; `OpenPinch/analysis/utility_placement`, `OpenPinch/analysis/heat_pumps` and `OpenPinch/analysis/orc` call it.

## Interface
- `run_multistart_minimisation` (`service.py`): validate, run the chosen backend, filter and return an `OptimisationResult`.
- `OptimisationProblem`, `OptimisationOptions`, `OptimisationCandidate`, `OptimisationResult`, `OptimisationMethod`, `ScalarObjective`, `LOCAL_METHODS` (`models.py`): frozen inputs and outputs; `OptimisationOptions.from_mapping` splits generic keys from backend options.
- `OptimisationError`, `InvalidOptimisationProblemError`, `InvalidObjectiveValueError`, `NoOptimisationCandidatesError` (`errors.py`): failure types.
- `is_feasible`, `validate_constraints`, `constraint_items`, `DEFAULT_FEASIBILITY_TOLERANCE` (`feasibility.py`): bounds and scipy-style constraint checks.
- `TaskScopedObjective`, `TaskPhase` (`execution.py`): protocol for objectives that hold per-task state.
- `_collect_candidates_in_parallel` (`execution.py`): run backend runs in worker processes when a batch is slow enough to justify them, serially otherwise.
- `_cluster_candidates`, `_polish_candidates`, `_postprocess_candidates`, `_evaluate_scalar_objective` (`candidates.py`): shared ranking, clustering and local polishing, listed in `__all__` for the backends.
- Subdirectories: [backends](./backends/README.md) - the four search backends and their multistart plumbing.

## Invariants
- Bounds must be a non-empty `(lower, upper)` list that is finite with `lower <= upper`; initial points must match the dimension and lie inside the bounds [service.py:_validate_problem].
- Backend options outside the common set plus the method's own set are rejected before any solve [service.py:_validate_options].
- `local_method` must be `None` or one of `LOCAL_METHODS`, all of which honour bounds [models.py:LOCAL_METHODS] [service.py:_validate_options].
- Integer options reject booleans, text and non-integral numbers [service.py:_integer_option].
- Constraints are validated before the solve [service.py:run_multistart_minimisation] [feasibility.py:validate_constraints].
- The backend is asked for `4 * max_minima` basins, because infeasible and non-finite ones are dropped before the list is cut to `max_minima` [service.py:run_multistart_minimisation].
- Candidates are ordered by objective, then coordinates; non-finite ones are skipped, and all-non-finite output raises `InvalidObjectiveValueError`, and `NoOptimisationCandidatesError` is raised when no finite, in-bounds, constraint-feasible candidate remains [models.py:OptimisationCandidate] [service.py:_normalise_candidates] [service.py:run_multistart_minimisation].
- `feasibility_tolerance` is not part of `to_backend_kwargs`; the service passes it separately [models.py:OptimisationOptions] [service.py:run_multistart_minimisation].
- A polish is kept only if it succeeds, is finite, and is not less feasible or (when equally feasible) worse than its start; only numerical solver errors are swallowed [candidates.py:_polish_single_candidate].
- Clustering normalises by the bound span and treats a zero span as one [candidates.py:_cluster_candidates].
- A `TaskScopedObjective` gets `for_task` calls in the parent in task order, runs each copy serially or in a worker, and receives every `task_report()` through `merge_task_reports` in task order [execution.py:_run_objective_tasks].
- An exception raised by the objective propagates unchanged, whether it ran in a worker or this process [execution.py:_map_in_process_pool] [execution.py:_map_with_late_pool].
- Under spawn and forkserver a batch runs in this process until `_POOL_PATIENCE_SECONDS` (1 s) pass, because each worker imports OpenPinch before its first task, which costs seconds. Only then does a pool of `workers - 1` start, this process being the other worker, and a worker gets a task only after it has finished importing, so a batch is never held up by importing workers [execution.py:_map_with_late_pool].
- Under fork, workers start in milliseconds, so a batch of two or more tasks starts its pool at once [execution.py:_map_in_process_pool].
- Results keep task order and do not depend on where a task ran: each task is independent, and workers get tasks pickled when the batch began, before this process changes any state they share [execution.py:_map_with_late_pool].

## Side Effects
- Spawns `ProcessPoolExecutor` workers; falls back to serial execution on Emscripten, an unpicklable task, a pool that cannot start or breaks, or a `__main__` script with no main guard [execution.py:_pool_worker_count] [execution.py:_map_in_process_pool] [execution.py:_map_with_late_pool].
- Under spawn and forkserver, starts a timer thread per batch, which outlives the batch if a pool it started is not up yet; that thread starts, feeds and stops the pool, while objectives still run on the calling thread [execution.py:_map_with_late_pool].
- A batch can return while its pool's workers are still importing; they then exit by themselves, so interpreter exit can wait for them for up to the import time [execution.py:_map_with_late_pool].
- Reads and parses the main script file to look for a main guard when the start method is not fork [execution.py:_main_module_reruns_in_workers].
- Logs warnings and debug messages through `logging` [execution.py:_map_in_process_pool] [candidates.py:_polish_single_candidate].
- Calls `scipy.optimize.minimize` for polishing [candidates.py:_polish_single_candidate].
- `merge_task_reports` mutates the caller's objective [execution.py:_run_objective_tasks].

## Verification
- Run: `uv run --no-sync pytest tests/optimisation`
