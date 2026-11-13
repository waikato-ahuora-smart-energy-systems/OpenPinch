# HEN Execution

## Overview
Resolves synthesis settings, and the options the search width and depth set, from a prepared problem, and executes batches of synthesis tasks, either with the local solver executor or a deterministic fake. The `targeting` package builds tasks and calls the executors here; `results` and `reporting` consume the outcomes. Solver models come from `../models` and solver arrays from `../solver`.

## Interface
- `SynthesisWorkflowSettings` (`settings.py`): frozen dataclass of resolved HENS controls, with derived properties (`runs_evolution`, `runs_thermal_derivative_method`, `milp_networks_seed_evolution`, EVM branch breadth), `heuristic_pdm_networks` (the heuristic network cap per multiplier, 0 for none), `runs_milp_pinch_design` (the MILP at every multiplier: `pdm_milp`, or no heuristic networks) and `solver_for` / `solver_options_for` per method; for the heuristic pinch design method `solver_for` gives HiGHS when it is installed (`heuristic_solver`), otherwise the PDM solver, and `solver_options_for` gives the PDM options only when that is the PDM solver [settings.py:heuristic_solver] [executor.py:_solver_for_task].
- `workflow_settings_from_problem` (`settings.py`): reads `master_zone.config.hens` into settings; `dt_cont_multipliers` is `HENS_DT_CONT_MULTIPLIERS`, or the search width's multipliers when it is unset, and the depth-set options resolve the same way [search.py:resolve_search].
- `StagePackingScope` (`settings.py`): alias for the stage-packing scope string.
- `WIDTH_MULTIPLIERS`, `DEPTH_LEVELS`, `DepthLevel`, `ResolvedSearch`, `resolve_search`, `reference_dt_cont_multiplier` (`search.py`): the search levels, the options they set after explicit overrides, and the multiplier a run falls back to.
- `SynthesisExecutor` (`executor.py`): protocol with `execute(tasks, *, problem, parent_outcomes, max_parallel)`.
- `LocalSynthesisExecutor` (`executor.py`): builds an internal model per task, solves it (optionally in a process pool), extracts the network and returns outcomes.
- `FakeSynthesisExecutor` (`fake_executor.py`): deterministic no-solver executor for workflow tests.
- `has_recovery_topology`, `is_utility_only_outcome`, `is_pdm_variant_outcome`, `required_topology_restrictions_from_outcome`, `topology_restrictions_from_network`, `stage_count_from_network`, `dt_cont_multiplier_from_network`, `derivative_threshold_from_network` (`task_builders.py`): helpers that derive downstream task inputs from outcomes or seed networks.
- Couenne fallback predicates (`fallbacks.py`): private (underscore) helpers used by `targeting/open_hens_method.py`.

## Invariants
- A downstream task needs a successful parent outcome and the parent's in-process solver problem; `_parent_problem` raises `WorkflowContractError`, but `LocalSynthesisExecutor.execute` catches it and returns a failed task outcome carrying the message, while `FakeSynthesisExecutor` lets it propagate [executor.py:LocalSynthesisExecutor._parent_problem] [executor.py:LocalSynthesisExecutor.execute] [fake_executor.py:FakeSynthesisExecutor.execute].
- Tasks are built inside pool workers only when no parent outcomes exist, no task has a parent and more than one worker is in use [executor.py:_can_build_tasks_in_workers].
- A task's internal approach is `2 * THERMAL_DT_CONT * dt_cont_multiplier` (`solver.arrays.nominal_approach_temperature`). PDM models use it; TDM and ESM models use `max(0.1 K, HENS_MIN_EXCHANGER_APPROACH_SHARE x it)` (0.1 K at the default share of 0) [executor.py:_task_model_approach_temperature, LocalSynthesisExecutor._build_problem].
- Extracted networks record the task's dt_cont multiplier as `dt_cont_multiplier` in summary metrics and source metadata; a seeded task takes it from there, or the reference multiplier (1.0 when configured, otherwise the first) for a network without one [executor.py:LocalSynthesisExecutor._build_problem] [task_builders.py:dt_cont_multiplier_from_network].
- A solve counts as successful only when `mSuccess == 1` and `verify()` (if present) passes; failures carry a non-empty error string [executor.py:_solve_built_task, _solved_outcome, _failed_task_outcome].
- A PDM task is built with `pdm_network_limits` from `HENS_PDM_MIN_UNIT_NETWORKS` and `HENS_PDM_EXTRA_UNIT_NETWORKS`. Each extra network its problem returns in `pdm_variants` becomes its own task (the PDM task with `pdm_variant: k` added to its settings, so a new task id), outcome and registered problem, right after the PDM task's outcome; the variants are then dropped from the PDM task's problem [executor.py:LocalSynthesisExecutor._pdm_variant_outcomes, _pdm_variant_task].
- A heuristic pinch design task builds a PDM problem (pinch decompositions, HiGHS when installed and otherwise the PDM solver, with the PDM options only for the PDM solver, automated stages, no minimum-unit variants) with `heuristic_network_limit` from its `heuristic_networks` setting (10 if absent); its extra networks fan out as above with `heuristic_variant: k`. Heuristic networks are labelled `heuristic_pinch_design_method` [executor.py:LocalSynthesisExecutor._build_problem, _variant_key, _model_framework].
- A successful solve whose solver stopped before converging records `diagnostics["unconverged_termination"]`; it stays a success so downstream tasks can start from it [executor.py:_solve_built_task].
- Stage-packing scope `"auto"` resolves to `"none"` at execution time [executor.py:_stage_packing_scope]. The scope decides packing for PDM and TDM tasks [executor.py:LocalSynthesisExecutor._model_factories_for_task].
- Outcomes are returned in the order of the input tasks [executor.py:LocalSynthesisExecutor._collect_outcomes].
- Settings are a frozen dataclass; only `evm_solver_options` is copied on construction, while `pdm_solver_options` and `tdm_solver_options` are shared references [settings.py:SynthesisWorkflowSettings.__post_init__].
- An option the width or depth sets keeps its explicit value when the configuration sets it (`None` means unset): `HENS_DT_CONT_MULTIPLIERS` for the width; `HENS_HEURISTIC_PDM_NETWORKS`, `HENS_PDM_MILP`, `HENS_EVM_N_AD_BRANCHES` and `HENS_EVM_N_RM_BRANCHES` for the depth. The minimum-unit network counts are not set by the depth. The executor resolves evolution breadth the same way [search.py:resolve_search] [executor.py:_evolution_options].
- The Couenne fallback applies only when the stage's configured solver (`pdm_solver` or `tdm_solver`) is `couenne`, the stage has tasks and outcomes, and every outcome failed with a "couenne ... not found on PATH" error [fallbacks.py:_missing_couenne_failure].
- Utility-only networks (no active staged recovery exchanger) are not valid seeds [task_builders.py:has_recovery_topology].
- The fake executor requires successful parents for downstream tasks [fake_executor.py:FakeSynthesisExecutor.execute].

## Side Effects
- Spawns solver worker processes through `ProcessPoolExecutor`, falling back to serial when the pool raises `NotImplementedError`. The default pool raises it, after logging a warning, when the start method is not fork and the `__main__` script has no main guard, so it reads and parses that script [executor.py:_process_pool, LocalSynthesisExecutor._map] [../../../optimisation/execution.py:_main_module_reruns_in_workers].
- Emits progress events [executor.py:LocalSynthesisExecutor._report_progress].
- Mutates its own bookkeeping (`executed_tasks`, `stage_order`, `problems_by_task_id`, `best_objective`) [executor.py:LocalSynthesisExecutor].
- Emits a `RuntimeWarning` when the Couenne fallback is taken [fallbacks.py:_warn_couenne_fallback].
- Imports stage-packed model classes lazily [executor.py:_stage_packed_pdm_factory, _stage_packed_stagewise_factory].

## Verification
- Run: `uv run --no-sync pytest tests/analysis/heat_exchanger_networks/test_support_edge_cases.py tests/analysis/heat_exchanger_networks/test_design_workflow.py`
- There is no execution-specific test directory; these are the nearest focused tests (the full set is `tests/analysis/heat_exchanger_networks`).
