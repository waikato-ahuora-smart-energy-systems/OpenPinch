# Heat Pump and Refrigeration Analysis

## Overview
Targeting of heat pumps and refrigeration (HPR) against a zone's heat cascade,
and process component records for mechanical vapour recompression (MVR). The
service module selects a load, builds the targeting inputs, dispatches to a
topology-specific placement function in `targeting/`, and merges the result
into problem tables, graphs and a target model. `optimisation_adapter.py` is the
bridge to the solver-neutral `OpenPinch/optimisation` package. Cycle physics
lives in `cycles/` and shared calculations in `common/`. `__init__.py` has only
a docstring.

## Interface
- `compute_direct_heat_pump_or_refrigeration_target`, `compute_indirect_heat_pump_or_refrigeration_target` (`service.py`): solve a direct or utility-system target for a `Zone`; return a target model, or `None` when the selected load is below `tol` or the selected period has no HPR load.
- `solve_hpr_placement`, `run_hpr_candidate_search`, `evaluate_hpr_candidate`, `raise_hpr_targeting_error` (`optimisation_adapter.py`): multistart search over a bounded HPR objective, candidate evaluation and detached `HPRTargetingError` reporting.
- `translate_hpr_result`, `translate_hpr_output` (`optimisation_adapter.py`): attach ambient streams to a backend result and validate it as the public `HeatPumpTargetOutputs`; `translate_hpr_result` marks the result successful, so call it only on an accepted candidate.
- `aggregate_hpr_period_results` (`optimisation_adapter.py`): combine period results for a shared multiperiod design (weighted operation, peak capital).
- `build_hpr_accounting`, `calc_hpr_obj`, `relative_feasibility_penalty`, `normalise_hpr_penalty_terms`, `normalise_initial_points`, `initialise_hpr_seed` (`optimisation_adapter.py`): objective, penalty and seed helpers (`initialise_hpr_seed` is public but not in `__all__`).
- `ProcessComponent`, `ProcessComponentHost` (`components.py`): component base record and the protocol it needs from the problem object.
- `ProcessMVRComponent`, `create_process_mvr_component`, `replay_process_components`, `restore_process_components` (`process_mvr.py`): direct process MVR component with replacement streams, its factory, its re-solve on a rebuilt zone tree, and its rebuild from saved records.
- `screen_refrigerants`, `screen_refrigerant`, `resolve_refrigerants`, `RefrigerantScreenThresholds`, `RefrigerantScreenEntry`, `NATURAL_REFRIGERANTS` (`refrigerant_screen.py`): screen candidate fluids as single-stage cycles between evaporating and condensing saturation levels, and resolve a candidate group (`all` CoolProp fluids, `natural`) or a list of names.
- Subdirectories:
  - [_multiperiod](./_multiperiod/README.md): shared-design preparation, execution and aggregation across periods (documented elsewhere).
  - [_process_mvr](./_process_mvr/README.md): private helpers for process MVR selection, membership, replacement streams and work.
  - [common](./common/README.md): shared preprocessing, costing, encoding and result assembly.
  - [cycles](./cycles/README.md): thermodynamic cycle models.
  - [direct_mvr](./direct_mvr/README.md): direct gas MVR models, thermodynamics and execution.
  - [performance_maps](./performance_maps/README.md): engine-neutral performance-map generation and TESPy/CoolProp targeting adapters.
  - [targeting](./targeting/README.md): per-topology placement optimisation.

## Invariants
- The service dispatches on `config.hpr.type` through a fixed handler table; an unknown type raises `ValueError`, and the Brayton entry's handler raises `NotImplementedError`. [service.py:_HP_PLACEMENT_HANDLERS] [service.py:_get_hpr_targets]
- Targeting runs against the zone's `DI` (direct) or `II` (indirect) target; the indirect case rebuilds the cascade from inverted utility streams. [service.py:_compute_single_period_hpr_target]
- Hot enthalpy is passed to input construction as non-positive and cold as non-negative. [service.py:_get_hpr_targets]
- The TESPy backend is preflighted before its handler runs; handlers for other backends are called without it. [service.py:_get_hpr_targets]
- Multiperiod optimisation is chosen from `config.hpr.multiperiod_optimization_enabled`; the multiperiod path returns `None` when the selected period has no case. [service.py:_use_multiperiod_hpr_optimization] [service.py:_compute_multiperiod_heat_pump_or_refrigeration_target]
- Problem-table columns from a shared grid are aligned by nearest temperature and must match within `tol`. [service.py:_align_problem_table_column]
- The default search budget is 300 iterations and 1,000,000 evaluations, overridable through `maximum_iterations` and `maximum_evaluations` in the call arguments. [service.py:_hpr_search_budget]
- The evaluation budget counts distinct points evaluated in the search, restarts and polish (cached repeats are free), and polishing stops once the remaining budget cannot cover one finite-difference gradient; it does not count the FINAL-mode re-evaluation of ranked candidates, which can take the total above `maximum_evaluations`: restarts run serially (`n_runs=1`), polishing is done in-process, and an exhausted budget stops the search without raising a physical failure. [optimisation_adapter.py:run_hpr_candidate_search] [optimisation_adapter.py:_HPRSearchBudgetExhausted]
- Candidate coordinates must be finite; identical points are evaluated once; failed candidates score `1e30`. [optimisation_adapter.py:_CachedHPRScalarObjective] [optimisation_adapter.py:_FAILED_CANDIDATE_OBJECTIVE]
- Objectives must return `HPRBackendResult`, and `SEARCH` mode strips artifacts and simulation records. [optimisation_adapter.py:evaluate_hpr_candidate]
- Candidates are returned best first and the first successful one is translated; a zero-useful-duty result ends the ranking and is reported as "no beneficial heat pump". [optimisation_adapter.py:solve_hpr_placement] [optimisation_adapter.py:_merge_ranked_candidates]
- Bounds must be finite and ordered, and warm starts must lie inside them; otherwise `ValueError`. [optimisation_adapter.py:run_hpr_candidate_search]
- Penalty terms must be numeric and finite and may not be boolean. [optimisation_adapter.py:normalise_hpr_penalty_terms]
- Public output must be copyable, acyclic and made of numeric arrays, `Value`, `StreamCollection`, known models, string-keyed dicts, lists and tuples; anything else raises `TypeError`. [optimisation_adapter.py:_validate_hpr_public_value]
- Period aggregation weights operation (duties, work, operating cost, penalty) with the schedule share of loaded periods, leaves ratios (`cop_h`, `eta_he`) undiluted, and sizes capital per machine for its own peak period when every period reports the same per-machine breakdown, otherwise taking the peak of the period totals. [optimisation_adapter.py:aggregate_hpr_period_results] [optimisation_adapter.py:_peak_machine_capital]
- Carnot candidates minimise the dimensionless ratio `(work + heat*ratio + cold*ratio + cooling water*ratio + penalty) / Q_hpr_target`. Vapour-compression, VC+MVR and TESPy candidates minimise total annualised cost in $/y plus a $/y feasibility penalty. [optimisation_adapter.py:calc_hpr_obj] [shared.py:evaluate_vapour_hpr_result]
- Process component ids are unique within a problem; `mvr_stage_t_lift` and `mvr_stage_pressure_ratio` are mutually exclusive; a conflicting `period_id` raises `ValueError`. [process_mvr.py:_resolve_component_id] [process_mvr.py:create_process_mvr_component]
- Activating an MVR component deactivates the original source streams and activates the replacement streams, and deactivating reverses this, except that a source another active component replaced stays off; both invalidate the problem's analysis. [process_mvr.py:ProcessMVRComponent.activate] [process_mvr.py:ProcessMVRComponent.deactivate] [components.py:ProcessComponent._invalidate_problem_targets]
- The analysis layer uses `ProcessComponentHost` so it does not import the application layer. [components.py:ProcessComponentHost]
- Adding, replaying or restoring a process component raises `DirectGasMVRPhaseError` if a stage would cross into liquid; see `direct_mvr/README.md`. Nothing is registered on add, and a replay failure changes no component, as below. A saved case with such a component no longer loads. [process_mvr.py:create_process_mvr_component] [process_mvr.py:replay_process_components]
- Replay keeps the component objects and each stream's `is_active` flag; efficiencies not passed explicitly and `dt_diff_max` are re-read from the new configuration; a failure changes no component. [process_mvr.py:replay_process_components]
- A saved record names each source by its owning zone's path below the root and its stream name, so a name selector that matched several streams saves all of them and a renamed project still resolves; the `options` and `period_id` of the original call only check the call and are not saved. A missing source raises `ValueError`. [process_mvr.py:ProcessMVRComponent._to_record] [process_mvr.py:restore_process_components]
- `original_streams`, `replacement_streams` and `stage_results_by_period` return deep copies. [process_mvr.py:ProcessMVRComponent]

- A refrigerant screen judges each fluid in this order: condensing within `dt_cont` of the critical point is above critical; evaporating below CoolProp's minimum temperature is below the triple point; then, from saturation properties, a pressure ratio above `max_pressure_ratio`, a suction pressure below `min_suction_pressure` and condensing within `critical_margin` of the critical point; otherwise practical. Fluids without a saturation state carry no pressures or capacity. [refrigerant_screen.py:screen_refrigerant]
- A screen covers every evaporating and condensing level pair with positive lift, in level order, and raises `ValueError` when none has; within a pair the practical fluids come first, ranked from 1 by descending suction volumetric heating capacity, then the others in candidate order. [refrigerant_screen.py:screen_refrigerants]
- Candidate lists are validated by name and mixtures are rejected; `all` is whatever the installed CoolProp lists, so it differs between CoolProp versions. [refrigerant_screen.py:resolve_refrigerants]

## Side Effects
- Deep-copies the base target's problem table and mutates the copy with HPR, air and residual columns. [service.py:_compute_single_period_hpr_target] [service.py:_calc_hpr_cascade]
- Emits progress events for `hpr_targeting` through `domain.progress`. [optimisation_adapter.py:solve_hpr_placement] [optimisation_adapter.py:_CachedHPRScalarObjective]
- Runs SciPy `minimize` (SLSQP) and the multistart optimiser from `OpenPinch/optimisation`; no worker processes are used by the HPR search. [optimisation_adapter.py:run_hpr_candidate_search]
- CoolProp or TESPy simulations run in the handlers and cycles this module calls; TESPy is checked by `preflight_tespy_hpr_targeting`. [service.py:_get_hpr_targets]
- `create_process_mvr_component` registers the component in `problem._process_components`, adds replacement streams to zone memberships, mutates `is_active` on streams and fills unset `supply_pressure` and `target_pressure` on the source streams. [process_mvr.py:create_process_mvr_component] [process_mvr.py:ProcessMVRComponent.activate]
- `components.py:_clear_zone_targets` is an unused helper; activation and deactivation invalidate the problem through `ProcessComponentHost._invalidate_analysis` only. [components.py:_clear_zone_targets]
- No network IO. Performance-map generation reads packaged data and, with the TESPy backend, writes a private temporary directory removed on close [performance_maps/adapters/tespy.py:TespyHprPointSimulator.prepare].

## Verification
- Run: `uv run --no-sync pytest tests/analysis/heat_pumps/test_targeting.py tests/analysis/heat_pumps/test_optimisation_adapter.py tests/analysis/heat_pumps/test_process_components.py tests/analysis/heat_pumps/test_hpr_search_reliability.py`
- The directory has no single test module for `service.py`; `test_targeting.py` and the `test_hpr_*` modules in `tests/analysis/heat_pumps/` exercise it.
