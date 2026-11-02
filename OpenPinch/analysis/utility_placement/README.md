# Utility Placement

## Overview
A pure model for choosing the supply temperatures and spans of utility levels so that total entropy generation is minimised across weighted periods. A request and template blueprints are normalised, combined with a detached numerical context into a bounded decision vector, searched through `OpenPinch/optimisation`, and the best candidates are replayed canonically. Request and result types live in `OpenPinch/contracts/utility_placement.py`; `OpenPinch/application/utility_placement.py` builds the context from a live problem and imports `profiles.py` and `replay.py` from here.

## Interface
- `optimise_utility_placement` (`service.py`): build the model, optimise, replay and assemble a `UtilityPlacementResult`, converting temperatures back to the request's units.
- `normalize_utility_placement_request`, `prepare_template_blueprints`, `convert_placement_value` (`normalization.py`): request and template normalisation into canonical units.
- `PlacementPeriodInput`, `PlacementTargetSnapshot`, `ProcessEntropySlice`, `UtilityPlacementContext`, `build_utility_placement_context` (`context.py`): frozen per-period numerical input and its validating builder.
- `derive_effective_templates`, `build_initial_values`, `ABSOLUTE_ZERO_C` (`bounds.py`): intersect period and caller bounds, propagate ordering, build the deterministic start.
- `build_utility_placement_model`, `build_decision_coordinates`, `decode_placement`, `decode_verified_placement`, `encode_placement`, `verify_candidate`, `verify_placement` (`codec.py`): decision-vector schema, encode, decode and verification.
- `generate_start_points`, `sensible_profile_start`, `coverage_start`, `interleaved_start`, `spread_start`, `gap_start`, `fractional_starts` (`start_points.py`): start-point heuristics.
- `allocate_placement_period`, `ExistingTargetingAllocationAdapter`, `PlacementAllocationAdapter`, `AllocationAdapterResult`, `PlacementPeriodAllocation`, `AllocatedUtilityLevel`, `UtilityAllocationSlice` (`allocation.py`): per-period duty allocation and coverage checks.
- `PlacementEvaluationSession`, `PlacementEvaluation`, `PlacementTaskReport`, `polish_budget_reserve`, `POLISH_BUDGET_FRACTION` (`evaluation.py`): memoised, budgeted candidate evaluation.
- `evaluate_thermodynamic_cost`, `stream_entropy_change`, `balanced_composite_entropy_generation` (`thermodynamics.py`): entropy and exergy kernels.
- `aggregate_weighted_objective`, `feasible_objective_scalar`, `infeasible_objective_scalar`, `penalized_feasible_objective_scalar`, `g_penalty` (`penalties.py`): scalar objective maps and fallback penalty.
- `coordinate_optimisation`, `map_optimisation_options`, `PlacementOptimisationOutcome` (`optimisation.py`): bridge to the scalar optimiser and canonical replay.
- `build_aggregate_composite`, `placement_utility_collections`, `allocation_result`, `allocate_from_aggregate_profiles`, `PreparedAggregateComposite` (`replay.py`): replay against prepared Total Site profiles.
- `INTERNAL_TEMPERATURE`, `INTERNAL_TEMPERATURE_DIFFERENCE`, `convert_to_request_units` (`units.py`): internal units and result conversion.
- `UtilityPlacementError` and its subclasses (`errors.py`): typed failures carrying a code and JSON-safe context.
- `profiles.py` has only underscore-prefixed helpers (`_load_profiles`, `_calibrate_profile`, `_process_entropy_slices`, `_coordinate_bounds`, `_finite_tuple`), which `replay.py` and the application layer import.
- Subdirectories: none.

## Invariants
- The analysis works internally in degC and delta_degC; `optimise_utility_placement` converts the temperatures of each returned candidate (`best` and `alternatives`) back to the request's units, while result-level `diagnostics` stay in degC and delta_degC [units.py:INTERNAL_TEMPERATURE] [service.py:optimise_utility_placement].
- Context and allocation models are frozen and reject extra fields and non-finite numbers; `PlacementTaskReport` and `PlacementOptimisationOutcome` are frozen and reject extra fields only [context.py:_FrozenContext] [allocation.py:_FrozenAllocation] [evaluation.py:PlacementTaskReport] [optimisation.py:PlacementOptimisationOutcome].
- Every period must supply each placement coordinate exactly once, period ids are unique, and at least one weight is positive [context.py:build_utility_placement_context] [bounds.py:_period_index].
- Caller bounds may narrow but never expand the physical bounds (expanding raises `UtilityTemplateValidationError`); an empty intersection raises `EmptyPlacementFeasibleRegionError` [bounds.py:_intersection].
- Without generated pairs, hot supplies descend and cold supplies ascend in declaration order by at least the minimum separation; with generated pairs, hot supplies descend within each level kind and across kinds only the physical spacing is enforced; all checks allow `tolerances.ordering` slack [bounds.py:_propagate_order] [codec.py:verify_candidate].
- Decision coordinates are ordered per side with isothermal supplies first, then each sensible supply followed by its span; paired requests list only the hot side [codec.py:build_decision_coordinates].
- For generated pairs, each cold level is the exact reversal of its hot partner [codec.py:decode_verified_placement] [codec.py:encode_placement].
- Scalar objectives partition the range: feasible values lie in (0, 1), infeasible in [1, 2) [penalties.py:feasible_objective_scalar] [penalties.py:infeasible_objective_scalar].
- Coverage tolerance is `tolerances.coverage + tolerances.relative * max(required, 1)` per side [allocation.py:allocate_placement_period].
- The optimiser searches the unit box; supply ordering is imposed structurally when mapping back to physical values [optimisation.py:_optimizer_point_to_physical] [optimisation.py:coordinate_optimisation].
- Memoisation is keyed on exact coordinates with negative zero normalised [evaluation.py:PlacementEvaluationSession.evaluate].
- The evaluation budget is split evenly across tasks, earlier tasks taking the remainder; a search batch from the top-level session leaves a polish reserve [evaluation.py:PlacementEvaluationSession.task_session].
- Only the error codes `invalid_balanced_composite`, `negative_entropy_generation`, `zero_utility_temperature_span` and `nonpositive_kelvin` reject one candidate; any other raises [evaluation.py:PlacementEvaluationSession.evaluate].
- The model's initial points and the optimiser's candidates are de-duplicated and re-evaluated in a fresh session, with the evaluation limit raised to cover them, before ranking [optimisation.py:coordinate_optimisation].
- Allocation adapter exceptions other than `PlacementTargetingError` are wrapped in one [allocation.py:allocate_placement_period].

## Side Effects
- Reports progress through `OpenPinch.domain.progress.emit`, which does nothing outside a `reporting` scope [optimisation.py:_PlacementProgress.report] [optimisation.py:coordinate_optimisation].
- The default runner starts optimiser worker processes; `PlacementEvaluationSession` is picklable and drops its memo and lock on pickling [service.py:optimise_utility_placement] [evaluation.py:PlacementEvaluationSession.__getstate__].
- `PlacementEvaluationSession` holds a per-process memo and a thread lock and accumulates counters; `merge_task_reports` mutates it [evaluation.py:PlacementEvaluationSession.merge_task_reports].
- Replay mutates only deep copies of utility collections [replay.py:allocate_from_aggregate_profiles].
- No file or network IO.

## Verification
- Run: `uv run --no-sync pytest tests/analysis/utility_placement`
- Guide: `docs/guides/utility-placement-optimisation.rst`
