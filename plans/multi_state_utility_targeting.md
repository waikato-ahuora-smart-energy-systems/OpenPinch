# Multi-State Utility-System Targeting Plan

## Summary
Keep the existing process pinch workflow single-state, and introduce a separate multi-state utility-system layer that orchestrates many steady-state process solves.

The package should be split now into two analysis subpackages:

- `analysis/process/`: classical single-state process and site targeting
- `analysis/utility_systems/`: utility-side targeting and optimisation that can consume one or many operating states

This keeps `Stream`, `ProblemTable`, `Zone`, and the current process algorithms unchanged in concept. Multi-state awareness belongs above them, not inside them.

## Implementation Changes

### 1. Package boundary and folder split
Reorganize `analysis/` into:

- `analysis/process/`
  - current data preparation
  - problem-table and cascade methods
  - direct/indirect integration entry points
  - utility targeting for one steady-state case
  - graph-data extraction tied to a solved steady-state zone tree
- `analysis/utility_systems/`
  - current heat-pump/refrigeration targeting logic
  - current cogeneration/turbine routines
  - new multi-state orchestration and aggregation logic
  - shared background-profile extraction utilities

Keep `OpenPinch.analysis` as a compatibility barrel that re-exports moved symbols during the transition, but make `main.py` and internal imports consume the new subpackages directly.

### 2. Data model: keep process inputs flat, add explicit state metadata for utility studies
Do not make the base process engine accept "true" multi-state problems.

Instead:

- keep `TargetInput` as the single-state process payload
- extend `StreamSchema` and `UtilitySchema` with optional `state_id`
- add a new top-level multi-state study model for utility systems:
  - `UtilitySystemStudyInput`
  - repeated stream and utility records distinguished by `state_id`
  - explicit `states` metadata list containing:
    - `state_id`
    - `weight`
    - optional `label`
  - shared `options`
  - shared `zone_tree`

Add system-specific wrappers on top of that common study model:

- `HeatPumpTargetingStudyInput`
- `CogenerationTargetingStudyInput`

These wrappers carry the shared multi-state dataset plus system-specific design bounds and optimisation settings.

### 3. Core architecture: resolve states, solve process backgrounds, then optimise shared utility design
Implement the workflow in three layers:

1. **State resolution**
   - group repeated stream and utility records by `state_id`
   - validate that each state resolves to a complete steady-state `TargetInput`
   - validate that all declared `states` have matching records and weights

2. **Per-state process background generation**
   - for each `state_id`, run the existing single-state process targeting path
   - extract the background information needed by utility-system methods:
     - cascades
     - utility profiles
     - pinch temperatures
     - net loads by level/zone
   - store those in a dedicated internal background object

3. **Shared-design utility optimisation**
   - heat-pump and cogeneration optimisers consume the list of per-state backgrounds
   - one shared design is optimised across all states
   - state operating points may vary by state, but design variables are shared
   - the default objective is weighted annual performance using the `weight` field

This is the key design choice: process targeting stays steady-state; utility-system targeting becomes a cross-state orchestration and optimisation layer.

### 4. Public API and service split
Keep current process-facing APIs unchanged for single-state work:

- `pinch_analysis_service(...)`
- `PinchProblem`

Add a separate utility-study front door:

- `utility_system_targeting_service(...)` as the core service
- `UtilityStudyProblem` as the high-level convenience wrapper

Recommended `UtilityStudyProblem` surface:

- `load(...)`
- `validate()`
- `build_state_backgrounds()`
- `run_heat_pump_targeting(...)`
- `run_cogeneration_targeting(...)`
- `state_summary_frame()`
- `design_summary_frame()`

Process APIs should reject mixed multi-state payloads with a clear error directing the caller to the utility-study API. Do not overload `PinchProblem.run()` to silently choose between single-state and multi-state behaviour.

### 5. Refactor existing utility analyses to consume background objects
Before implementing the full multi-state layer, refactor the existing utility-side methods so they are no longer tightly coupled to one direct/indirect entry module.

Specifically:

- extract a reusable background-profile contract from the current heat-pump targeting flow
- make single-state heat-pump targeting consume that contract
- do the same for cogeneration/turbine targeting
- then implement multi-state variants that consume `list[StateBackground]` plus shared design variables

This avoids duplicating the current single-state logic and makes the multi-state optimisation a true extension rather than a parallel rewrite.

## Test Plan

### Process regression
- Existing single-state process targeting results remain unchanged.
- `pinch_analysis_service(...)` still accepts legacy payloads with no `state_id`.
- Existing heat-pump and cogeneration single-state workflows still run after the folder split.

### Schema and validation
- `UtilitySystemStudyInput` rejects:
  - missing `state_id` on repeated records
  - undeclared states
  - declared states with no records
  - duplicate state metadata
  - missing or nonpositive weights
- `TargetInput`-based process entry points reject mixed-state datasets.

### State resolution and background generation
- Repeated-record inputs resolve deterministically into per-state steady-state payloads.
- Each resolved state reproduces the same result as running that state alone through the current single-state engine.
- Zone-tree and utility definitions remain consistent across states where required.

### Multi-state utility optimisation
- Heat-pump shared-design targeting runs across at least 2-3 states and produces:
  - per-state operating results
  - shared design variables
  - weighted annual objective
- Cogeneration/turbine targeting does the same.
- Weighted-objective calculations match the declared state weights.
- Shared design variables are constant across states, while state operating points may vary.

### API and package split
- New `analysis/process` and `analysis/utility_systems` imports work.
- `OpenPinch.analysis` compatibility exports continue to function.
- `PinchProblem` remains single-state only.
- `UtilityStudyProblem` covers the multi-state utility-study workflow end to end.

## Assumptions and Defaults
- The physical split of `analysis/` happens now.
- Process targeting remains single-state by design.
- Multi-state support is added only for utility-system studies, not by making `Stream`, `Zone`, or `ProblemTable` multi-state objects.
- Input records use `state_id`, not `index`.
- Shared utility-system design across states is the primary optimisation mode.
- Cross-state objective defaults to weighted annual performance using explicit state weights.
- Heat-pump and cogeneration studies share the same multi-state orchestration pattern, but use separate system-specific study models and optimisation logic.
