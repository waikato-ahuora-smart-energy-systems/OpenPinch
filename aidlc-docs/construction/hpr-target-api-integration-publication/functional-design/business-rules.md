# Unit 3 Business Rules

## Public API and Selector Rules

- **BR-U3-001**: Only the current `vapour_compression_heat_pump` and
  `vapour_compression_refrigeration` target methods accept the new
  `simulation_backend` selector.
- **BR-U3-002**: Omitting the selector is exactly equivalent to selecting
  `coolprop` explicitly.
- **BR-U3-003**: Selector normalization accepts strings only, trims whitespace,
  compares case-insensitively, and returns only `coolprop` or `tespy`.
- **BR-U3-004**: Empty, non-string, and unknown selector values fail before any
  target or simulator work.
- **BR-U3-005**: The thermodynamic simulation backend, HPR cycle identity, and
  black-box optimization method are separate values and may not be overloaded.
- **BR-U3-006**: No backend failure may silently fall back to the other backend.
- **BR-U3-007**: An ordinary target call returns its existing target subtype and
  performs no performance-map grid generation.
- **BR-U3-008**: The feature adds no package-root export, CLI command, automatic
  map attachment, or second public HPR workflow.

## CoolProp Compatibility Rules

- **BR-U3-009**: The omitted-selector CoolProp path retains the current
  candidate encoding, objective functions, fluid selection, thermodynamic
  calculations, streams, accounting, optimizer behavior, and result types.
- **BR-U3-010**: Explicit `coolprop` and omitted-selector calls with identical
  effective inputs must produce equivalent observable targets within existing
  deterministic numerical tolerances.
- **BR-U3-011**: Backend plumbing may not add a CoolProp solve, change a
  tolerance, reorder refrigerants, or modify cycle/model objects on the default
  path.

## TESPy Targeting Rules

- **BR-U3-012**: Selecting `tespy` means TESPy thermodynamic evaluations are
  used inside ordinary HPR placement optimization and determine the returned
  target's duties, power, COP, streams, accounting, and objective.
- **BR-U3-013**: Initial TESPy targeting supports exactly one active evaporator,
  one active condenser, one refrigerant loop, and one scalar period context.
- **BR-U3-014**: Analytic Carnot, Brayton, MVR, integrated-expander, multi-loop
  cascade, multi-loop parallel, and shared-vector multi-period HPR targeting are
  rejected for TESPy before optimization.
- **BR-U3-015**: A selected-period scalar call and each independent scalar call
  made by an all-period wrapper may use TESPy.
- **BR-U3-016**: Pure fluids, registered blends, and explicit molar mixtures of
  arbitrary positive component count are eligible; no OpenPinch refrigerant
  allowlist or mixture-size limit is permitted.
- **BR-U3-017**: REFPROP is rejected. Other working-fluid support is established
  by constructing the installed property wrapper and required dew/bubble states.
- **BR-U3-018**: Every optimizer candidate is a nominal equipment design and
  receives a fresh TESPy design solve; it is not evaluated offdesign against a
  previously visited candidate.
- **BR-U3-019**: Candidate output may not depend on candidate evaluation order.
- **BR-U3-020**: A targeting evaluator may reuse only topology shells and
  immutable resources; it must reapply the complete candidate specification and
  clear mutable design state before each solve.
- **BR-U3-021**: The TESPy targeting leaf is the only module allowed to import
  TESPy. All application, contracts, domain, target, and engine-neutral modules
  remain importable without TESPy installed.
- **BR-U3-022**: Candidate output crosses the TESPy leaf only as finite plain
  values, immutable records, ordered temperature-enthalpy profiles, and
  sanitized JSON-compatible metadata.
- **BR-U3-023**: A candidate-local convergence or state failure marks that
  candidate infeasible and allows later candidates to run when the evaluator
  can restore a clean state.
- **BR-U3-024**: Dependency absence, unsupported topology, lifecycle corruption,
  and cleanup failure are request-fatal.
- **BR-U3-025**: The evaluator closes exactly once for every targeting attempt,
  including optimizer and translation failures.

## Thermodynamic and Accounting Rules

- **BR-U3-026**: Evaporation for zeotropic fluids uses the dew anchor and
  condensation uses the bubble anchor, consistent with Unit 2 and current
  vapour-compression semantics.
- **BR-U3-027**: Heat-pump useful duty is nonnegative sink duty; refrigeration
  useful duty is nonnegative source duty.
- **BR-U3-028**: Compressor-only electric power is finite and positive for an
  accepted candidate; pumps and other auxiliaries are excluded and documented.
- **BR-U3-029**: Accepted candidates satisfy sink duty equals source duty plus
  compressor power within the thermodynamic evaluation tolerance.
- **BR-U3-030**: Heating COP is sink duty divided by compressor power; cooling
  COP is source duty divided by compressor power.
- **BR-U3-031**: TESPy duties, power, and thermal profiles enter the existing HPR
  utility, feasibility, costing, and objective pipeline; no backend-specific
  accounting equation is introduced.
- **BR-U3-032**: Normalized source and sink temperature-enthalpy profiles must be
  sufficient to build ordinary OpenPinch HPR streams without retaining TESPy
  objects.

## Target Record and Map-Basis Rules

- **BR-U3-033**: Every successful supported CoolProp or TESPy target carries one
  frozen plain targeting simulation record and one matching
  `hpr_simulation_backend` value.
- **BR-U3-034**: The record contains backend, mode, stable cycle/model identity,
  selected fluid, nominal temperatures, nominal useful duty, approaches,
  efficiencies, topology, period identity, engine version, power boundary, and
  structured assumptions.
- **BR-U3-035**: No CoolProp state, TESPy object, mutable configuration, stream
  collection, optimizer instance, or temporary path may enter the record.
- **BR-U3-036**: Map-basis extraction uses only the successful target and its
  record; it never reads the accessor's current problem configuration or private
  backend model.
- **BR-U3-037**: Extraction requires a scalar successful
  `single_stage_vapour_compression` target with one source, one sink, one fluid,
  and internally matching backend and mode.
- **BR-U3-038**: Failed, aggregate, array-valued, multi-port, missing-record, and
  inconsistent targets raise the typed HPR map compatibility error before map
  simulation.
- **BR-U3-039**: Basis extraction is deterministic, side-effect free, and deep
  detached.
- **BR-U3-040**: A request reference capacity overrides nominal useful duty for
  map generation only and never changes the target or its simulation record.

## Explicit Map Operation Rules

- **BR-U3-041**: `hpr_performance_map` requires keyword-only `target` and
  `request` arguments and returns one `HprPerformanceMap`.
- **BR-U3-042**: The selected backend comes only from the target simulation
  record; the request cannot override or relabel it.
- **BR-U3-043**: The bridge delegates exactly once to the Unit 2 generation
  service after compatibility validation and basis extraction.
- **BR-U3-044**: Map generation does not mutate the target, problem,
  configuration, caches, workspace selection, or request.
- **BR-U3-045**: Unit 2's all-or-nothing failure behavior remains authoritative;
  schema `1.0` never returns a partial map.
- **BR-U3-046**: The map operation is intentionally absent from automatic
  all-period and workspace batch mirrors; the caller selects one scalar target.

## Replay and Publication Rules

- **BR-U3-047**: Selected-period, all-period, and workspace batch target wrappers
  forward the normalized selector without changing canonical result order.
- **BR-U3-048**: Each independently targeted period or workspace case carries
  its own detached targeting simulation record.
- **BR-U3-049**: Public documentation states that TESPy selection changes
  ordinary targeting thermodynamics and is not merely a map-export choice.
- **BR-U3-050**: Documentation identifies all first-release topology,
  multi-period, working-fluid, power-boundary, convergence, and optional-install
  limitations.
- **BR-U3-051**: Schema and golden fixtures remain canonical versioned package
  resources and may be consumed without importing OpenPinch.
- **BR-U3-052**: OpenPinch production and tests for this feature may not import
  OpenUtility, Pyomo, or HiGHS.

## Error Rules

- **BR-U3-053**: Invalid public arguments fail before expensive work and identify
  the accepted values or compatibility requirement.
- **BR-U3-054**: Missing TESPy errors identify the optional installation extra
  through repository-standard guidance.
- **BR-U3-055**: Internal TESPy candidate diagnostics identify backend, model,
  temperatures, fluid identity, and a sanitized reason without exposing engine
  objects or temporary local paths.
- **BR-U3-056**: No successful target or map contains fabricated values for a
  non-converged candidate or point.

## Property-Based Testing Rules

- **BR-U3-057**: Code Generation must implement the PBT-01 properties identified
  in `business-logic-model.md` using reusable constrained HPR strategies.
- **BR-U3-058**: Omitted-versus-explicit CoolProp equivalence requires both a
  generated oracle property and concrete regression examples.
- **BR-U3-059**: Target-record round trips, target-to-basis invariants,
  non-mutation, wrapper propagation, unsupported-context rejection, mixture
  preservation, evaluator order independence, and physical consistency require
  generated properties plus critical explicit examples.
- **BR-U3-060**: Hypothesis shrinking and fixed-seed reproducibility remain
  enabled; real TESPy integration tests complement properties but do not replace
  deterministic fake-evaluator properties.

## Extension Compliance

- **PBT-01**: Compliant. Rules BR-U3-057 through BR-U3-060 bind the identified
  property categories to later Code Generation planning.
- **Security Baseline**: Disabled; not applicable.
- **Resiliency Baseline**: Disabled; not applicable.
