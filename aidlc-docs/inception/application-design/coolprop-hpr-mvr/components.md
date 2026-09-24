# Component Design: CoolProp HPR and MVR Reliability

## Scope

This design extends existing OpenPinch owners. It introduces no facade, service
deployment, persistence layer, or runtime dependency. Detailed equations and
candidate-classification rules are deferred to Functional Design.

## Components

### 1. Public HPR Target Accessor

**Owner**: `OpenPinch/application/_problem/accessors/target.py`

**Status**: Existing, extended.

**Responsibilities**:

- expose optional `maximum_iterations` and `maximum_evaluations` on every
  public heat-pump, refrigeration, and `mvr_heat_pump` targeting method;
- apply one precedence rule: an explicit named argument overrides the equivalent
  value in `options`, which overrides configuration defaults;
- forward the selected backend and budget without interpreting CoolProp states;
- preserve the existing application transaction and derived-case behavior.

It does not import CoolProp or inspect engine objects.

### 2. HPR Request and Result Contracts

**Owner**: `OpenPinch/contracts/hpr.py`

**Status**: Existing, extended.

**Responsibilities**:

- carry an immutable validated `HPRSearchBudget`;
- carry bounded engine-neutral failure diagnostics;
- keep `HeatPumpTargetOutputs.model` for compatibility while requiring its
  published value to be `None`;
- make `target_simulation_record` the canonical public design evidence;
- broaden the existing simulation record so every accepted CoolProp HPR topology,
  including VC+MVR, can be represented without embedding an engine object;
- carry the same budget and diagnostic shapes through single- and multiperiod
  requests.

Contract models may use primitive values, arrays, Values, and existing detached
domain records. They must not retain CoolProp `AbstractState`, TESPy networks,
closures, figures, or other live engine state.

### 3. HPR Failure Taxonomy and Diagnostic Accumulator

**Owner**: small shared module under
`OpenPinch/analysis/heat_pumps/common/`.

**Status**: New helper within the existing analysis boundary.

**Responsibilities**:

- define `HPRTargetingError`, compatible with `ValueError`;
- distinguish preflight rejection, candidate-local infeasibility, budget
  exhaustion, no viable candidate, and fatal internal failures;
- count every evaluated candidate by bounded category;
- retain a fixed maximum number of sanitized representative reasons;
- attach backend, cycle, fluid, stage or period context without raw property
  traces or engine objects.

Unexpected type, lifecycle, contract, and detachment defects are never converted
to candidate-local failures.

### 4. CoolProp Capability Preflight

**Owner**: existing performance-map/fluid boundary plus a shared HPR preflight
helper under `OpenPinch/analysis/heat_pumps/`.

**Status**: Existing capability functions reused and extended.

**Responsibilities**:

- resolve each configured VC refrigerant and MVR fluid once;
- check the dew, bubble, compression, and required temperature/pressure states
  implied by the prepared HPR bounds;
- identify critical-state limitations before optimisation;
- return a detached prepared specification for later evaluation;
- reject an unsupported fluid or state with a typed diagnostic before an
  optimiser call.

The preflight is selected only for `simulation_backend="coolprop"`. TESPy keeps
its existing fail-fast preflight and never falls back to CoolProp.

### 5. Candidate Objective Evaluators

**Owners**:
`OpenPinch/analysis/heat_pumps/targeting/` and
`OpenPinch/analysis/heat_pumps/_multiperiod/`.

**Status**: Existing, corrected.

**Responsibilities**:

- normalize nested/scalar penalty contributions before aggregation;
- provide a lightweight search evaluation that computes rankable scalar facts
  without public streams, figures, or live model publication;
- classify only documented thermodynamic infeasibility as candidate-local;
- perform a final evaluation for the accepted point to construct streams,
  economics, simulation evidence, and other public artifacts once;
- preserve the declared thermodynamic equations and compressor-only power
  boundary.

### 6. HPR Search Coordinator

**Owner**: `OpenPinch/analysis/heat_pumps/optimisation_adapter.py`

**Status**: Existing, extended.

**Responsibilities**:

- evaluate normalized warm starts before invoking the global backend;
- retain every finite viable warm start as a deterministic fallback;
- translate `HPRSearchBudget.maximum_iterations` to
  `OptimisationOptions.maxiter` and `maximum_evaluations` to
  `OptimisationOptions.maxfun`;
- rank unique candidates without repeating exact expensive evaluations;
- continue global search only within the configured limits;
- return the best accepted result, or raise `HPRTargetingError` with a bounded
  diagnostic summary;
- allow unexpected internal exceptions to propagate with their cause.

The generic `OpenPinch/optimisation` package remains unaware of HPR, CoolProp,
fluids, and HPR result contracts.

### 7. Detached Result Finalizer

**Owners**: `OpenPinch/contracts/hpr.py` and the HPR analysis translation seam.

**Status**: Existing translation corrected.

**Responsibilities**:

- validate an accepted backend result;
- discard live `HPRThermoArtifacts.model` before public output construction;
- publish `model=None` even when an internal final evaluation used an engine
  model;
- require a detached `target_simulation_record` for simulated accepted targets;
- recursively detach period outputs before multiperiod aggregation/publication;
- prove copy/deep-copy safety before the application transaction receives data.

### 8. HPR Service Coordinator

**Owner**: `OpenPinch/analysis/heat_pumps/service.py` and multiperiod execution.

**Status**: Existing, extended.

**Responsibilities**:

- construct request inputs, including the budget;
- select and run the matching backend preflight exactly once;
- dispatch the existing topology handler;
- finalize one detached output before cascade and transaction work;
- keep direct, utility, heat-pump, refrigeration, and multiperiod paths
  behaviorally aligned.

### 9. Direct Process-MVR Solver

**Owners**: `OpenPinch/analysis/heat_pumps/direct_mvr/`.

**Status**: Existing, hardened.

**Responsibilities**:

- validate the requested lift or pressure ratio against CoolProp capability
  before stage iteration;
- translate property failures into stream-, period-, and stage-specific
  `ValueError`-compatible domain errors;
- permit only named saturation-state and profile-breakpoint fallback policies;
- attach typed `DirectGasMVRFallbackDiagnostic` records to stage results;
- propagate all unexpected failures.

The direct process-MVR path remains deterministic and separate from optimised
VC+MVR targeting.

### 10. Verification and Tutorial Consumers

**Owners**: tests, notebook 09, and notebook 11 generators.

**Status**: Existing, strengthened later.

**Responsibilities**:

- exercise real public CoolProp target transactions;
- prove preflight occurs before optimisation;
- verify bounded calls, warm-start survival, detachment, and failure taxonomy;
- verify direct process-MVR fallback evidence;
- stop treating all target exceptions as demonstrated infeasibility.

## Boundary Invariants

1. No live property-engine object crosses from analysis to a public target.
2. A viable warm start cannot be discarded solely because global search exhausts
   its budget.
3. Candidate-local failures cannot abort a later viable candidate.
4. Fatal implementation and contract failures cannot become a scalar penalty.
5. CoolProp preflight completes before any global optimiser evaluation.
6. Public budget controls behave identically for single- and multiperiod paths.
7. Permitted direct process-MVR fallback behavior is visible in the result.
8. The reusable optimiser remains engine-neutral.

## Extension Compliance

- **Property-Based Testing**: applicable. The contracts expose finite objective,
  failure-classification, detachment, bounded-diagnostic, and state-transition
  invariants for later model and property tests.
- **Security Baseline**: disabled; skipped.
- **Resiliency Baseline**: disabled; skipped.
