# CoolProp-Backed HPR and MVR Services Investigation

Date: 2026-09-24. Status: investigation complete; runtime remediation not started.

## Intent Analysis

- **User request**: Investigate failing and insufficiently robust CoolProp-backed
  HPR optimisation services, expanded to all MVR-prefixed services.
- **Request type**: Failure diagnosis and reliability assessment.
- **Scope**: Public direct and indirect vapour-compression heat-pump and
  refrigeration services; cascade, parallel, and VC+MVR targeting objectives;
  direct process-MVR component execution; the shared optimisation boundary;
  result detachment; and focused tests.
- **Complexity**: Moderate to high. The public failure is produced by interacting
  numerical, thermodynamic, optimisation, and application-transaction layers.
- **Authorization boundary**: Diagnosis only. No runtime implementation change is
  authorized by the request.

## Investigation Checklist

- [x] Load the current architecture, component, API, extension, and workflow state.
- [x] Run the focused HPR, optimisation, and service-orchestration test suites.
- [x] Reproduce real public-service failures on the packaged HPR sample.
- [x] Sample real CoolProp candidate outcomes and retain the hidden failure reasons.
- [x] Trace a successful-cycle failure to its exact source line.
- [x] Test a temporary in-memory penalty-shape correction.
- [x] Trace the next public-transaction failure exposed by that correction.
- [x] Test temporary in-memory corrections without changing repository runtime code.
- [x] Assess preflight, error reporting, optimisation budgets, and test coverage.
- [x] Trace the optimised `mvr_heat_pump` and deterministic process-MVR paths.
- [x] Reproduce direct and utility VC+MVR targets through the public API.
- [x] Probe VC+MVR warm-start and randomized candidate feasibility.
- [x] Verify the packaged process-MVR workflow and out-of-domain failure boundary.

## Reproduction Evidence

Environment: Python 3.14.2 and CoolProp 8.0.0.

The configured focused suite is green:

- 758 HPR, reusable optimisation, and service-orchestration tests passed.
- A separately selected MVR and process-component profile passed 97 tests with
  1,023 unrelated application and heat-pump tests deselected.

The real public API is not green. On
`OpenPinch/tutorials/sample_cases/heat_pump_targeting.json`, one condenser, one
evaporator, a 25 percent load selection, and one restart produced the following:

| Public service | Working fluid | Observed result |
|---|---|---|
| Direct heat pump | Ammonia | Generic optimal-result failure |
| Indirect heat pump | Ammonia | Generic optimal-result failure |
| Direct refrigeration | Ammonia | Generic optimal-result failure |
| Indirect refrigeration | Ammonia | Generic optimal-result failure |
| Direct cascade heat pump | Water | Generic optimal-result failure in 5.529 seconds |
| Direct parallel heat pump | Water | Generic optimal-result failure in 7.058 seconds |

The common public error was:

```text
Heat pump and refrigeration targeting (...) failed to return an optimal result.
```

That message is downstream of the real causes and contains no candidate failure
context.

The expanded real MVR service probes produced the following additional evidence:

| Public MVR surface | Packaged case | Observed result |
|---|---|---|
| Direct VC+MVR heat-pump target | `heat_pump_targeting.json` | Generic optimal-result failure in 19.868 seconds |
| Utility VC+MVR heat-pump target | `heat_pump_targeting.json` | Live CoolProp model deep-copy failure in 21.699 seconds |
| Direct VC+MVR heat-pump target | `process_mvr.json` | Live CoolProp model deep-copy failure in 25.011 seconds |
| VC+MVR target with invalid MVR fluid | `process_mvr.json` | Generic optimal-result failure in 7.855 seconds |
| Direct process-MVR component plus targeting | `process_mvr.json` | Success; component solve in 0.0895 seconds and target in 0.0318 seconds |

The direct process-MVR service is therefore not generally broken. The blocking
public failures are concentrated in optimised VC+MVR targeting, while direct
process-MVR has narrower failure-boundary and observability gaps.

## Confirmed Findings

### F-1: Viable CoolProp candidates are destroyed by a penalty-shape defect

**Severity**: Blocking functional defect.

`VapourCompressionCycle.penalty` is stored as a NumPy array. The cascade and
parallel aggregate `penalty` properties can therefore return an array even though
their annotations and docstrings claim a scalar. Both targeting objectives then
wrap that value in another list:

- `targeting/cascade_vapour_compression.py:363`
- `targeting/parallel_vapour_compression.py:287`

`evaluate_vapour_hpr_result()` appends two scalar profile penalties and
`common/shared.py:54` attempts to convert the heterogeneous sequence to a float
array. A solved candidate then raises:

```text
ValueError: setting an array element with a sequence
```

The objective catches this programming error and marks the candidate failed. A
deterministic 104-point direct heat-pump probe with water found no successful
candidates on the current path. With only a temporary in-memory flattening of
the penalty terms, the same points produced 18 successful candidates; the
existing Carnot warm start also became successful.

The defect affects both cascade and parallel simulated vapour-compression
targeting. It is not a CoolProp property failure; it occurs after CoolProp has
returned a solved cycle. VC+MVR already flattens its nested cycle penalties and
does not reproduce this specific shape defect.

### F-2: A successful CoolProp result cannot cross the application transaction

**Severity**: Blocking functional defect exposed after F-1.

The CoolProp objectives attach the live cycle object as `artifacts.model`.
`HPRBackendResult.to_output_fields()` copies it into the public
`HeatPumpTargetOutputs.model` field at `contracts/hpr.py:368`. The target
transaction then deep-copies the public result at
`application/_problem/targeting/state.py:204` before committing atomically.

The live cycle owns `CoolProp.AbstractState`, which is not pickleable or
deep-copyable. A successful warm-start target therefore fails with:

```text
TypeError: cannot pickle 'CoolProp.CoolProp.AbstractState' object
```

The new detached `HprTargetSimulationRecord` already contains the required
winning design facts. Retaining the live CoolProp model in the public target is
incompatible with the current detached-result contract.

This defect also blocks optimised VC+MVR. Both the utility target on
`heat_pump_targeting.json` and the direct target on `process_mvr.json` found a
successful CoolProp result and then failed at the same transaction deep-copy.

With temporary in-memory corrections for F-1 and removal of only this transient
model, the four public service variants returned valid water/CoolProp targets:

| Public service | Result | Elapsed time using the existing Carnot warm start |
|---|---|---:|
| Direct heat pump | Success | 0.254 seconds |
| Indirect heat pump | Success | 0.207 seconds |
| Direct refrigeration | Success | 0.226 seconds |
| Indirect refrigeration | Success | 0.254 seconds |

These probes preserved streams, economics, duties, backend identity, and the
detached target simulation record. They bypassed the long global search only to
isolate the public result path.

### F-3: Broad exception conversion hides programming defects and physical causes

**Severity**: High reliability and supportability defect.

The cascade, parallel, and VC+MVR objective wrappers catch every `Exception` and
convert it to `HPRBackendResult.failure`. The MVR cycle itself also catches every
exception around actual-state construction and reduces it to an unsolved cycle.
`_scalar_hpr_objective()` then maps every such failure to the same finite value,
`1e30`. `solve_hpr_placement()` finally raises a generic error after reevaluating
the ranked candidates.

Consequences:

- programming defects such as F-1 look identical to thermodynamic infeasibility;
- candidate reason counts and the best failure are discarded;
- the optimiser sees a large flat plateau and spends work exploring points that
  provide no ranking information;
- the public service provides no actionable fluid, state, point, or cause;
- `debug=False` is hardcoded during service preprocessing, so the public path
  cannot expose the original traceback.

For the 20-variable VC+MVR target on `heat_pump_targeting.json`, a deterministic
probe of the Carnot warm start plus 100 seeded points found zero direct-target
successes. The utility target found one success, and it was the warm start; all
100 seeded points were infeasible. Some direct failures contained the useful
underlying fact that the requested pressure exceeded CoolProp's numerical
critical point, but that detail is discarded by the public service.

Candidate-local thermodynamic failures should remain recoverable, but unexpected
programming, contract, lifecycle, and serialization failures must not be silently
reclassified as ordinary infeasible points.

### F-4: CoolProp targeting lacks the documented state-capability preflight

**Severity**: High robustness and diagnostics gap.

The performance-map fluid owner already provides
`resolve_hpr_working_fluid()`, which proves dew and bubble saturation states.
TESPy targeting has a dedicated preflight. CoolProp targeting instead only sorts,
truncates, or pads refrigerant names in
`common/shared.py:409-433` and starts the optimiser. VC+MVR only strips and pads
its separate MVR-fluid list before starting the same optimiser.

For this sample, ammonia warm starts attempted condenser pressures above the
CoolProp numerical critical pressure. The user receives the same generic
optimal-result failure rather than a fail-fast explanation that the selected
fluid cannot support the required operating states. This contradicts the current
documentation statement that capability is checked at the actual states before
optimisation.

An invalid VC+MVR fluid name was not rejected at the service boundary. One
restart spent 7.855 seconds evaluating property failures before returning only
the generic optimal-result error. CoolProp preflight must cover both the VC
refrigerants and MVR stage fluids.

### F-5: Optimisation budgets and candidate construction are not robust for the real objective

**Severity**: High performance and operability risk once F-1 is repaired.

HPR creates `OptimisationOptions` with only `n_runs`; the reusable defaults fix
`maxiter=300` and `maxfun=1,000,000`. The default HPR restart count is 10. Public
HPR arguments expose restarts but not evaluation or wall-time budgets.

After only the temporary F-1 correction, a single-run default CMA-ES direct heat
pump was still evaluating candidates after more than 140 seconds and was
interrupted. Each successful scalar evaluation constructs cycle streams,
problem tables, economics, a simulation record, and a live model, even though
the search consumes only a scalar objective. The already valid Carnot warm start
is merged into the result set only after the backend completes.

VC+MVR has the same budget architecture and a larger 20-variable default search
for the packaged case. A single restart took 19.868 to 25.011 seconds in the
observed public probes. In the utility probe, the only success among the warm
start and 100 seeded points was the warm start, yet the backend search still ran
before that known usable point could be returned.

The service needs explicit bounded budgets, early usable-baseline handling, and
a separation between lightweight search evaluation and final artifact assembly.
Repeated exact candidates should not repeat expensive property and profile work.

### F-6: The green test suite does not contain a blocking real public CoolProp oracle

**Severity**: High regression-coverage gap.

The focused suite passes because the critical layers are tested separately:

- solved objective tests use fake cycles whose `penalty` is a scalar;
- shared penalty tests use flat scalar lists;
- application orchestration tests stub the HPR solver;
- CoolProp point-simulator tests exercise the performance-map adapter rather than
  the optimisation service;
- notebook 09 catches `ValueError` and labels it `no feasible solution`, so a
  broken default CoolProp target can remain a successful notebook execution;
- notebook 11 similarly catches `ValueError`, `RuntimeError`, and
  `NotImplementedError` around `mvr_heat_pump`, converting a broken public MVR
  target into tutorial data labelled `no feasible solution`;
- no test deep-copies a public target containing a real CoolProp cycle.

The enabled Property-Based Testing extension did not generate nested scalar and
array penalty-term shapes, and no end-to-end property links a finite real-engine
candidate to a detached public target.

### F-7: Direct process-MVR succeeds normally but has an unbounded raw-property edge

**Severity**: Medium robustness and diagnostics gap.

The deterministic direct process-MVR component is materially healthier than the
optimised services. Its packaged Air case created a component, replacement
stream, stage record, and downstream direct target successfully. Its selection,
phase, saturation-pressure, unit, energy-balance, lifecycle, and serial/parallel
behaviour are also covered by the passing focused tests.

The service nevertheless accepts any positive stage-temperature lift without an
operating-domain bound or a domain-level error translation. On the packaged Air
case, a 5,000 degree Celsius stage lift exits quickly but exposes an internal
CoolProp one-phase flash message, including solver limits and raw `PropsSI`
arguments. In addition, liquid-injection saturation lookup and profile
saturation-breakpoint lookup catch every exception and silently fall back to a
dry stage or a reduced profile.

Those fallbacks may be valid policies, but they are currently indistinguishable
from accidental property failures. Direct process-MVR should reject obviously
unsupported compression requests before solving, translate remaining property
failures into stream/stage-specific domain diagnostics, and make any permitted
fallback observable in the result.

## Remediation Requirements

If implementation is authorized, the correction must satisfy all of the
following requirements.

### Functional requirements

1. Normalize every cycle and allocation penalty to an explicit scalar or a
   documented one-dimensional collection before shared aggregation.
2. A finite solved CoolProp candidate must remain successful through objective
   evaluation, ranking, final reevaluation, public validation, detachment, and
   transaction commit.
3. Public HPR results must contain detached, serializable facts only. Live
   CoolProp/TESPy engine objects must not cross the public result boundary.
4. Candidate-local thermodynamic infeasibility must remain recoverable, while
   unexpected code, contract, lifecycle, and detachment failures propagate with
   their cause.
5. When no candidate succeeds, the raised error must include bounded structured
   diagnostics such as evaluated count, failure categories, representative
   reasons, selected backend, cycle, and fluid.
6. CoolProp fluid and required-state capability must be checked before the
   expensive optimisation begins, with explicit handling of critical-state
   limits and unsupported dew/bubble states for both VC refrigerants and MVR
   stage fluids.
7. HPR optimisation must accept bounded execution controls for evaluations or
   iterations and must have a deterministic usable fallback policy for a valid
   warm start when the configured search budget is exhausted.
8. Search-time evaluation must avoid constructing transient public artifacts
   that are not needed to rank a point; final artifacts are assembled only for
   accepted candidates.
9. VC+MVR must evaluate and retain a valid Carnot-derived warm start before an
   optional global search, and an exhausted search budget must not discard that
   usable baseline.
10. Direct process-MVR must validate supported compression targets and translate
    CoolProp failures into bounded stream-, period-, and stage-specific errors.
11. Any permitted direct process-MVR dry-stage or reduced-profile fallback must
    be explicit and inspectable rather than silently swallowing arbitrary
    exceptions.

### Verification requirements

1. Add real CoolProp public-workflow regressions for direct and indirect heat
   pump and refrigeration services and for cascade, parallel, and VC+MVR
   topology.
2. Add an explicit unsupported-fluid/state preflight regression that proves the
   optimiser is not entered.
3. Add an atomic detachment regression using a real CoolProp-backed successful
   result.
4. Add property tests for scalar, empty-array, one-element-array, and multi-term
   penalty normalization, plus finite objective invariants.
5. Add a model property or bounded randomized test proving that candidate-local
   failures do not abort later viable candidates and that unexpected internal
   failures do abort.
6. Make notebook 09 assert a successful default CoolProp target and map rather
   than treating every `ValueError` as an acceptable screening result.
7. Add deterministic evaluation-count and elapsed-time gates for the packaged
   HPR example using a documented test profile.
8. Add real public direct and utility `mvr_heat_pump` regressions that prove a
   valid warm start crosses the detached transaction boundary.
9. Add an invalid-MVR-fluid preflight regression that proves no optimiser call is
   made and reports the offending stage and fluid.
10. Keep the packaged direct process-MVR workflow green and add atomic failure
    regressions for unsupported lift or pressure targets and property fallback
    reporting.
11. Make notebook 11 distinguish demonstrated infeasibility from service defects
    instead of accepting every target exception as a valid tutorial outcome.

## Scope Exclusions

- No change to the thermodynamic equations or declared compressor-only power
  boundary is inferred from this investigation.
- No fallback from an explicitly selected TESPy backend to CoolProp is proposed.
- No claim is made that ammonia is feasible for the packaged high-temperature
  study; the requirement is to reject unsupported states clearly and early.
- No redesign of the direct process-MVR thermodynamic model or its documented
  compressor-power boundary is inferred from the expanded investigation.
- No runtime code, public API, sample case, notebook, or dependency was changed
  during this investigation.

## Extension Compliance

- **Property-Based Testing**: Applicable remediation properties are identified
  for penalty normalization, failure classification, and end-to-end detachment.
  No implementation-stage compliance claim is made.
- **Security Baseline**: Disabled in `aidlc-state.md`; skipped.
- **Resiliency Baseline**: Disabled in `aidlc-state.md`; skipped.
