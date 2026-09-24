# Unit 1 Business Rules

## Rule Conventions

- **MUST** and **MUST NOT** are blocking requirements.
- A **candidate-local failure** is data describing one physically infeasible
  point; it is not an exception object.
- A **fatal failure** aborts the solve with the original causal chain.
- Numerical tolerance choices remain the existing HPR tolerances unless a later
  Functional Design explicitly changes them.

## Penalty Rules

### BR-U1-001 — Accepted input forms

A penalty source MUST be a non-boolean real scalar or a rectangular numeric
array-like value. Array rank is unrestricted because flattening removes the
observed topology-dependent shape distinction.

### BR-U1-002 — Stable flattening

Every valid array MUST flatten in row-major order to one immutable tuple of
Python floats. Equivalent scalar and singleton inputs MUST normalize to the same
one-term tuple.

### BR-U1-003 — Empty input

An empty valid numeric array MUST normalize to an empty tuple and contribute
zero feasibility penalty.

### BR-U1-004 — Invalid structure

Ragged, object-dtype, non-numeric, boolean-containing, NaN, and infinite values
MUST raise a fatal penalty-contract error. They MUST NOT become a failed
candidate or the legacy sentinel objective.

### BR-U1-005 — Sign ownership

Normalization MUST preserve finite signs and magnitudes. The shared accounting
step, not the normalizer, owns clipping negative inequality terms to zero and
applying the configured penalty form.

### BR-U1-006 — Single path

Cascade, parallel, Carnot, vapour-compression, VC+MVR, single-period, and
multiperiod objectives MUST use the same normalizer. Topology-specific wrapper
shapes such as `[hp.penalty]` MUST have no semantic effect.

## Candidate Result Rules

### BR-U1-007 — Result type

Every objective evaluation MUST return `HPRBackendResult`. A different type is
a fatal contract failure.

### BR-U1-008 — Successful scalar

A search result with `success=True` MUST have a finite scalar `obj`.
Non-finite or non-scalar objectives are fatal contract failures rather than
rankable candidates.

### BR-U1-009 — Search artifact boundary

Search mode MUST NOT publish or retain a live model, debug figure, public
simulation record, or transaction-ready public output. It MAY retain detached
scalar and small array facts required for ranking and diagnostics.

### BR-U1-010 — Final artifact boundary

Only final mode MAY construct public streams, complete economics, simulation
evidence, and debug artifacts. Debug artifacts remain internal even in final
mode.

### BR-U1-011 — Formula equivalence

Search and final mode MUST use the same thermodynamic equations, penalty
normalization, objective formula, and configured tolerance. Final mode adds
artifacts; it does not define a second objective.

### BR-U1-012 — Final physical divergence

If final reevaluation produces a newly classified physical infeasibility, the
point becomes candidate-local and the coordinator MAY try the next ranked
candidate.

### BR-U1-013 — Final fatal divergence

Artifact construction, malformed output, lifecycle misuse, simulation-record
construction, detachment, and copy failures are fatal. They MUST NOT be
converted to an infeasible point.

### BR-U1-014 — Failure reason boundary

A candidate-local backend result contains a stable reason code and bounded
summary. It MUST NOT retain the caught exception or raw traceback.

## Detachment Rules

### BR-U1-015 — Public model compatibility

`HeatPumpTargetOutputs.model` remains an optional compatibility field and MUST
be `None` on every published output.

### BR-U1-016 — Internal engine lifetime

A live engine model MAY exist only inside one analysis evaluation and an
internal `HPRThermoArtifacts` container. It MUST be released or become
unreachable when that evaluation/finalization ends.

### BR-U1-017 — Output field translation

`HPRBackendResult.to_output_fields()` MUST omit the internal model and debug
figure unconditionally. It MUST NOT rely on a later copy failure to identify
leakage.

### BR-U1-018 — Canonical evidence

A successful simulated CoolProp or TESPy target MUST carry a valid detached
`target_simulation_record`. Analytical targets that do not instantiate a
simulation engine MAY omit it.

### BR-U1-019 — Recursive period detachment

Every nested entry in `period_outputs`, the selected-period output, and the
weighted parent MUST satisfy the same public no-engine-object rules.

### BR-U1-020 — Closed public graph

The finalizer MUST traverse the complete public object graph and fail closed on
an unknown arbitrary object. Allowed existing domain objects must be enumerated
and independently known to be copy-safe.

### BR-U1-021 — Atomic commit

The existing application transaction MUST deep-copy the detached result before
mutating committed application state. A copy failure leaves prior state
unchanged and propagates as fatal.

### BR-U1-022 — Serialization meaning

“Serializable facts” means typed detached records and data that support their
declared model/mapping representation. Unit 1 does not require the complete
stream-rich target graph to be plain JSON.

## Simulation Evidence Rules

### BR-U1-023 — Backward-compatible extension

The existing simulation record MUST remain the public type. Existing
single-stage top-level fields retain their names and meaning.

### BR-U1-024 — Closed topology identifier

The record MUST identify a supported topology through a closed value such as
single-stage VC, cascade VC, parallel VC, or VC+MVR. Unknown identifiers fail
validation.

### BR-U1-025 — Ordered loop and stage evidence

Topology-specific loops and stages MUST be immutable, ordered by physical
position, and uniquely identified within the record.

### BR-U1-026 — Detached stage facts

Loop/stage records MAY contain finite temperatures, pressures, duties, work,
efficiencies, fluid specifications, roles, and bounded assumptions. They MUST
NOT contain engine states, callables, exceptions, mutable sessions, or figures.

### BR-U1-027 — Topology consistency

The count, order, role, and fluid category of loop/stage records MUST be
consistent with the topology and the accepted design arrays. Missing or
duplicate evidence is fatal during finalization.

### BR-U1-028 — Power boundary

The record retains the declared compressor-only power boundary. Unit 1 MUST NOT
change accounting equations or silently include auxiliaries.

### BR-U1-029 — Period identity

A period identifier, when present, is non-empty and belongs to the containing
period output. Recursive finalization MUST NOT assign one period's identifier to
another.

## Budget Contract Rules

### BR-U1-030 — Exact type

`maximum_iterations` and `maximum_evaluations` MUST be positive,
non-boolean integers. Integral floats, strings, booleans, zero, negatives, NaN,
and infinity are rejected.

### BR-U1-031 — Defaults

Omitted values resolve to 300 iterations and 1,000,000 evaluations, matching the
current reusable optimizer defaults while making the resolved budget explicit.

### BR-U1-032 — Immutability

The resolved budget is immutable, engine-neutral, and suitable for inclusion in
bounded diagnostics.

### BR-U1-033 — Ownership

Unit 1 defines validation and defaults. Unit 2 owns public parameter precedence,
forwarding, and translation to `OptimisationOptions.maxiter` and `maxfun`.

## Diagnostic Contract Rules

### BR-U1-034 — Closed category

Failure categories form a closed enum covering preflight rejection,
candidate-local physical infeasibility, budget exhaustion, no viable candidate,
and fatal/internal boundaries. Direct process-MVR may reuse applicable category
values without importing optimized search.

### BR-U1-035 — Stable reason code

Every diagnostic uses a stable OpenPinch reason code. Raw engine messages may
inform internal causal logging but are not the stable public code.

### BR-U1-036 — Bounded summary

Human-readable summaries are sanitized, single-line, and length-bounded.
Representative diagnostic collections have a fixed small maximum set during
Functional Design for Unit 2; Unit 1 requires the cap to be finite and positive.

### BR-U1-037 — Detached context

Fluid, stage index, period identifier, and candidate index are optional detached
context. Stage/candidate indices, when present, are non-negative or follow the
explicit one-based stage convention documented by the owning service.

### BR-U1-038 — Count consistency

Summary category counts are non-negative; their total cannot exceed evaluated
outcomes plus preflight-only failures. The evaluated count is non-negative.

### BR-U1-039 — Error compatibility

`HPRTargetingError` inherits from `ValueError`, exposes one immutable
diagnostic summary, and uses a concise bounded message. It does not own search
or preflight behavior in Unit 1.

## Decision Tables

### Candidate outcome classification

| Condition | Search result | Final result | Action |
|---|---|---|---|
| Documented physical infeasibility | Candidate-local failure fact | Candidate-local failure fact | Continue when another point exists |
| Finite successful calculation | Rankable fact | Complete result | Rank or finalize |
| Non-finite objective with success flag | Fatal contract error | Fatal contract error | Propagate |
| Malformed penalty/result shape | Fatal contract error | Fatal contract error | Propagate |
| Artifact construction error | N/A in search mode | Fatal error | Propagate |
| Detachment/deep-copy error | N/A in search mode | Fatal error | Abort transaction |

### Simulation-record requirement

| Target kind | Engine session used | Record required | Public model |
|---|---:|---:|---|
| Analytical Carnot/Brayton without simulator | No | No | `None` |
| CoolProp vapour-compression | Yes | Yes | `None` |
| CoolProp cascade/parallel | Yes | Yes | `None` |
| CoolProp VC+MVR | Yes | Yes | `None` |
| Explicit TESPy supported topology | Yes | Yes | `None` |
| Multiperiod simulated target | Yes | Yes for every published period | `None` recursively |

## Compatibility Rules

1. Existing callers may continue to access `model`; its value is `None`.
2. Existing single-stage simulation-record fields retain meaning.
3. Existing result and target field names remain stable.
4. `HPRTargetingError` remains catchable by callers already catching
   `ValueError`.
5. Default budgets resolve to current generic optimizer values.
6. No new root import, deployment boundary, dependency, or persistence model is
   introduced.

## Testable Properties

| Property | Category | Required assertion |
|---|---|---|
| Penalty flattening | Invariant | Output is finite, one-dimensional, immutable, and row-major |
| Penalty normalization twice | Idempotence | Second normalization equals first |
| Penalty scalar oracle | Oracle | Vector result equals independent per-term scalar accumulation |
| Search/final core values | Oracle | Equivalent within declared tolerance for deterministic fake engines |
| Public object graph | Invariant | No forbidden live type appears at any depth |
| Contract records | Round-trip | Model → mapping/JSON-compatible representation → model preserves values |
| Period recursion | Induction | Adding one detached period preserves parent detachment |
| Diagnostic cap | Invariant | Representatives never exceed the positive cap |
| Budget validation | Invariant | Accepted values are exact positive integers and defaults are fixed |
| Deep-copy safety | Easy verification | Copy succeeds and preserves observable equality |

## PBT Compliance

- **PBT-01**: compliant; testable properties are enumerated and categorized.
- **PBT-02**: required for budget, diagnostic, summary, and simulation records.
- **PBT-03**: required for penalty, finite-objective, detachment, ordering, and
  bounded-diagnostic invariants.
- **PBT-04**: required for penalty normalization idempotence.
- **PBT-05**: required for scalar penalty and search/final reference comparison.
- **PBT-06**: N/A to Unit 1 immutable contracts and pure transformations;
  stateful candidate/cache orchestration belongs to Unit 2.
- **PBT-07**: constrained reusable domain strategies are required.
- **PBT-08**: shrinking and deterministic reproduction remain mandatory.
- **PBT-09**: existing Hypothesis/pytest selection is retained for later NFR
  confirmation.
- **PBT-10**: every critical property must be accompanied by explicit regression
  examples.

No blocking PBT finding exists at Functional Design.
