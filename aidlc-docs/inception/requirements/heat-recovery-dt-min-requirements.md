# Heat-Recovery `dt_min` Requirements

## Intent

Provide a non-mutating inverse pinch-analysis service that returns the global
heat-recovery `dt_min` corresponding to a
requested process heat recovery.

## Functional requirements

- Accept stream data through the existing `PinchProblem(source)` contract.
- Expose selected-period and all-period targeting through problem, workspace,
  and ordered case-batch accessors.
- Accept finite, non-negative scalar heat recovery values, including the
  existing scalar value-with-unit representations.
- Reject Boolean, negative, non-finite, and above-limit recovery requests.
- Calculate the thermodynamic limit at zero global `dt_min` and report it in
  above-limit errors together with scope, period, and units.
- For a request at the thermodynamic limit, return the greatest feasible
  approach that retains maximum recovery. This boundary may be positive for a
  threshold problem; return zero only when no positive maximum-recovery
  plateau exists or the limit itself is zero.
- Ignore utilities and configured stream-specific `dt_cont` values by applying
  half the trial global `dt_min` to detached hot and cold process streams.
- Return the greatest feasible approach for an interior recovery plateau and
  the smallest zero-recovery boundary for a zero request.
- Preserve the canonical period order and isolate case-batch failures.
- Leave problem inputs, stream state, configured contributions, results,
  period caches, targets, and last-run metadata unchanged.

## Result contract

`HeatRecoveryDtMinResult` is frozen, strict, finite, and JSON serializable.
It contains scope, canonical period ID, `dt_min`, requested and
achieved recovery, thermodynamic limit, recovery residual, status, and
iteration count. Thermal values include explicit units and respect configured
output-unit overrides.

## Numerical requirements

- Use the existing vectorized process heat cascade without an optimizer.
- Use a no-overlap upper bound of the maximum hot temperature minus the
  minimum cold temperature, clamped at zero.
- Use deterministic bisection with a `1e-6 delta_degC` temperature tolerance,
  `1e-6 kW` absolute and `1e-9` relative recovery tolerances, and at most 100
  iterations.
- Fail closed on non-finite evaluations, invalid brackets, or non-convergence.

## Scope and compatibility

The feature is additive. It does not alter `direct_heat_integration`,
`DirectIntegrationTarget`, input schemas, persistence, configuration,
infrastructure, or package-root exports. Global `dt_min` is a process composite-curve
spacing and is explicitly distinct from exchanger-level EMAT.

## Verification requirements

Use TDD for contracts, public signatures, analytical and packaged regressions,
application orchestration, immutability, workspace/batch delegation, and
documentation inventories. Property-based testing applies under PBT-01 through
PBT-05 and PBT-07 through PBT-10; PBT-06 is N/A because the solver owns no
persistent mutable state.

## Audit-resolution amendment

This amendment resolves the six findings from the post-implementation audit.
It is a moderate, multi-component bug fix spanning contracts, application
normalization, zone orchestration, numerical targeting, tests, and user
documentation. It does not add a new public method or change the established
method signatures.

### AR-01: Strict recovery input shapes

- `heat_recovery` accepts only a finite, non-negative, non-Boolean numeric
  scalar; a scalar OpenPinch `Value`; a scalar Pint quantity; or an exact
  scalar mapping with `value` and `unit` fields.
- Numeric strings, bytes, sequences, NumPy arrays, arbitrary mappings,
  multiperiod values on the selected-period surface, and Boolean values at any
  nesting level are rejected before targeting.
- The numerical contributor functions reject Boolean `dt_min` and
  `period_idx` values, non-integral period indices, and other implicit scalar
  coercions that are outside their annotations.
- Equivalent supported heat-flow units remain invariant after conversion to
  the canonical internal unit.

### AR-02: Physical boundary accuracy

- The returned `dt_min` boundary must be accurate within
  `1e-6 delta_degC`, including maximum-recovery plateaus, interior plateaus,
  zero-recovery boundaries, and coincident shifted temperatures.
- The evaluator used by the inverse service must not allow temperature-grid
  canonicalization noise to violate the monotonic predicate assumed by
  bisection at the documented tolerance.
- Every returned boundary is post-verified against recovery evaluations at the
  boundary and immediately beyond it, using an offset sufficient to distinguish
  the documented temperature tolerance without weakening it.
- Existing ordinary targeting behavior and its shared temperature-grid rules
  remain unchanged unless a shared correction is proven safe by the complete
  regression suite.

### AR-03: Zone ownership and local resolution

- A string zone address continues to resolve against the current problem's
  execution root.
- A supplied `Zone` object is converted to its address and resolved against the
  current problem's execution root. Its attached streams and configuration are
  never used directly.
- If that address does not exist in the current problem, the service rejects
  the request with an error identifying the requested scope.
- Workspace and case-batch calls therefore resolve the same address
  independently for every case and cannot reuse another case's streams.

### AR-04: Exact-zero and micro-recovery semantics

- `zero_recovery_boundary` is reserved for a request whose canonical recovery
  is exactly zero.
- Every positive request, including one at or below the absolute recovery
  comparison tolerance, follows the positive-target path and returns `solved`
  unless it is within the thermodynamic-limit boundary classification.
- For positive requests at or below `1e-6 kW`, feasibility uses a comparison
  strict enough that the achieved recovery is not below the requested recovery;
  zero recovery is not accepted merely because the residual is within the
  general absolute tolerance.
- Above-limit validation and the existing relative tolerance for ordinary-scale
  recovery requests remain unchanged.

### AR-05: Strict result contract

- Contract construction rejects numeric strings, bytes, Boolean values, and
  other coercible non-numeric representations for thermal magnitudes.
- `dt_min` must be finite, non-negative, and dimensionally compatible with a
  temperature difference. Requested recovery, achieved recovery, and the
  thermodynamic limit must be finite, non-negative, and dimensionally
  compatible with heat flow. The residual may be signed but must use a
  compatible heat-flow unit.
- Contract validation converts compatible fields to common units before
  checking that the thermodynamic limit is not below requested or achieved
  recovery, achieved recovery meets a positive request under AR-04, and the
  residual equals achieved minus requested within the documented recovery
  tolerance.
- Status-specific relationships are validated: zero-boundary status requires
  an exact zero request; thermodynamic-limit status requires the request to
  equal the limit within tolerance; solved status represents a positive
  non-limit request.
- Frozen behavior, rejection of extra fields, finite JSON serialization, and
  JSON round-trip equality are preserved.

### AR-06: All-period mapping precedence

- A mapping whose keys exactly equal the canonical period IDs is always treated
  as a period mapping, even when those IDs are `value`, `unit`, or both.
- Only a mapping that does not exactly match the canonical period IDs may be
  interpreted as a scalar value-with-unit representation.
- Missing, extra, or duplicate-equivalent period identifiers remain invalid,
  and returned results preserve canonical period order.

### Audit-resolution verification

- Add permanent example regressions for every adversarial input found in the
  audit, foreign and same-address `Zone` objects, micro-recovery requests, the
  analytical threshold boundary, result-contract dimensional and relational
  failures, and `value`/`unit` period identifiers.
- Expand reusable Hypothesis strategies to generate multiple active hot and
  cold streams, sensible and segmented profiles, active and inactive streams,
  no-overlap and threshold cases, fractional and tolerance-scale duties,
  arbitrary valid period identifiers, and equivalent units.
- Property tests must cover monotonic recovery evaluation, boundary maximality,
  achieved-request feasibility, unit invariance, stream-order invariance,
  repeated-call idempotence, JSON round trips, local zone resolution, and
  sequential/parallel equivalence.
- Shrinking remains enabled, fixed seeds remain recorded, and any new
  counterexample receives a permanent example-based regression.
- PBT-01 through PBT-05 and PBT-07 through PBT-10 remain applicable. PBT-06
  remains N/A because the service owns no persistent mutable state and its
  non-mutation behavior is verified directly.
