# Heat-Recovery Approach Temperature Requirements

## Intent

Provide a non-mutating inverse pinch-analysis service that returns the global
heat-recovery approach temperature (HRAT/global delta Tmin) corresponding to a
requested process heat recovery.

## Functional requirements

- Accept stream data through the existing `PinchProblem(source)` contract.
- Expose selected-period and all-period targeting through problem, workspace,
  and ordered case-batch accessors.
- Accept finite, non-negative scalar heat recovery values, including the
  existing scalar value-with-unit representations.
- Reject Boolean, negative, non-finite, and above-limit recovery requests.
- Calculate the thermodynamic limit at zero global approach and report it in
  above-limit errors together with scope, period, and units.
- Ignore utilities and configured stream-specific `dt_cont` values by applying
  half the trial global approach to detached hot and cold process streams.
- Return the greatest feasible approach for an interior recovery plateau and
  the smallest zero-recovery boundary for a zero request.
- Preserve the canonical period order and isolate case-batch failures.
- Leave problem inputs, stream state, configured contributions, results,
  period caches, targets, and last-run metadata unchanged.

## Result contract

`HeatRecoveryApproachResult` is frozen, strict, finite, and JSON serializable.
It contains scope, canonical period ID, approach temperature, requested and
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
infrastructure, or package-root exports. HRAT is a process composite-curve
spacing and is explicitly distinct from exchanger-level EMAT.

## Verification requirements

Use TDD for contracts, public signatures, analytical and packaged regressions,
application orchestration, immutability, workspace/batch delegation, and
documentation inventories. Property-based testing applies under PBT-01 through
PBT-05 and PBT-07 through PBT-10; PBT-06 is N/A because the solver owns no
persistent mutable state.
