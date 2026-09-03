# Heat-Recovery `dt_min` Code-Generation Summary

## Delivered behavior

Implemented the selected-period and all-period
`heat_recovery_dt_min` target methods and their active workspace,
ordered batch, and batch all-period mirrors. The service accepts configured
plain heat-flow values or explicit scalar units and returns a frozen specialist
result with explicit configured output units.

The numerical owner deep-copies process streams, assigns half the trial global
approach to both sides with contribution multipliers set to one, and evaluates
the existing vectorized process cascade. It calculates the zero-`dt_min` limit,
uses the finite no-overlap bracket, and applies deterministic bounded bisection
for thermodynamic-limit, interior/plateau, and zero-recovery outcomes.
For a threshold problem, the thermodynamic-limit outcome is the greatest
positive global `dt_min` that retains maximum recovery rather than an
unconditional zero.

## Ownership

- Contract: `OpenPinch/contracts/heat_recovery_dt_min.py`
- Numerical inversion: `OpenPinch/analysis/targeting/dt_min.py`
- Problem and period orchestration:
  `OpenPinch/application/heat_recovery_dt_min.py`
- Public delegation: problem target accessors and workspace case batches

No package-root export, input schema, optimizer, dependency, persistence, or
ordinary direct-target contract changed.

## TDD evidence

Red tests first pinned missing contract imports, public signatures, solver
module behavior, and application orchestration. Green implementation then
satisfied analytical two-stream, packaged forward/inverse, plateau,
zero-boundary, no-overlap, empty-side, invalid-input, numerical-clamping,
unit-conversion, canonical-period, concurrency, batch-isolation, and
byte-for-byte non-mutation assertions.

Seeded Hypothesis properties cover sensible and segmented scalar streams,
inactive streams, bounded multiperiod inputs, monotonicity, order invariance,
maximal inverse behavior, equivalent units, idempotence, forward/inverse
consistency, JSON round trips, and sequential/parallel parity.
