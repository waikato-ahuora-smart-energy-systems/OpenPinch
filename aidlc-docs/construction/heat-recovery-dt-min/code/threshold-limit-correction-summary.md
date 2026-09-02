# Threshold-Limit Correction Summary

## Corrected behavior

The inverse heat-recovery service no longer assumes that a request at the
thermodynamic limit implies zero global `dt_min`. A threshold
problem can retain its maximum recovery over a positive `dt_min`
plateau. The solver now returns the greatest approach that still achieves the
limit, consistently with its maximal inverse rule for interior plateaus.

Zero remains the result, within the configured temperature tolerance, when the
maximum-recovery plateau has no positive extent or when the thermodynamic limit
itself is zero.

For the Bleaching process in tutorial notebook 02, the ordinary direct target
and thermodynamic limit are 14,121.972 kW. Inverting that target now returns
approximately 58.34505 delta_degC with `at_thermodynamic_limit` status.

## TDD coverage

- Analytical and packaged regressions pin positive threshold-limit boundaries.
- A generated threshold-domain property checks maximality, stream-order
  invariance, repeated-call idempotence, equivalent units, and JSON round trips.
- Existing above-limit, zero-limit, interior, tolerance, period, workspace,
  batch, and non-mutation coverage remains green.
- Notebook 02 directly inverts its ordinary target and numerically pins the
  threshold result.

## Verification

- Inverse feature suite: 46 passed.
- Notebook and RTD gate: 47 passed, 3 expected skips.
- Full configured suite: 2,583 passed, 4 skipped.
- Exact non-solver coverage selection: 2,580 passed, 3 skipped, 4 deselected.
- Branch coverage: 96 percent; the 95 percent gate passed.
- Ruff lint, feature-file formatting, and patch hygiene: passed.

Security and Resiliency extensions remained disabled. Property-Based Testing
rules PBT-01 through PBT-05 and PBT-07 through PBT-10 are satisfied; PBT-06 is
N/A because the solver owns no persistent mutable state.
