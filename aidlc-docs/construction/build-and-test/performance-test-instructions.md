# Performance Test Instructions

## Scope

Segment expansion is internal numeric work; parent collection and solver axes remain unchanged. Performance checks should therefore measure both profile expansion cost and HEN solve cost as segment count and operating-period count increase.

## Suggested Matrix

- Parent counts: 10, 100, and 1,000.
- Segment counts per parent: 1, 4, 16, and 64.
- Operating periods: 1, 4, and 12.
- HEN stages: 1, 3, and 5 for representative feasible cases.

Measure construction of `StreamCollection.segment_numeric_view()`, problem-table targeting, solver-array preparation, and APOPT/IPOPT HEN solve time. Record peak memory and confirm that parent-axis dimensions are invariant with segment count.

## Regression Thresholds

No new release-blocking wall-time threshold was introduced by this refactor. Flag super-linear growth in numeric-view or solver-array preparation relative to total expanded segment rows, and compare HEN solve time only within the same solver, topology, and convergence tolerance.

## GitHub CI Heat-Pump Zero-Duty Follow-Up

No performance test is required for this isolated guard. Zero-duty process
sides now skip CoolProp profile generation, so the repair removes work from that
path and introduces no loop, allocation, solver, network, or scaling behavior.
The full non-solver runtime remained approximately two minutes locally.

## Segment Pricing Follow-Up

No new release-blocking performance threshold is required. Exact utility cost is
piecewise linear in the already-prepared segment count and preserves parent HEN
axes. The complete non-solver suite finished in 120.22 seconds, and the four-case
solver-marked run finished in 85.49 seconds in the acceptance environment.

## Residual Compatibility Shim Removal

Dedicated performance testing is N/A. The change removes retry, mapping, alias,
and state-repair branches and does not alter a numerical kernel, loop bound,
solver axis, tolerance, or data-size relationship. The complete 2,063-test
non-solver suite finished in 104.45 seconds in the verification environment.
Future performance regressions remain covered by the package-level performance
matrix above.
