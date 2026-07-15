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
