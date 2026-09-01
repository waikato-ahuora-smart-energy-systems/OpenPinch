# Utility Placement Maximum-Duty Implementation Summary

## Delivered Behavior

- `problem.target.utility_placement(...)` accepts an optional
  `maximum_duties` mapping keyed by the final generated or inferred utility
  name.
- Scalar, explicit-unit, and period-resolved limits normalize to the request's
  heat-flow unit. Scalars broadcast to every selected period.
- Exact candidate replay passes the current period's scalar cap. Returned-case
  metadata retains explicit period identities, maps them to canonical problem
  order, and leaves unselected periods internally unbounded.
- Each named limit is enforced independently in the shared utility targeting
  allocator. An omitted utility remains unbounded and a zero limit disables
  only that utility.
- Reserved `HU` and `CU` utilities are excluded from inferred placement
  options. They are added only as residual balancing fallbacks after named
  utility capacity is exhausted.
- Fallback duties participate in the balanced-composite entropy calculation.
  Their separate dimensionless ranking term delegates the normalized hot/cold
  residual vector to canonical
  `g_ineq_penalty(..., form=PenaltyForm.SQUARE)` with default `rho=10`, then
  aggregates as `sum(w[p] * g[p])` across all selected periods.
- The detached returned case retains maximum-duty metadata and any positive
  fallback definition. Ordinary direct or Total Site targeting therefore
  enforces the caps and standard GCC or TSP plots show the resulting utilities.
- Shared utility targeting calculates both sides before mutation and replaces
  every selected-period duty, including zero, preventing stale input duty from
  surviving a cap, unused level, or zero-load side.

## Public and Runtime Integration

The accessor, single-case all-period wrapper, and workspace batch wrappers use
the same mapping without a CLI surface. Runtime utility streams carry an
optional `maximum_heat_flow` separately from their allocated `heat_flow`.
Fallback endpoints use common temperature support across selected periods so
the returned utility definitions are replayable.

## TDD Evidence Before Complete Verification

- RED established 12 expected failures and 21 unaffected passes.
- The expanded GREEN suite passed 81 focused tests after final invalid-input
  branch review.
- The broad utility-placement, targeting, input-contract, and notebook suite
  passed 298 tests with 3 environment-guarded skips.
- Notebook 19 executes two isothermal plus two sensible levels per side,
  demonstrates four independent hot-side caps and residual `HU`, reports the
  fallback penalty, then uses the normal GCC workflow. Its Total Site example
  remains uncapped and uses the normal TSP workflow.

## Scope Compliance

- Property-Based Testing: compliant for capacity, coverage, scaling, and
  deterministic aggregation properties.
- Security Baseline: disabled; N/A.
- Resiliency Baseline: disabled; N/A.
- Monetary optimization, CLI integration, new plotting APIs, dependencies,
  and infrastructure changes remain excluded.
