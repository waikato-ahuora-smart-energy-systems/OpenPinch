# Heat-Recovery Approach Temperature Functional Design

## Solver invariants

For each evaluation, detached hot and cold streams receive half the trial
global approach and a contribution multiplier of one. The existing shifted
problem table then represents the requested global HRAT. Recovery must be
finite and non-increasing over the search bracket.

At zero approach, calculated recovery is the thermodynamic limit. At the
no-overlap bound, recovery must be zero within tolerance. Interior bisection
maintains a feasible lower endpoint and infeasible upper endpoint, returning
the lower endpoint so plateaus resolve to their greatest feasible approach.
The zero request instead bisects for the earliest zero-recovery endpoint.

## Failure behavior

Invalid user values raise `ValueError` or `TypeError` before solving.
Above-limit errors identify requested recovery, limit, scope, period, and unit.
Numerical non-finiteness, invalid brackets, and convergence exhaustion raise a
closed runtime error rather than returning an unverified result.

## Non-mutation boundary

Only detached stream collections are changed. The service does not call the
stateful ordinary targeting executor and does not update problem results,
period caches, target-run specifications, configurations, or source models.
