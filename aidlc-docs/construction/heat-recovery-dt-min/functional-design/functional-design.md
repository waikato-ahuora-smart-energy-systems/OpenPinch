# Heat-Recovery `dt_min` Functional Design

## Solver invariants

For each evaluation, detached hot and cold streams receive half the trial
global `dt_min` and a contribution multiplier of one. The existing shifted
problem table then represents the requested global `dt_min`. Recovery must be
finite and non-increasing over the search bracket.

At zero approach, calculated recovery is the thermodynamic limit. At the
no-overlap bound, recovery must be zero within tolerance. Interior bisection
maintains a feasible lower endpoint and infeasible upper endpoint, returning
the lower endpoint so plateaus resolve to their greatest feasible approach.
The zero request instead bisects for the earliest zero-recovery endpoint.

A request at the thermodynamic limit uses the same greatest-feasible rule.
Threshold problems can retain maximum recovery across a positive `dt_min`
plateau, so ``at_thermodynamic_limit`` does not imply a zero approach. A
zero-limit problem remains the degenerate zero-`dt_min` case.

## Failure behavior

Invalid user values raise `ValueError` or `TypeError` before solving.
Above-limit errors identify requested recovery, limit, scope, period, and unit.
Numerical non-finiteness, invalid brackets, and convergence exhaustion raise a
closed runtime error rather than returning an unverified result.

## Non-mutation boundary

Only detached stream collections are changed. The service does not call the
stateful ordinary targeting executor and does not update problem results,
period caches, target-run specifications, configurations, or source models.
