# Default-Utility Penalty Tenfold Summary

## Outcome

Utility Placement now weights normalized `HU` and `CU` fallback use with a
private squared-penalty coefficient of 100 rather than 10. For period `p`:

`g_p = 100 * ((Q_HU / Q_heat_required)^2 + (Q_CU / Q_cool_required)^2)`.

This is exactly ten times the previous default-utility ranking contribution.
The generic `g_ineq_penalty` default remains 10 for all other consumers.

## Preserved behavior

- physical balanced-composite entropy generation is unchanged;
- fallback and named utility duties remain target-owned outputs;
- the penalty remains dimensionless, non-negative, squared, and invariant
  under common fallback/required-duty scaling;
- period weights still aggregate raw per-period penalties; and
- the feasible/infeasible scalar partition remains unchanged.

## Verification

The RED gate produced eight exact factor-of-ten failures across fixed,
generated oracle, evaluation, and public capped-workflow tests. GREEN passes
all 27 of those checks. The expanded focused gate passes 289 tests with 3
expected optional-profile skips. The complete configured-solver suite passes
2,471 tests with 4 expected skips and no deselections.
