# Default-Utility Penalty 1000 Summary

## Outcome

Utility Placement now uses a private squared fallback coefficient of 1000:

`g_p = 1000 * ((Q_HU / Q_heat_required)^2 + (Q_CU / Q_cool_required)^2)`.

This is ten times the immediately preceding coefficient of 100, one hundred
times the original coefficient of 10, and does not change the generic numerical
kernel's default for other analyses.

## Preserved behavior

Physical balanced-composite entropy, target-owned duties, feasibility,
dimensionless reporting, common-duty scaling invariance, raw period weighting,
and the feasible/infeasible ranking partition are unchanged.

## Verification

Coefficient-1000 fixed, generated oracle, scaling, evaluation, and public
workflow checks pass. The focused coefficient and notebook gate passes 47
tests. Ruff lint and warnings-as-errors Sphinx pass. The complete configured-
solver suite passes 2,471 tests with 4 expected skips and no deselections.
Notebook executability is verified live; stored execution counts and outputs
are not required because they are transient editor state.
