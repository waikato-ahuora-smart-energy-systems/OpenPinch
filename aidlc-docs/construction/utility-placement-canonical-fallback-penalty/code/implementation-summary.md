# Canonical Fallback Penalty Implementation Summary

## Outcome

Utility Placement now normalizes hot and cold fallback duties against their
required residual duties and delegates the resulting vector to
`g_ineq_penalty(..., form=PenaltyForm.SQUARE)`. The canonical default `rho=10`
replaces the former coefficient-free local squaring.

Validation for non-finite, negative, and inconsistent zero-required duties is
unchanged. Per-period reporting remains dimensionless, raw period weighting is
unchanged, and the feasible/infeasible optimizer scalar partition is preserved.

## Modified Production File

- `OpenPinch/analysis/utility_placement/penalties.py`

## Test Coverage

- hand-calculable canonical coefficient example;
- evaluation-session and real capped-dispatch integration regressions;
- Hypothesis scale-invariance and canonical-oracle property;
- unchanged invalid-duty, monotonicity, and scalar-boundary tests.

## PBT Compliance

- PBT-03: compliant; non-negativity, scale invariance, and scalar bounds pass.
- PBT-05: compliant; generated residuals match the canonical helper oracle.
- PBT-07: compliant; generators use bounded fractions and positive duties.
- PBT-08: compliant; shrinking remains enabled and the fixed seed was used.
- PBT-09: compliant; Hypothesis and pytest remain configured.
- PBT-10: compliant; example and generated tests both cover the change.
- PBT-02, PBT-04, PBT-06: N/A for this stateless one-way calculation.
