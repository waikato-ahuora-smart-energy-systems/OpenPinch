# Utility Placement Canonical Fallback Penalty Requirements

## Intent Analysis

- **Request type**: bounded numerical behavior correction.
- **Scope**: Utility Placement penalty calculation and focused tests.
- **Complexity**: low; the canonical penalty implementation already exists.
- **Authorization**: the user's direct implementation instruction approves this
  exact correction without additional clarification.

## Requirements

1. Preserve the existing dimensionless fallback residual for each side:
   fallback duty divided by required residual duty.
2. Calculate the combined fallback penalty with
   `g_ineq_penalty(residuals, form=PenaltyForm.SQUARE)`.
3. Use the canonical squared form's default `rho=10` coefficient; do not create
   a second Utility Placement penalty coefficient.
4. Preserve zero-duty handling and reject non-finite, negative, or physically
   inconsistent duty combinations.
5. Preserve raw period weighting, separate dimensionless reporting, and the
   feasible/infeasible scalar ranking partition.
6. Add an example regression and a Hypothesis oracle/invariance property proving
   equivalence with the canonical squared penalty.

## Stage Assessment

- User Stories, Application Design, Units Generation, Functional Design, NFR
  stages, and Infrastructure Design are skipped because this correction stays
  within the existing numerical component and public contract.
- Property-Based Testing remains enabled. PBT-03, PBT-05, PBT-07, PBT-08,
  PBT-09, and PBT-10 apply; the remaining PBT rules are N/A.
