# Utility Placement Canonical Fallback Penalty Code Generation Plan

This plan is the single source of truth for the bounded fallback-penalty
correction. The user's direct instruction authorizes the complete sequence.

- [x] **Step 1 - Context and requirements.** Load the current Utility Placement
  requirements, penalty implementation, evaluation call site, tests, and enabled
  Property-Based Testing rules.
- [x] **Step 2 - Scope and workflow.** Limit the change to the existing penalty
  adapter, focused expectations/properties, requirements wording, and workflow
  records. Skip new APIs, schemas, dependencies, and infrastructure.
- [x] **Step 3 - RED example and oracle coverage.** Change the hand-calculable
  expectation to the canonical squared result and add an assertion that Utility
  Placement matches `g_ineq_penalty` with `PenaltyForm.SQUARE`.
- [x] **Step 4 - RED integration coverage.** Update evaluation and public
  workflow expectations for the canonical coefficient and prove the old local
  result fails.
- [x] **Step 5 - GREEN implementation.** Route normalized fallback residuals
  through the canonical squared inequality-penalty function while preserving
  validation and aggregation behavior.
- [x] **Step 6 - Property verification.** Preserve scale invariance and prove
  canonical-oracle equivalence over generated valid fallback fractions.
- [x] **Step 7 - Focused Build and Test.** Run penalty, evaluation, application,
  and Utility Placement suites with the fixed Hypothesis seed, then run Ruff and
  patch-hygiene checks.
- [x] **Step 8 - Records and completion.** Update requirements, state, audit,
  implementation summary, and Build and Test evidence in the same interaction.

## PBT Compliance Plan

- PBT-03: retain non-negativity, scale invariance, and bounded scalar invariants.
- PBT-05: compare the Utility Placement adapter against `g_ineq_penalty` as the
  canonical oracle.
- PBT-07: use constrained duty fractions and positive required-duty generators.
- PBT-08: retain Hypothesis shrinking and the repository fixed CI seed.
- PBT-09: continue using Hypothesis with pytest.
- PBT-10: retain both hand-calculable example tests and generated properties.
- PBT-02, PBT-04, and PBT-06: N/A; there is no inverse, idempotent transform, or
  mutable state machine in this correction.
