# Utility Placement Codex Review Corrections Plan

This plan is the single source of truth for the two valid findings on GitHub
PR #91. The correction is bounded to maximum-duty replay serialization and the
existing shared utility-targeting assignment path. Public API shape,
thermodynamic objective, optimizer coordinates, notebook 19, and plotting are
unchanged.

## TDD checklist

- [x] **Step 1 - RED period-aware cap replay.** Add example regressions proving
  unequal multiperiod maximum duties validate, select by period identity even
  when the request order differs, and survive detached returned-case replay.
- [x] **Step 2 - RED complete duty replacement.** Add regressions proving a
  zero cap clears a pre-existing positive duty, a zero-load side clears all
  stale duties, and targeting updates only the selected multiperiod index.
- [x] **Step 3 - GREEN corrections.** Serialize the current period's cap as a
  scalar for single-period candidate replay, keep returned-case persistence
  separate and canonical, and apply every calculated duty including exact
  zeros only after both sides calculate successfully.
- [x] **Step 4 - Property and focused verification.** Add or retain Hypothesis
  invariants for cap enforcement, period isolation, and idempotent replacement;
  run the focused contracts, application, targeting, utility-placement, Ruff,
  and patch-hygiene gates.
- [x] **Step 5 - Complete verification and records.** Run the configured-solver
  repository suite without deselection, review every changed file for necessity,
  and align state, audit, implementation summary, and Build and Test evidence.

## Traceability

- GitHub P1: multiperiod limits must use an accepted value contract during
  replay and preserve period identity.
- GitHub P2: capped or unused utilities must not retain stale input duty.
- Requirements: Utility Placement FR-008 and acceptance criteria 24 through 27.
- Existing design: maximum-duty business rules 2, 4, 5, and 11.

## PBT compliance plan

- PBT-01: period isolation, cap range, and targeting idempotence are the
  identified properties.
- PBT-03: every written duty remains non-negative and no greater than its
  selected-period cap.
- PBT-04: retargeting the same profiles and caps twice produces the same duty
  vector as targeting once.
- PBT-05: the pure duty calculator is the oracle for the mutating assignment
  adapter.
- PBT-07: generated utility inputs use finite non-negative loads and caps with
  valid temperature ordering.
- PBT-08: Hypothesis shrinking remains enabled and verification uses the
  repository's fixed seed.
- PBT-09: use the existing Hypothesis and pytest stack.
- PBT-10: retain the two reviewed failures as explicit examples alongside the
  general properties.
- PBT-02 and PBT-06: N/A; no inverse transformation or command-sequence state
  machine is introduced.
