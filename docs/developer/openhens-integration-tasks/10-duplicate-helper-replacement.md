# HENS-10 Duplicate Helper Replacement

## PRD Summary

Reduce duplicate migrated OpenHENS helper code in small, independently
reviewable slices after parity has proven the OpenPinch replacement behavior.

## User Outcome

OpenPinch owns one coherent implementation of shared pinch, LMTD, costing, and
export helper behavior without changing HEN synthesis results.

## Scope

- Focused replacements of duplicate helper families.
- One helper family per PR where possible.
- No broad solver model rewrites.
- No tolerance widening unless reviewed as a scientific decision.

## Plan Context

Read these sections before implementation:

- [Reuse Map](../../../OPENHENS_MIGRATION_PLAN.md#reuse-map)
- [OpenPinch Reuse Commitments](../../../OPENHENS_MIGRATION_PLAN.md#openpinch-reuse-commitments)
- [Phase 8: Replace Duplicate Helpers Incrementally](../../../OPENHENS_MIGRATION_PLAN.md#phase-8-replace-duplicate-helpers-incrementally)
- [Validation Strategy](../../../OPENHENS_MIGRATION_PLAN.md#validation-strategy)
- [Regression Tolerances](../../../OPENHENS_MIGRATION_PLAN.md#regression-tolerances)

Settled decisions for this task:

- Helper replacement happens only after focused parity tests prove formula and
  convention equivalence.
- Replace one helper family at a time so each change is independently
  revertible.
- If OpenPinch and OpenHENS formulas differ, keep the migrated helper or make
  the scientific decision explicit in a separate review.
- Aggregate TAC parity is insufficient by itself; structural, utility, area,
  unit-count, and export fields must be checked where affected.
- The required workflow in `README.md` is mandatory; helper replacement must not
  add bypass APIs for testing or convenience.

## Requirements Checklist

- [ ] For each helper family, identify the source OpenHENS behavior and the
      candidate OpenPinch replacement.
- [ ] Add focused parity tests before replacing the helper.
- [ ] Replace `pinch_classes` with OpenPinch `ProblemTable` and
      direct-integration services only after HENS-04 parity is green.
- [ ] Replace hand-coded LMTD post-processing with
      `OpenPinch.utils.heat_exchanger.compute_LMTD_from_dts` only where
      denominator and tolerance behavior match.
- [ ] Replace annualized exchanger cost helpers with `OpenPinch.utils.costing`
      only where fixed cost, variable cost, exponent, discount rate, service
      life, and annualization formulas match.
- [ ] Keep OpenHENS TAC convention explicit if it differs from OpenPinch
      area/cost targeting conventions.
- [ ] Replace ad hoc output tables with OpenPinch export helpers only when the
      synthesis artifact CSV contract remains unchanged.
- [ ] Compare TAC, topology, utility loads, area, unit counts, and exported CSV
      metrics before and after each replacement.
- [ ] Keep every helper replacement independently revertible.
- [ ] Update documentation if replacement changes which OpenPinch helper is the
      authoritative implementation.
- [ ] Remove duplicate tests only after equivalent OpenPinch-owned coverage
      exists.

## General Standards That Apply

- [ ] Enforce the required OpenPinch workflow from `README.md`; deviations are
      not permitted.
- [ ] No helper replacement proceeds without executable parity evidence.
- [ ] Do not combine helper replacement with equation moves.
- [ ] Do not change solver defaults, objective formulas, or tolerances.
- [ ] Do not rely only on aggregate TAC; compare structural and export fields.
- [ ] Prefer OpenPinch helpers when formulas and conventions truly match.

## Verification Checklist

- [ ] New parity tests fail against an intentional helper drift.
- [ ] Replacement tests pass for affected examples.
- [ ] Four-stream solver baseline still passes when affected by the helper
      family.
- [ ] Nine-stream final verification still passes only when the helper
      replacement is part of the final verification or retirement gate.
- [ ] Export snapshots still match when output helpers are touched.
- [ ] Existing fast OpenPinch tests pass.

## Definition of Done

- [ ] The targeted duplicate helper family has been replaced or deliberately
      left in place with documented non-equivalence.
- [ ] No TAC, topology, utility, area, unit-count, or export drift is introduced.
- [ ] OpenPinch has one authoritative helper path for the replaced behavior.
- [ ] The PR is revertible without undoing unrelated solver migration work.

## Out of Scope

- New HEN synthesis algorithms.
- Moving equation model slices.
- Public API expansion.
- OpenHENS repository retirement.

## Implementation Notes

- 
