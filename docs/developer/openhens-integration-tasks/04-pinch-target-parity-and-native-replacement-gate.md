# HENS-04 Pinch Target Parity and Native Replacement Gate

## PRD Summary

Prove that OpenPinch's native problem-table and targeting logic can replace the
OpenHENS pinch decomposition path for migrated cases before deleting or
rewriting the OpenHENS-adapted pinch code.

## User Outcome

The PDM/TDM/ESM workflow can rely on OpenPinch-owned targeting primitives
without silently changing utility targets, pinch temperatures, stream masks, or
downstream topology generation.

## Scope

- Parity harness and focused adapter fixes.
- Minimal production code changes.
- Replacement of OpenHENS `pinch_classes` only after parity is proven.

## Plan Context

Read these sections before implementation:

- [Reuse Map](../../../OPENHENS_MIGRATION_PLAN.md#reuse-map)
- [OpenPinch Reuse Commitments](../../../OPENHENS_MIGRATION_PLAN.md#openpinch-reuse-commitments)
- [Canonical Synthesis Problem Contract](../../../OPENHENS_MIGRATION_PLAN.md#canonical-synthesis-problem-contract)
- [Phase 4: Pinch Target Parity and Replacement Plan](../../../OPENHENS_MIGRATION_PLAN.md#phase-4-pinch-target-parity-and-replacement-plan)
- [Regression Tolerances](../../../OPENHENS_MIGRATION_PLAN.md#regression-tolerances)

Settled decisions for this task:

- OpenHENS pinch decomposition can be removed only after OpenPinch
  `ProblemTable` and targeting helpers prove equivalent for the cases and grid
  coverage named in the README case acceptance matrix.
- Parity must include structural PDM fields, not only aggregate utility target
  values.
- Access must use `ProblemTableLabel` or canonical labels; raw row/column
  assumptions are not acceptable.
- Non-equivalence is either fixed at the adapter/convention boundary or recorded
  as a blocked algorithm decision.
- The required workflow in `README.md` is mandatory; parity work must validate
  the `PinchProblem`-prepared path, not a separate OpenHENS-like entry point.

## Requirements Checklist

- [ ] Build a parity harness that runs source OpenHENS
      `PinchDecompModel.calculate_pinch()` logic and OpenPinch problem-table /
      direct-integration logic for the same prepared `PinchProblem`.
- [ ] Compare hot utility target for every matrix-required case and required
      `dTmin`.
- [ ] Compare cold utility target for every matrix-required case and required
      `dTmin`.
- [ ] Compare heat recovery target for every matrix-required case and required
      `dTmin`.
- [ ] Compare hot pinch, cold pinch, and shifted pinch temperature.
- [ ] Compare above-pinch and below-pinch active stream masks.
- [ ] Compare structural PDM fields: `z_i_active`, `z_j_active`, clipped hot and
      cold stream temperatures, `S`, `K`, and manual stage selection.
- [ ] Cover threshold cases where `HU_target == 0`.
- [ ] Cover threshold cases where `CU_target == 0`.
- [ ] Confirm `ProblemTable` access uses `ProblemTableLabel` or canonical
      string labels instead of raw row/column assumptions.
- [ ] Add stream-order tests proving parity does not depend on input row order.
- [ ] Identify unit convention differences explicitly.
- [ ] Identify `dt_cont` convention differences explicitly.
- [ ] Fix adapter-level convention mismatches where they are mechanical.
- [ ] Document any remaining non-equivalence as a blocked algorithm decision.
- [ ] Replace OpenHENS private `pinch_classes` usage only after parity passes
      for the required examples and `dTmin` grid.
- [ ] When replacing, route PDM decomposition data through OpenPinch
      `ProblemTable` and targeting helpers.

## General Standards That Apply

- [ ] Enforce the required OpenPinch workflow from `README.md`; deviations are
      not permitted.
- [ ] Do not change solver defaults, objectives, tolerances, or topology
      evolution as part of pinch target replacement.
- [ ] Do not delete source OpenHENS-adapted behavior until parity evidence
      exists.
- [ ] Keep structural parity checks, not only aggregate TAC checks.
- [ ] Keep parity tied to prepared `PinchProblem` state.

## Verification Checklist

- [ ] Parity tests pass across the README case acceptance matrix grid coverage.
- [ ] Tests cover target values, pinch temperatures, masks, and structural PDM
      fields.
- [ ] Row-order invariance tests pass.
- [ ] Any unit or `dt_cont` differences are documented or fixed.
- [ ] If OpenHENS `pinch_classes` were removed, tests prove OpenPinch-native
      targeting feeds the same downstream fields.

## Definition of Done

- [ ] The team has executable evidence that OpenPinch-native targeting is
      equivalent for the migrated examples and grids named in the README case
      acceptance matrix.
- [ ] OpenHENS-adapted pinch code is either still present behind a documented
      parity gate or removed with passing replacement tests.
- [ ] Downstream TDM/ESM task-shaping fields are covered by tests.
- [ ] No public workflow API changed in this task except as required by already
      completed schema/domain work.

## Out of Scope

- Moving StageWiseModel or GenericHENModel.
- Adding public design APIs.
- Replacing LMTD or costing helpers.
- Expanding to multi-utility synthesis beyond current OpenHENS behavior.

## Implementation Notes

- 
