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

- [x] Build a parity harness that runs source OpenHENS
      `PinchDecompModel.calculate_pinch()` logic and OpenPinch problem-table /
      direct-integration logic for the same prepared `PinchProblem`.
- [x] Compare hot utility target for every matrix-required case and required
      `dTmin`.
- [x] Compare cold utility target for every matrix-required case and required
      `dTmin`.
- [x] Compare heat recovery target for every matrix-required case and required
      `dTmin`.
- [x] Compare hot pinch, cold pinch, and shifted pinch temperature.
- [x] Compare above-pinch and below-pinch active stream masks.
- [x] Compare structural PDM fields: `z_i_active`, `z_j_active`, clipped hot and
      cold stream temperatures, `S`, `K`, and manual stage selection.
- [ ] Cover threshold cases where `HU_target == 0`.
- [ ] Cover threshold cases where `CU_target == 0`.
- [x] Confirm `ProblemTable` access uses `ProblemTableLabel` or canonical
      string labels instead of raw row/column assumptions.
- [x] Add stream-order tests proving parity does not depend on input row order.
- [x] Identify unit convention differences explicitly.
- [x] Identify `dt_cont` convention differences explicitly.
- [x] Fix adapter-level convention mismatches where they are mechanical.
- [x] Document any remaining non-equivalence as a blocked algorithm decision.
- [ ] Replace OpenHENS private `pinch_classes` usage only after parity passes
      for the required examples and `dTmin` grid.
- [ ] When replacing, route PDM decomposition data through OpenPinch
      `ProblemTable` and targeting helpers.

## General Standards That Apply

- [x] Enforce the required OpenPinch workflow from `README.md`; deviations are
      not permitted.
- [x] Do not change solver defaults, objectives, tolerances, or topology
      evolution as part of pinch target replacement.
- [x] Do not delete source OpenHENS-adapted behavior until parity evidence
      exists.
- [x] Keep structural parity checks, not only aggregate TAC checks.
- [x] Keep parity tied to prepared `PinchProblem` state.

## Verification Checklist

- [x] Parity tests pass across the README case acceptance matrix grid coverage.
- [x] Tests cover target values, pinch temperatures, masks, and structural PDM
      fields.
- [x] Row-order invariance tests pass.
- [x] Any unit or `dt_cont` differences are documented or fixed.
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

- 2026-06-16: Added private OpenPinch parity helper
  `OpenPinch/services/heat_exchanger_network_synthesis/pinch_decomposition.py`.
  It accepts a prepared `PinchProblem`, calls `problem_to_solver_arrays(...)` for
  the same problem/dTmin, copies the prepared `Zone`, applies the existing
  solver-array `dt_cont` fallback to process streams, then computes targets via
  OpenPinch `compute_direct_integration_targets(...)`.
- 2026-06-16 re-review fix: Native target extraction no longer reads raw
  `ProblemTable` row positions. The helper consumes OpenPinch's semantic
  `DirectIntegrationTarget` fields for HU target, CU target, heat recovery
  target, and hot/cold pinch temperatures; the underlying targeting service owns
  `ProblemTable` row interpretation.
- 2026-06-16: Added
  `tests/test_heat_exchanger_network_pinch_parity.py`. The test harness uses
  `PinchDecompModel.__new__` from the sibling source OpenHENS checkout and calls
  the source `calculate_pinch()` and `set_preprocessing()` methods directly, so
  parity is checked without running GEKKO solves or full solver benchmarks.
- 2026-06-16 re-review fix: Source OpenHENS parity is non-skippable. The focused
  parity test now requires sibling checkout `/Users/ca107/Desktop/ahuora/OpenHENS`
  at commit `2afc14b7779482fc829edb1c3fa187b918d7fb19`; a missing checkout,
  wrong commit, or non-importable `PinchDecompModel` fails with an actionable
  assertion instead of skipping the required parity evidence.
- 2026-06-16: Parity coverage includes
  `Four-stream-Yee-and-Grossmann-1990-1` and
  `Nine-stream-Linnhoff-and-Ahmad-1999-1` for the required dTmin grid
  `[2, 4, 6, 8, 10, 12, 14, 16, 18, 20]`, both above- and below-pinch
  decompositions, automated stage selection, manual stage selection `(2, 3)`,
  and base/reordered fixtures.
- 2026-06-16: Compared HU target, CU target, heat recovery target, hot pinch,
  cold pinch, shifted pinch temperature, `z_i_active`, `z_j_active`, clipped hot
  and cold stream temperatures, `S`, and `K`. The required matrix has no
  `HU_target == 0` or `CU_target == 0` rows; the new threshold-status test
  asserts that gap explicitly. Threshold behavior remains blocked until the case
  acceptance matrix names a threshold fixture or grid row.
- 2026-06-16: Unit convention difference identified: source OpenHENS
  `pinch_classes` builds the problem table in degC and stores PDM `T_pinch` in
  K by adding `273.15`; OpenPinch fixtures enter as explicit K values, while
  `ProblemTable` pinch temperatures compare on the degC table scale. Tests
  compare hot/cold pinch in degC and shifted pinch temperature in K.
- 2026-06-16: `dt_cont` convention difference identified and mechanically fixed
  in the private helper: converted fixtures prepare process streams with
  `dt_cont=0`, while OpenHENS PDM uses `dTmin / 2`; the helper applies the
  `problem_to_solver_arrays(...)` fallback to the copied prepared zone before
  calling OpenPinch targeting.
- 2026-06-16: No remaining target or structural non-equivalence was found for
  the required non-threshold matrix. OpenHENS private `pinch_classes` usage was
  not removed; replacement stays gated for adversarial review and future
  threshold coverage.
- 2026-06-16 re-review fix: Helper docstrings were revised to describe the
  stable private OpenPinch contract: prepared `PinchProblem` input,
  convention-normalized copied `Zone`, semantic target extraction, structural
  PDM snapshot output, private-only status, and the HENS-04 replacement gate.
- 2026-06-16 verification:
  `rtk uv run pytest tests/test_heat_exchanger_network_pinch_parity.py` -> 83
  passed.
- 2026-06-16 verification:
  `rtk uv run pytest tests/test_heat_exchanger_network_array_adapter.py tests/test_lib/test_synthesis_schemas.py`
  -> 34 passed.
- 2026-06-16 verification:
  `rtk uv run ruff check .` -> all checks passed.
- 2026-06-16 verification:
  `rtk git diff --check -- . ':(exclude).DS_Store'` -> passed with no output.
- 2026-06-16 re-review verification:
  `rtk uv run pytest tests/test_heat_exchanger_network_pinch_parity.py` -> 83
  passed with source OpenHENS commit
  `2afc14b7779482fc829edb1c3fa187b918d7fb19`.
- 2026-06-16 re-review verification:
  `rtk uv run pytest tests/test_heat_exchanger_network_array_adapter.py tests/test_lib/test_synthesis_schemas.py`
  -> 34 passed.
- 2026-06-16 re-review verification:
  `rtk uv run ruff check .` -> all checks passed.
- 2026-06-16 re-review verification:
  `rtk git diff --check -- . ':(exclude).DS_Store'` -> passed with no output.
