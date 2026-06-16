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

- [x] For each helper family, identify the source OpenHENS behavior and the
      candidate OpenPinch replacement.
- [x] Add focused parity tests before replacing the helper.
- [ ] Replace `pinch_classes` with OpenPinch `ProblemTable` and
      direct-integration services only after HENS-04 parity is green.
- [x] Replace hand-coded LMTD post-processing with
      `OpenPinch.utils.heat_exchanger.compute_LMTD_from_dts` only where
      denominator and tolerance behavior match.
- [ ] Replace annualized exchanger cost helpers with `OpenPinch.utils.costing`
      only where fixed cost, variable cost, exponent, discount rate, service
      life, and annualization formulas match.
- [ ] Keep OpenHENS TAC convention explicit if it differs from OpenPinch
      area/cost targeting conventions.
- [ ] Replace ad hoc output tables with OpenPinch export helpers only when the
      synthesis artifact CSV contract remains unchanged.
- [x] Compare TAC, topology, utility loads, area, unit counts, and exported CSV
      metrics before and after each replacement.
- [x] Keep every helper replacement independently revertible.
- [x] Update documentation if replacement changes which OpenPinch helper is the
      authoritative implementation.
- [ ] Remove duplicate tests only after equivalent OpenPinch-owned coverage
      exists.

## General Standards That Apply

- [x] Enforce the required OpenPinch workflow from `README.md`; deviations are
      not permitted.
- [x] No helper replacement proceeds without executable parity evidence.
- [x] Do not combine helper replacement with equation moves.
- [x] Do not change solver defaults, objective formulas, or tolerances.
- [x] Do not rely only on aggregate TAC; compare structural and export fields.
- [x] Prefer OpenPinch helpers when formulas and conventions truly match.

## Verification Checklist

- [x] New parity tests fail against an intentional helper drift.
- [x] Replacement tests pass for affected examples.
- [ ] Four-stream solver baseline still passes when affected by the helper
      family.
- [ ] Nine-stream final verification still passes only when the helper
      replacement is part of the final verification or retirement gate.
- [x] Export snapshots still match when output helpers are touched.
- [x] Existing fast OpenPinch tests pass.

## Definition of Done

- [x] The targeted duplicate helper family has been replaced or deliberately
      left in place with documented non-equivalence.
- [ ] No TAC, topology, utility, area, unit-count, or export drift is introduced.
- [x] OpenPinch has one authoritative helper path for the replaced behavior.
- [ ] The PR is revertible without undoing unrelated solver migration work.

## Out of Scope

- New HEN synthesis algorithms.
- Moving equation model slices.
- Public API expansion.
- OpenHENS repository retirement.

## Implementation Notes

- Target helper family: LMTD post-processing only. `pinch_classes`, annualized
  exchanger cost helpers, and output-table helper replacements are intentionally
  untouched in this slice.
- Source behavior identified at OpenHENS commit
  `2afc14b7779482fc829edb1c3fa187b918d7fb19` with
  `rtk git -C /Users/ca107/Desktop/ahuora/OpenHENS rev-parse HEAD`. Source
  formulas are the post-process `math.log(delta_1 / delta_2)` LMTD expressions
  in `openhens/classes/stage_wise_model.py` and
  `openhens/classes/pinch_decomp_model.py`; the migrated OpenPinch model files
  had the same hand-coded formulas before this change.
- Replacement: `BaseHeatExchangerNetworkModel._post_process_lmtd(...)` now
  keeps the OpenHENS active-unit and dTmin/tolerance gates in the synthesis
  model layer, while delegating the positive endpoint logarithmic-mean formula
  to `OpenPinch.utils.heat_exchanger.compute_LMTD_from_dts`. This is private
  synthesis code only; no public API or workflow bypass was added.
- Parity evidence added in
  `tests/test_heat_exchanger_network_synthesis_models.py`:
  `test_lmtd_replacement_preserves_openhens_post_process_metrics` runs both
  `StageWiseModel.get_post_process()` and `PinchDecompModel.get_post_process()`
  on source-shaped solved arrays and compares independently computed source
  LMTD values, recovery/HU/CU areas, TAC, recovery/HU/CU unit counts, and
  hot/cold/recovery utility loads. Because expected LMTD values are computed
  independently of `compute_LMTD_from_dts`, a drift in the delegated helper
  changes `model.LMTD_*` and fails this parity test.
- Focused helper command:
  `rtk uv run pytest tests/test_heat_exchanger_network_synthesis_models.py -k lmtd_replacement -q`
  -> `2 passed, 17 deselected`.
- Focused HEN/export/topology command:
  `rtk uv run pytest tests/test_heat_exchanger_network_synthesis_models.py tests/test_heat_exchanger_network_synthesis_workflow.py::test_optional_exports_round_trip_from_problem_results tests/test_heat_exchanger_network_synthesis_hens08.py tests/test_utils/test_heat_exchanger_eq.py -q`
  -> `66 passed`.
- Existing fast-suite command:
  `rtk uv run pytest -q -m 'not solver'` -> `1132 passed, 31 warnings`.
  Warnings were existing zone/subzone and Pint warnings, not HENS-10 failures.
- Ruff command:
  `rtk uv run ruff check OpenPinch/services/heat_exchanger_network_synthesis/models/base.py OpenPinch/services/heat_exchanger_network_synthesis/models/pinch_decomposition.py OpenPinch/services/heat_exchanger_network_synthesis/models/stagewise.py tests/test_heat_exchanger_network_synthesis_models.py`
  -> `All checks passed!`.
- Full ruff command:
  `rtk uv run ruff check .` -> `All checks passed!`.
- Diff whitespace command:
  `rtk git diff --check -- . ':(exclude).DS_Store'` -> passed with no
  output.
- Solver baseline blocker: `rtk which couenne` and `rtk which ipopt` both
  exited `1` with no path output, so the live Four-stream solver baseline was
  not runnable in this environment. `rtk uv run pytest -q -m solver` in
  OpenPinch exited `5` with `1132 deselected`; this checkout currently declares
  the solver marker but has no marked OpenPinch solver tests.
- Pending review: the no-drift DoD and PR revertibility DoD are left unchecked
  until adversarial review confirms the evidence and slice boundaries.
