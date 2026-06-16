# HENS-11 Regression Expansion and OpenHENS Retirement

## PRD Summary

Expand solver and invariant coverage after migration, then retire OpenHENS as
an independent implementation only after OpenPinch has proven parity and shipped
the replacement API.

## User Outcome

There is one maintained HEN synthesis implementation in OpenPinch, supported by
strong regression coverage and a clear documentation-only migration path from
OpenHENS.

## Scope

- Additional solver regression tiers.
- Numerical invariant tests.
- Temporary shadow-run cleanup.
- OpenHENS archival/deprecation documentation.
- No wrapper package or compatibility layer.

## Plan Context

Read these sections before implementation:

- [Phase 9: Expand Regression Coverage](../../../OPENHENS_MIGRATION_PLAN.md#phase-9-expand-regression-coverage)
- [Phase 10: Retire OpenHENS as an Independent Implementation](../../../OPENHENS_MIGRATION_PLAN.md#phase-10-retire-openhens-as-an-independent-implementation)
- [Validation Strategy](../../../OPENHENS_MIGRATION_PLAN.md#validation-strategy)
- [Regression Tolerances](../../../OPENHENS_MIGRATION_PLAN.md#regression-tolerances)
- [Non-Goals for the First Migration](../../../OPENHENS_MIGRATION_PLAN.md#non-goals-for-the-first-migration)

Settled decisions for this task:

- OpenHENS is retired only after OpenPinch has proven parity and an OpenPinch
  release is available.
- Four-stream is the primary solver baseline for the migration. Nine-stream is
  the required final-verification benchmark before retirement.
- No wrapper package, runtime compatibility layer, import alias, or field alias
  should be maintained.
- Coverage must compare solver metrics and `HeatExchangerNetwork` content by
  stream identity.
- Final verification must compare the curated Nine-stream best-ESM network
  snapshot once HENS-00 defines the snapshot artifact and the migration creates
  it.
- Temporary shadow tests are transition tools and should be removed when
  OpenPinch-owned benchmark baselines are sufficient.
- The required workflow in `README.md` is mandatory; retirement is allowed only
  after verification proves the migrated implementation follows that workflow.

## Requirements Checklist

- [x] Keep Four-stream solver regression as the mandatory routine baseline.
- [ ] Run Nine-stream solver regression as the mandatory final-verification
      baseline before OpenHENS retirement.
- [ ] Add marked solver tests for additional OpenHENS example cases in tiers:
      small cases first, then medium cases, then large benchmark cases.
- [x] For every solver baseline, compare best ESM TAC.
- [x] Compare quartiles.
- [x] Compare within-2/5/10 percent counts.
- [x] Compare attempted jobs.
- [x] Compare solved ESM count.
- [x] Compare best stage count.
- [x] Compare recovery unit count.
- [x] Compare hot utility unit count.
- [x] Compare cold utility unit count.
- [x] Compare best `dTmin`.
- [x] Compare best derivative threshold.
- [x] Compare hot and cold utility loads.
- [x] Compare verification failures.
- [x] Compare `HeatExchangerNetwork` content by stream identity:
      recovery exchanger source/sink/stage/duty/area, hot utility links, cold
      utility links, utility loads, and unit counts.
- [x] Compare curated best-ESM network snapshots for Four-stream and Nine-stream
      before retirement.
- [x] Add numerical invariant tests that do not depend on benchmark workbooks:
      stream heat balances, stage heat balances, approach-temperature
      feasibility, utility cost recomputation, area cost recomputation, no
      negative areas, and no active exchanger with impossible temperature
      ordering.
- [x] Keep temporary shadow-run tests while the frozen OpenHENS reference is
      still needed.
- [ ] Remove temporary shadow-run tests once OpenPinch owns the implementation
      and benchmark baselines fully cover the migrated behavior.
- [x] Move long-term documentation, examples, and benchmark data that should
      live in OpenPinch.
- [ ] Archive or freeze the OpenHENS repository only after OpenPinch synthesis
      parity and release availability are confirmed.
- [ ] Add deprecation notes to OpenHENS README and package metadata only after
      the OpenPinch release is available.
- [x] Do not maintain a thin wrapper package.
- [x] Do not add OpenHENS import aliases or runtime compatibility shims.
- [x] Keep the OpenHENS-to-OpenPinch mapping documentation-only.

## General Standards That Apply

- [x] Enforce the required OpenPinch workflow from `README.md`; deviations are
      not permitted.
- [x] Solver regressions stay marked separately from fast default tests.
- [x] Coverage compares network content by stable stream identity.
- [x] Retirement happens after release readiness, not merely after code lands.
- [ ] There is one maintained implementation; users migrate directly to
      OpenPinch-native APIs.

## Verification Checklist

- [ ] Fast test suite passes.
- [ ] Mandatory Four-stream solver test passes on a solver-capable machine.
- [ ] Mandatory Nine-stream final-verification solver test passes before
      OpenHENS retirement.
- [x] Curated Nine-stream best-ESM network snapshot comparison passes before
      OpenHENS retirement.
- [ ] Added tiered solver tests pass or have documented solver-only blockers.
- [x] Numerical invariant tests pass.
- [x] Docs build passes with the migration/retirement documentation.
- [ ] OpenHENS deprecation notes reference an available OpenPinch release.

## Definition of Done

- [ ] Regression coverage is broad enough that future equation changes can be
      reviewed against the Four-stream executable baseline and Nine-stream final
      verification.
- [x] Temporary shadow tests are either still justified or removed with adequate
      OpenPinch-owned baselines.
- [ ] OpenHENS is archived/frozen only after OpenPinch release availability is
      confirmed.
- [x] Users have a documentation-only migration map and no runtime wrapper.
- [ ] The maintained implementation of HEN synthesis is OpenPinch.

## Out of Scope

- Adding new solver algorithms.
- Adding distributed execution.
- Adding multi-utility synthesis beyond current OpenHENS behavior.
- Maintaining OpenHENS as a compatibility wrapper.

## Implementation Notes

- 2026-06-16: Added
  `openhens_baseline_results/network_snapshots/Nine-stream-Linnhoff-and-Ahmad-1999-1/best-esm.json`
  from the checked-in
  `openhens_baseline_results/refactor/Nine-stream-Linnhoff-and-Ahmad-1999-1/results/esm-187ea46995ce.json`
  best ESM artifact and
  `openhens_baseline_results/refactor/Nine-stream-Linnhoff-and-Ahmad-1999-1/summary.json`.
  The snapshot records stream-identity source/sink links, duty, area, unit
  counts, utility loads, and TAC tolerances for the final-verification
  Nine-stream baseline.
- 2026-06-16: Added
  `tests/test_heat_exchanger_network_synthesis_hens11.py` to check the
  Four-stream routine and Nine-stream final-verification baseline summaries,
  best ESM rows, utility loads, verification failures, and curated network
  snapshots. The test decodes checked-in OpenHENS source artifacts by
  source-case stream identity so snapshot comparisons do not depend on raw
  solver axis positions.
- 2026-06-16: Added workbook-independent numerical invariant coverage over the
  checked-in best ESM artifacts: process-stream heat balances, stage recovery
  heat balances, recovery approach-temperature feasibility from labelled
  terminal temperatures, utility-cost recomputation, area/fixed-cost
  recomputation, non-negative areas, and active exchanger temperature-ordering
  checks.
- 2026-06-16: Added `@pytest.mark.solver` regression scaffolding for
  Four-stream and Nine-stream through the OpenPinch `PinchProblem` ->
  `heat_exchanger_network_synthesis_service(..., executor=LocalSynthesisExecutor(...))`
  path. These tests are excluded from fast default runs and require Couenne and
  IPOPT on `PATH`.
- 2026-06-16: `rtk .venv/bin/pytest tests/test_heat_exchanger_network_synthesis_hens11.py -q`
  passed with `6 passed, 2 skipped in 1.45s`; the two skipped tests are the
  solver-marked live baselines.
- 2026-06-16: `rtk .venv/bin/pytest -m "not solver" tests/test_heat_exchanger_network_synthesis_hens11.py tests/test_heat_exchanger_network_synthesis_hens08.py tests/test_heat_exchanger_network_synthesis_models.py tests/test_heat_exchanger_network_synthesis_workflow.py tests/test_heat_exchanger_network_public_service.py tests/test_heat_exchanger_network_array_adapter.py tests/test_heat_exchanger_network_pinch_parity.py tests/test_classes/test_heat_exchanger_network.py tests/test_lib/test_synthesis_schemas.py tests/test_synthesis_dependency_boundaries.py tests/test_package_api_surface.py -q`
  passed with `181 passed, 2 deselected in 26.41s`.
- 2026-06-16: Full fast suite checkbox remains unchecked because this slice ran
  the focused broad HEN-related fast set above rather than the complete
  repository test suite.
- 2026-06-16: `rtk which couenne` exited 1 with no path, and
  `rtk which ipopt` exited 1 with no path. `rtk .venv/bin/pytest -m solver tests/test_heat_exchanger_network_synthesis_hens11.py -q`
  therefore reported `2 skipped, 6 deselected in 1.25s`. Rerun the live solver
  gate on a solver-capable machine with `rtk uv run pytest -m solver` after
  installing Couenne and IPOPT on `PATH`.
- 2026-06-16: `rtk .venv/bin/ruff check tests/test_heat_exchanger_network_synthesis_hens11.py`
  passed.
- 2026-06-16: `rtk .venv/bin/ruff check` passed.
- 2026-06-16: `rtk .venv/bin/python scripts/build_docs.py` passed; Sphinx
  reported the existing 25 warnings for missing heat-pump/autodoc modules.
- 2026-06-16: `rtk git diff --check -- ':!.DS_Store'` passed.
- 2026-06-16: No additional small/medium/large OpenHENS tier cases were added
  because the task set has not named exact additional OpenHENS source paths,
  source hashes, migrated fixture paths, grids, and expected artifact paths
  beyond Four-stream and Nine-stream. This keeps the expanded-tier checkbox
  unchecked.
- 2026-06-16: OpenHENS retirement, archive/freeze, README/package deprecation,
  and release-readiness checkboxes remain unchecked. This environment has not
  proven live OpenPinch Four-stream/Nine-stream solver parity on a
  solver-capable machine and does not confirm an available OpenPinch release
  containing the replacement synthesis API.
