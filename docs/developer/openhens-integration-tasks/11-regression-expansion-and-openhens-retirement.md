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

- [ ] Keep Four-stream solver regression as the mandatory routine baseline.
- [ ] Run Nine-stream solver regression as the mandatory final-verification
      baseline before OpenHENS retirement.
- [ ] Add marked solver tests for additional OpenHENS example cases in tiers:
      small cases first, then medium cases, then large benchmark cases.
- [ ] For every solver baseline, compare best ESM TAC.
- [ ] Compare quartiles.
- [ ] Compare within-2/5/10 percent counts.
- [ ] Compare attempted jobs.
- [ ] Compare solved ESM count.
- [ ] Compare best stage count.
- [ ] Compare recovery unit count.
- [ ] Compare hot utility unit count.
- [ ] Compare cold utility unit count.
- [ ] Compare best `dTmin`.
- [ ] Compare best derivative threshold.
- [ ] Compare hot and cold utility loads.
- [ ] Compare verification failures.
- [ ] Compare `HeatExchangerNetwork` content by stream identity:
      recovery exchanger source/sink/stage/duty/area, hot utility links, cold
      utility links, utility loads, and unit counts.
- [ ] Compare curated best-ESM network snapshots for Four-stream and Nine-stream
      before retirement.
- [ ] Add numerical invariant tests that do not depend on benchmark workbooks:
      stream heat balances, stage heat balances, approach-temperature
      feasibility, utility cost recomputation, area cost recomputation, no
      negative areas, and no active exchanger with impossible temperature
      ordering.
- [ ] Keep temporary shadow-run tests while the frozen OpenHENS reference is
      still needed.
- [ ] Remove temporary shadow-run tests once OpenPinch owns the implementation
      and benchmark baselines fully cover the migrated behavior.
- [ ] Move long-term documentation, examples, and benchmark data that should
      live in OpenPinch.
- [ ] Archive or freeze the OpenHENS repository only after OpenPinch synthesis
      parity and release availability are confirmed.
- [ ] Add deprecation notes to OpenHENS README and package metadata only after
      the OpenPinch release is available.
- [ ] Do not maintain a thin wrapper package.
- [ ] Do not add OpenHENS import aliases or runtime compatibility shims.
- [ ] Keep the OpenHENS-to-OpenPinch mapping documentation-only.

## General Standards That Apply

- [ ] Enforce the required OpenPinch workflow from `README.md`; deviations are
      not permitted.
- [ ] Solver regressions stay marked separately from fast default tests.
- [ ] Coverage compares network content by stable stream identity.
- [ ] Retirement happens after release readiness, not merely after code lands.
- [ ] There is one maintained implementation; users migrate directly to
      OpenPinch-native APIs.

## Verification Checklist

- [ ] Fast test suite passes.
- [ ] Mandatory Four-stream solver test passes on a solver-capable machine.
- [ ] Mandatory Nine-stream final-verification solver test passes before
      OpenHENS retirement.
- [ ] Curated Nine-stream best-ESM network snapshot comparison passes before
      OpenHENS retirement.
- [ ] Added tiered solver tests pass or have documented solver-only blockers.
- [ ] Numerical invariant tests pass.
- [ ] Docs build passes with the migration/retirement documentation.
- [ ] OpenHENS deprecation notes reference an available OpenPinch release.

## Definition of Done

- [ ] Regression coverage is broad enough that future equation changes can be
      reviewed against the Four-stream executable baseline and Nine-stream final
      verification.
- [ ] Temporary shadow tests are either still justified or removed with adequate
      OpenPinch-owned baselines.
- [ ] OpenHENS is archived/frozen only after OpenPinch release availability is
      confirmed.
- [ ] Users have a documentation-only migration map and no runtime wrapper.
- [ ] The maintained implementation of HEN synthesis is OpenPinch.

## Out of Scope

- Adding new solver algorithms.
- Adding distributed execution.
- Adding multi-utility synthesis beyond current OpenHENS behavior.
- Maintaining OpenHENS as a compatibility wrapper.

## Implementation Notes

- 
