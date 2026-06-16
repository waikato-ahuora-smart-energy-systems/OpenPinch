# HENS-08 Stage Reduction and Topology Evolution Move

## PRD Summary

Move the remaining stage-reduction and topology-evolution behavior into the
OpenPinch synthesis implementation, preserving existing OpenHENS heuristics and
solver results.

## User Outcome

The migrated OpenPinch synthesis solver can reproduce the full OpenHENS
workflow, including model refinement and evolution behavior, without depending
on the OpenHENS runtime package.

## Scope

- Stage-reduction logic.
- Topology evolution logic.
- End-to-end solver regression through the moved OpenPinch implementation.
- No public API expansion beyond already approved OpenPinch-native names.

## Plan Context

Read these sections before implementation:

- [Design Principles](../../../OPENHENS_MIGRATION_PLAN.md#design-principles)
- [Phase 6: Move Existing Equation Models Behind the New Service Boundary](../../../OPENHENS_MIGRATION_PLAN.md#phase-6-move-existing-equation-models-behind-the-new-service-boundary)
- [Phase 9: Expand Regression Coverage](../../../OPENHENS_MIGRATION_PLAN.md#phase-9-expand-regression-coverage)
- [Regression Tolerances](../../../OPENHENS_MIGRATION_PLAN.md#regression-tolerances)
- [Non-Goals for the First Migration](../../../OPENHENS_MIGRATION_PLAN.md#non-goals-for-the-first-migration)

Settled decisions for this task:

- Stage-reduction and topology-evolution heuristics move unchanged.
- Four-stream is the required baseline for this moved-workflow task unless a
  solver-capability blocker is documented exactly.
- Nine-stream remains final verification at the end of the migration and should
  not be required for this routine solver-moving task.
- Final accepted networks must be compared by OpenPinch stream identity, not
  only by positional arrays.
- Removing the OpenHENS runtime dependency is allowed only after parity through
  the moved OpenPinch path is proven.
- The required workflow in `README.md` is mandatory; stage reduction and topology
  evolution must not expose an alternate public execution route.

## Requirements Checklist

- [ ] Move stage-reduction logic into the OpenPinch synthesis models package.
- [ ] Move topology evolution logic into the OpenPinch synthesis models package.
- [ ] Keep heuristic ordering unchanged.
- [ ] Keep stopping conditions unchanged.
- [ ] Keep objective and tie-breaking behavior unchanged.
- [ ] Keep solver defaults and tolerances unchanged.
- [ ] Preserve deterministic task ids and parent-child task references.
- [ ] Preserve outcome serialization without live solver objects.
- [ ] Convert all final accepted networks into `HeatExchangerNetwork` records.
- [ ] Preserve diagnostic raw arrays only where parity tests still need them.
- [ ] Add tests for stage count, recovery unit count, hot utility unit count,
      cold utility unit count, best `dTmin`, and best derivative threshold.
- [ ] Add tests that compare `HeatExchangerNetwork` content by stream identity,
      including source/sink/stage/duty/area for recovery exchangers and utility
      links.
- [ ] Compare the curated best-ESM network snapshot for Four-stream once HENS-00
      defines the snapshot artifact and this task creates or refreshes it.
- [ ] Add solver-regression evidence for Four-stream workbook baseline using
      `openhens_baseline_results/refactor/Four-stream-Yee-and-Grossmann-1990-1/summary.json`.
- [ ] Do not require Nine-stream solver evidence in this task unless this task is
      also being used as the final verification gate.
- [ ] Document any blocked solver run with exact missing dependency or solver
      binary error.
- [ ] Remove remaining OpenHENS runtime package dependency from the synthesis
      execution path once parity is proven.

## General Standards That Apply

- [ ] Enforce the required OpenPinch workflow from `README.md`; deviations are
      not permitted.
- [ ] This is still a behavior-preserving move.
- [ ] No new algorithms, distributed execution, multi-utility expansion, or
      solver-default changes.
- [ ] Network comparison must use OpenPinch stream identities, not only raw
      positional arrays.
- [ ] Public workflow remains `PinchProblem.design` and workspace dispatch.

## Verification Checklist

- [ ] Four-stream solver baseline matches within the accepted tolerances.
- [ ] Nine-stream final verification is either explicitly out of scope for this
      task or run only when this task is the final migration gate.
- [ ] Fast serialization, task graph, and result-cache tests pass.
- [ ] Network-by-stream-identity comparisons pass.
- [ ] Best-ESM network snapshot comparison passes for Four-stream once the
      snapshot artifact exists.
- [ ] Core import smoke tests still pass without synthesis dependencies.

## Definition of Done

- [ ] The full moved OpenPinch synthesis path can reproduce the source OpenHENS
      workflow for the Four-stream baseline.
- [ ] Stage-reduction and topology evolution behavior has focused regression
      evidence.
- [ ] The runtime synthesis path no longer depends on OpenHENS public runtime
      APIs.
- [ ] All accepted results are available through `TargetOutput.design`.
- [ ] Any unavailable solver evidence is precisely documented as a blocker.

## Out of Scope

- Replacing duplicate helper formulas.
- Adding public migration docs.
- Archiving OpenHENS.
- Adding new solver backends.

## Implementation Notes

- 
