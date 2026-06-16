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

- [x] Move stage-reduction logic into the OpenPinch synthesis models package.
- [x] Move topology evolution logic into the OpenPinch synthesis models package.
- [x] Keep heuristic ordering unchanged.
- [x] Keep stopping conditions unchanged.
- [x] Keep objective and tie-breaking behavior unchanged.
- [x] Keep solver defaults and tolerances unchanged.
- [x] Preserve deterministic task ids and parent-child task references.
- [x] Preserve outcome serialization without live solver objects.
- [x] Convert all final accepted networks into `HeatExchangerNetwork` records.
- [x] Preserve diagnostic raw arrays only where parity tests still need them.
- [x] Add tests for stage count, recovery unit count, hot utility unit count,
      cold utility unit count, best `dTmin`, and best derivative threshold.
- [x] Add tests that compare `HeatExchangerNetwork` content by stream identity,
      including source/sink/stage/duty/area for recovery exchangers and utility
      links.
- [x] Compare the curated best-ESM network snapshot for Four-stream once HENS-00
      defines the snapshot artifact and this task creates or refreshes it.
- [x] Add solver-regression evidence for Four-stream workbook baseline using
      `openhens_baseline_results/refactor/Four-stream-Yee-and-Grossmann-1990-1/summary.json`.
- [x] Do not require Nine-stream solver evidence in this task unless this task is
      also being used as the final verification gate.
- [x] Document any blocked solver run with exact missing dependency or solver
      binary error.
- [ ] Remove remaining OpenHENS runtime package dependency from the synthesis
      execution path once parity is proven.

## General Standards That Apply

- [x] Enforce the required OpenPinch workflow from `README.md`; deviations are
      not permitted.
- [x] This is still a behavior-preserving move.
- [x] No new algorithms, distributed execution, multi-utility expansion, or
      solver-default changes.
- [x] Network comparison must use OpenPinch stream identities, not only raw
      positional arrays.
- [x] Public workflow remains `PinchProblem.design` and workspace dispatch.

## Verification Checklist

- [ ] Four-stream solver baseline matches within the accepted tolerances.
- [x] Nine-stream final verification is either explicitly out of scope for this
      task or run only when this task is the final migration gate.
- [x] Fast serialization, task graph, and result-cache tests pass.
- [x] Network-by-stream-identity comparisons pass.
- [x] Best-ESM network snapshot comparison passes for Four-stream once the
      snapshot artifact exists.
- [x] Core import smoke tests still pass without synthesis dependencies.

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

- Source OpenHENS verified before implementation: `rtk git -C /Users/ca107/Desktop/ahuora/OpenHENS rev-parse HEAD` returned `2afc14b7779482fc829edb1c3fa187b918d7fb19`.
- Moved HENS-08 behavior into OpenPinch-owned internals only: `InternalHeatExchangerNetworkProblem.get_solution(...)` now applies source stage-utilisation reduction after successful PDM/TDM solves, and `StageWiseModel` owns source topology evolution methods (`get_net_benefit_evolution`, plus/minus candidate construction, candidate selection, and benefit-ranking helpers). `LocalSynthesisExecutor` enables evolution only for `energy_stage_refinement` tasks while keeping PDM/TDM unchanged; the executor-wide evolution flag cannot enable evolution for PDM/TDM.
- Final accepted outcomes still flow through `HeatExchangerNetwork` extraction and `TargetOutput.design`; no public OpenHENS-style case/study/runner route was added.
- Created curated snapshot `openhens_baseline_results/network_snapshots/Four-stream-Yee-and-Grossmann-1990-1/best-esm.json` from `openhens_baseline_results/refactor/Four-stream-Yee-and-Grossmann-1990-1/results/esm-b520e740f0d6.json`. The snapshot records best TAC `154853.8518602861`, `dTmin=14.0`, derivative threshold `0.5`, 3 stages, 3 recovery units, 1 hot utility unit, 2 cold utility units, utility loads, identity-labelled exchanger links, and per-field tolerances.
- `tests/test_heat_exchanger_network_synthesis_hens08.py` reconstructs the best ESM artifact through the OpenPinch adapter/extraction path and compares recovery and utility links by source/sink stream identity, stage, duty, and area. It also cross-checks counts, best `dTmin`, and best derivative threshold against `openhens_baseline_results/refactor/Four-stream-Yee-and-Grossmann-1990-1/summary.json`.
- Focused HENS-08/HENS-07 evidence after re-review fixes: `rtk uv run pytest tests/test_heat_exchanger_network_synthesis_models.py tests/test_heat_exchanger_network_synthesis_hens08.py` -> `18 passed`. This includes `test_local_executor_preserves_parent_links_and_esm_only_evolution`, which runs `LocalSynthesisExecutor(evolution=True)` and asserts PDM/TDM pass `evolution=False` while ESM passes `evolution=True`.
- Relevant HENS-05/HENS-07/import/API evidence after re-review fixes: `rtk uv run pytest tests/test_heat_exchanger_network_synthesis_workflow.py tests/test_heat_exchanger_network_array_adapter.py tests/test_synthesis_dependency_boundaries.py tests/test_package_api_surface.py` -> `24 passed`.
- OpenPinch solver marker check: `rtk uv run pytest -m solver --collect-only` selected 0 tests (`1111 deselected / 0 selected`), so this repo does not yet provide a migrated marked solver gate for HENS-08.
- Live Four-stream solver rerun is blocked in this environment because `rtk which couenne` and `rtk which ipopt` both exited 1 with no binary on PATH. Exact rerun command once binaries are available: from `/Users/ca107/Desktop/ahuora/OpenHENS`, run `rtk uv run pytest 'openhens/tests/test_regression.py::test_solver_regression_matches_saved_workbook_baselines[Four-stream-Yee-and-Grossmann-1990-1]'`.
- Dependency-boundary scan `rtk rg -n "OpenHENS|openhens|CaseStudy|SynthesisStudy|run_synthesis_workflow" OpenPinch/services/heat_exchanger_network_synthesis OpenPinch/classes OpenPinch/lib/schemas tests/test_heat_exchanger_network_synthesis_hens08.py tests/test_heat_exchanger_network_synthesis_models.py` found no OpenHENS public runtime imports in the currently moved synthesis execution code; remaining hits are source-reference comments/diagnostics and source-comparison tests. This is boundary evidence only: the runtime-dependency removal requirement and DoD remain unchecked until moved OpenPinch Four-stream parity is proven.
- Quality checks after re-review fixes: `rtk uv run ruff check` -> `All checks passed!`; `rtk git diff --check -- . ':!.DS_Store'` -> clean.
- Definition of Done remains unchecked pending re-review. The live Four-stream solver rerun is blocked by missing external solver binaries; that documents the parity blocker but is not completion evidence for runtime dependency removal.
