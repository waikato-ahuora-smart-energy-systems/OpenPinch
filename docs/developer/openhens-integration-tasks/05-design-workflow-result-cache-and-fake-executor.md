# HENS-05 Design Workflow, Result Cache, and Fake Executor

## PRD Summary

Create the OpenPinch-owned HEN workflow path, result cache contract, and fake
executor coverage before live solver models move. This task makes
`PinchProblem.design` and workspace dispatch the only public execution path.

## User Outcome

Users can run the HEN synthesis workflow shape from OpenPinch primitives and
receive an in-memory `TargetOutput.design` payload. Maintainers can test task
fan-out, result storage, workspace dispatch, and optional exports without live
solver dependencies.

## Scope

- Workflow orchestration, result envelope, design accessor, workspace dispatch,
  verification shell, optional export shell, and fake executor tests.
- No live GEKKO/Pyomo solver execution required.

## Plan Context

Read these sections before implementation:

- [Target Architecture](../../../OPENHENS_MIGRATION_PLAN.md#target-architecture)
- [Root Primitive Mandate and Parallel Workflow Purge](../../../OPENHENS_MIGRATION_PLAN.md#root-primitive-mandate-and-parallel-workflow-purge)
- [End-to-End Flow Comparison](../../../OPENHENS_MIGRATION_PLAN.md#end-to-end-flow-comparison)
- [Result Envelope Model](../../../OPENHENS_MIGRATION_PLAN.md#result-envelope-model)
- [Phase 5: Workflow, Solver Metadata, Result Cache, and Optional Export Modules](../../../OPENHENS_MIGRATION_PLAN.md#phase-5-workflow-solver-metadata-result-cache-and-optional-export-modules)

Settled decisions for this task:

- `PinchProblem._results` / `problem.results` is the canonical output. Artifact
  folders are optional export views only.
- The public execution path is `problem.design.heat_exchanger_network_synthesis(...)`
  or workspace dispatch to that design path.
- `heat_exchanger_network_synthesis_service(problem)` is internal-facing and
  must not be root-exported or documented as a user-facing execution path.
- HEN synthesis must not be exposed from `problem.target`.
- The service is problem-rooted and must not accept raw stream lists, raw
  utilities, source CSV rows, or public case/study objects.
- Fake executor tests must exercise task fan-out and result-cache semantics
  before live solvers move.
- The required workflow in `README.md` is mandatory; this task is responsible
  for making the design accessor and workspace dispatch follow it.

## Requirements Checklist

- [x] Translate OpenHENS workflow, solver metadata, result, optional export, and
      verification concepts into the OpenPinch synthesis service package with
      OpenPinch-native module and class names.
- [x] Extend `TargetOutput` with optional `design` adjacent to `targets`.
- [x] Define the `TargetOutput.design` payload as the canonical in-memory HEN
      result for one problem state/workflow run.
- [x] Include accepted `HeatExchangerNetwork`, TAC, utility costs, capital
      costs, solver status, method/stage metadata, state id, and task/outcome
      references where needed.
- [x] Add a `PinchProblem.design` accessor consistent with existing `target` and
      `plot` handles.
- [x] Add `PinchProblem.design.heat_exchanger_network_synthesis(...)`.
- [x] Route the design accessor through the internal
      `heat_exchanger_network_synthesis_service(problem)`.
- [x] Make the service problem-rooted and internal-facing: the first required
      argument is a live `PinchProblem`, and it must not be exported as a
      public/root API.
- [x] Preserve existing `TargetResults` when refreshing only HEN design output.
- [x] If targeting has not run, compute the target data needed for HEN synthesis
      and populate both `targets` and `design` in `TargetOutput`.
- [x] Register `heat_exchanger_network_synthesis` as an advanced
      `PinchWorkspace` workflow.
- [x] Extend `run_problem_workflow` so design workflows dispatch to
      `problem.design` without pretending HEN synthesis is a target method.
- [x] Add workspace tests proving `PinchWorkspace.solve_variant(...,
      workflow="heat_exchanger_network_synthesis")` dispatches through the same
      live `PinchProblem` path as direct calls.
- [x] Keep HEN controls in `TargetInput.options` / `Configuration`.
- [x] Keep utility economics, costing configuration, case ownership, and
      variant ownership in `PinchProblem` and `PinchWorkspace`.
- [x] Do not add public `run_synthesis_workflow(...)`.
- [x] Do not expose HEN synthesis from `PinchProblem.target`.
- [x] Keep `LocalSynthesisExecutor` behind optional synthesis dependencies.
- [x] Add fake executor outputs that return
      `HeatExchangerNetworkSynthesisResult` objects containing
      `HeatExchangerNetwork`.
- [x] Convert accepted fake outcomes into `TargetOutput.design` and store them
      on `problem._results`.
- [x] Add fake-executor workflow tests:
      failed PDM does not spawn TDM, failed TDM does not spawn ESM, deterministic
      task ids, topology restrictions are required, and outcomes serialize
      without live solver objects.
- [x] Add result-cache tests proving direct problem and workspace design runs
      populate `problem.results` / `problem._results` as `TargetOutput` with
      `design`.
- [x] Add optional export helpers that write JSON/CSV only when requested from
      `problem.results`.
- [x] Ensure optional exports identify the run by OpenPinch problem or workspace
      variant identity, not OpenHENS case/study identity.

## General Standards That Apply

- [x] Enforce the required OpenPinch workflow from `README.md`; deviations are
      not permitted.
- [x] The primary output is `PinchProblem._results`, not a directory of
      artifacts.
- [x] JSON/CSV export is optional and generated from `problem.results`.
- [x] There is no public raw-input synthesis runner.
- [x] The workflow must be testable without importing GEKKO/Pyomo.
- [x] `PinchProblem` and `PinchWorkspace` remain the only public roots.

## Verification Checklist

- [x] Fake-executor task graph tests pass.
- [x] `TargetOutput` validation and serialization tests pass with both existing
      `targets` and optional `design`.
- [x] Direct `PinchProblem.design` result-cache tests pass.
- [x] `PinchWorkspace.solve_variant(...,
      workflow="heat_exchanger_network_synthesis")` dispatch tests pass.
- [x] Negative API tests prove no `PinchProblem.target` HEN method and no public
      raw-input runner or public/root synthesis service export were added.
- [x] Optional export round-trip tests pass through the export helper path.
- [x] Import smoke tests prove live solver dependencies are not imported.

## Definition of Done

- [ ] OpenPinch can generate, fake-execute, verify, store, and optionally export
      synthesis outcomes without live solver dependencies.
- [ ] `PinchProblem._results` is the canonical result cache for HEN design.
- [ ] Workspace execution uses the existing variant/workflow machinery.
- [ ] No parallel case/study/workspace owner was introduced.
- [ ] The public execution path is OpenPinch-native and problem-rooted.

## Out of Scope

- Live solver execution.
- Moving equation models.
- Replacing duplicate LMTD/costing helpers.
- Visualization.

## Implementation Notes

- 2026-06-16: Added `PinchProblem.design.heat_exchanger_network_synthesis(...)`,
  internal `heat_exchanger_network_synthesis_service(problem)`, OpenPinch-native
  workflow/fake executor, verification shell, and optional JSON/CSV exports from
  `problem.results`. The service remains absent from `OpenPinch.__all__` and
  `OpenPinch.services.__all__`.
- 2026-06-16: Added task graph tests covering failed PDM not spawning TDM,
  failed TDM not spawning ESM, deterministic task IDs, required topology
  restrictions, and JSON serialization without live solver objects.
- 2026-06-16: Added direct and workspace cache tests proving `TargetOutput.design`
  is stored on `problem._results` while preserving existing `TargetResults` or
  computing targets when the cache is empty.
- 2026-06-16: Added optional export round-trip tests for `manifest.json`,
  `results/<task_id>.json`, `metrics/solution_metrics.csv`, and
  `metrics/run_summary.csv`, keyed by OpenPinch problem/workspace variant
  identity.
- 2026-06-16: `rtk uv run pytest tests/test_heat_exchanger_network_synthesis_workflow.py tests/test_lib/test_synthesis_schemas.py tests/test_classes/test_heat_exchanger_network.py tests/test_synthesis_dependency_boundaries.py tests/test_package_api_surface.py tests/test_classes/test_pinch_workspace.py -q` passed: 54 passed in 140.66s.
- 2026-06-16: `rtk uv run ruff check OpenPinch tests/test_heat_exchanger_network_synthesis_workflow.py` passed: All checks passed.
- 2026-06-16: `rtk git diff --check -- . ':!.DS_Store'` passed with no output;
  root `.DS_Store` remains pre-existing/user-owned and was not touched.
- 2026-06-16: Definition of Done intentionally left unchecked pending
  adversarial review.
- 2026-06-16: Re-review blocker resolved by renaming the internal workflow
  runner from public-looking `run_synthesis_workflow(...)` to private
  `_execute_synthesis_workflow(...)`; negative tests now assert the banned name
  is absent from the synthesis package and workflow module.
- 2026-06-16: `rtk uv run pytest tests/test_heat_exchanger_network_synthesis_workflow.py tests/test_package_api_surface.py tests/test_lib/test_synthesis_schemas.py -q` passed: 37 passed in 3.54s.
- 2026-06-16: `rtk uv run ruff check OpenPinch tests/test_heat_exchanger_network_synthesis_workflow.py` passed: All checks passed.
- 2026-06-16: `rtk git diff --check -- . ':!.DS_Store'` passed with no output
  after the re-review blocker fix.
