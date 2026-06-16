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

- [ ] Translate OpenHENS workflow, solver metadata, result, optional export, and
      verification concepts into the OpenPinch synthesis service package with
      OpenPinch-native module and class names.
- [ ] Extend `TargetOutput` with optional `design` adjacent to `targets`.
- [ ] Define the `TargetOutput.design` payload as the canonical in-memory HEN
      result for one problem state/workflow run.
- [ ] Include accepted `HeatExchangerNetwork`, TAC, utility costs, capital
      costs, solver status, method/stage metadata, state id, and task/outcome
      references where needed.
- [ ] Add a `PinchProblem.design` accessor consistent with existing `target` and
      `plot` handles.
- [ ] Add `PinchProblem.design.heat_exchanger_network_synthesis(...)`.
- [ ] Route the design accessor through the internal
      `heat_exchanger_network_synthesis_service(problem)`.
- [ ] Make the service problem-rooted and internal-facing: the first required
      argument is a live `PinchProblem`, and it must not be exported as a
      public/root API.
- [ ] Preserve existing `TargetResults` when refreshing only HEN design output.
- [ ] If targeting has not run, compute the target data needed for HEN synthesis
      and populate both `targets` and `design` in `TargetOutput`.
- [ ] Register `heat_exchanger_network_synthesis` as an advanced
      `PinchWorkspace` workflow.
- [ ] Extend `run_problem_workflow` so design workflows dispatch to
      `problem.design` without pretending HEN synthesis is a target method.
- [ ] Add workspace tests proving `PinchWorkspace.solve_variant(...,
      workflow="heat_exchanger_network_synthesis")` dispatches through the same
      live `PinchProblem` path as direct calls.
- [ ] Keep HEN controls in `TargetInput.options` / `Configuration`.
- [ ] Keep utility economics, costing configuration, case ownership, and
      variant ownership in `PinchProblem` and `PinchWorkspace`.
- [ ] Do not add public `run_synthesis_workflow(...)`.
- [ ] Do not expose HEN synthesis from `PinchProblem.target`.
- [ ] Keep `LocalSynthesisExecutor` behind optional synthesis dependencies.
- [ ] Add fake executor outputs that return
      `HeatExchangerNetworkSynthesisResult` objects containing
      `HeatExchangerNetwork`.
- [ ] Convert accepted fake outcomes into `TargetOutput.design` and store them
      on `problem._results`.
- [ ] Add fake-executor workflow tests:
      failed PDM does not spawn TDM, failed TDM does not spawn ESM, deterministic
      task ids, topology restrictions are required, and outcomes serialize
      without live solver objects.
- [ ] Add result-cache tests proving direct problem and workspace design runs
      populate `problem.results` / `problem._results` as `TargetOutput` with
      `design`.
- [ ] Add optional export helpers that write JSON/CSV only when requested from
      `problem.results`.
- [ ] Ensure optional exports identify the run by OpenPinch problem or workspace
      variant identity, not OpenHENS case/study identity.

## General Standards That Apply

- [ ] Enforce the required OpenPinch workflow from `README.md`; deviations are
      not permitted.
- [ ] The primary output is `PinchProblem._results`, not a directory of
      artifacts.
- [ ] JSON/CSV export is optional and generated from `problem.results`.
- [ ] There is no public raw-input synthesis runner.
- [ ] The workflow must be testable without importing GEKKO/Pyomo.
- [ ] `PinchProblem` and `PinchWorkspace` remain the only public roots.

## Verification Checklist

- [ ] Fake-executor task graph tests pass.
- [ ] `TargetOutput` validation and serialization tests pass with both existing
      `targets` and optional `design`.
- [ ] Direct `PinchProblem.design` result-cache tests pass.
- [ ] `PinchWorkspace.solve_variant(...,
      workflow="heat_exchanger_network_synthesis")` dispatch tests pass.
- [ ] Negative API tests prove no `PinchProblem.target` HEN method and no public
      raw-input runner or public/root synthesis service export were added.
- [ ] Optional export round-trip tests pass through the export helper path.
- [ ] Import smoke tests prove live solver dependencies are not imported.

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

- 
