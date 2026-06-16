# HENS-07 Stagewise and PDM Model Move

## PRD Summary

Move the StageWise TDM/ESM model construction and PDM coordinator behind the
OpenPinch synthesis service boundary while preserving the source OpenHENS
algorithm and downstream task behavior.

## User Outcome

OpenPinch can execute the core PDM -> TDM -> ESM model progression through the
new problem-rooted synthesis workflow, with parity protection around topology
generation and stagewise restrictions.

## Scope

- Stagewise TDM/ESM equation construction.
- PDM coordinator and above/below-pinch construction.
- Solver task construction and parent/topology handoff.
- Solver-regression evidence for moved slices.
- No helper formula replacement unless already proven by earlier parity tasks.

## Plan Context

Read these sections before implementation:

- [End-to-End Flow Comparison](../../../OPENHENS_MIGRATION_PLAN.md#end-to-end-flow-comparison)
- [Phase 4: Pinch Target Parity and Replacement Plan](../../../OPENHENS_MIGRATION_PLAN.md#phase-4-pinch-target-parity-and-replacement-plan)
- [Phase 5: Workflow, Solver Metadata, Result Cache, and Optional Export Modules](../../../OPENHENS_MIGRATION_PLAN.md#phase-5-workflow-solver-metadata-result-cache-and-optional-export-modules)
- [Phase 6: Move Existing Equation Models Behind the New Service Boundary](../../../OPENHENS_MIGRATION_PLAN.md#phase-6-move-existing-equation-models-behind-the-new-service-boundary)
- [Recommended Review Slices](../../../OPENHENS_MIGRATION_PLAN.md#recommended-review-slices)

Settled decisions for this task:

- PDM/TDM/ESM semantics are preserved scientifically but not as public OpenHENS
  architecture.
- Downstream TDM and ESM task construction must preserve the solved
  parent/topology context needed by source behavior.
- Use Four-stream as the solver baseline evidence for routine moved-slice
  validation because it is the quicker benchmark.
- Do not promote Nine-stream to a routine moved-slice gate; keep it for final
  verification unless a specific regression requires it.
- OpenPinch `ProblemTable` replacement is allowed only where HENS-04 parity is
  already green.
- LMTD, costing, visualization, and helper cleanup belong in separate tasks
  unless already covered by their own parity gates.
- The required workflow in `README.md` is mandatory; moved PDM/TDM/ESM logic
  must be invoked through that path only.

## Requirements Checklist

- [ ] Move StageWise TDM/ESM equation construction into the OpenPinch synthesis
      models package.
- [ ] Move the PDM coordinator into the OpenPinch synthesis models package.
- [ ] Keep equations, variable names, constraints, objectives, and solver
      defaults unchanged.
- [ ] Preserve above-pinch and below-pinch construction behavior.
- [ ] Preserve manual stage selection behavior.
- [ ] Preserve hot/cold threshold behavior where utility targets are zero.
- [ ] Preserve topology restriction semantics from successful PDM outcomes into
      TDM tasks.
- [ ] Preserve topology restriction semantics from successful TDM outcomes into
      ESM refinement tasks.
- [ ] Preserve solved parent/topology/problem state needed by downstream
      TDM/ESM warm-start or restriction construction. Do not rebuild downstream
      model state from cold defaults unless parity evidence proves equivalence.
- [ ] Ensure task IDs remain deterministic across PDM, TDM, and ESM fan-out.
- [ ] Ensure failed PDM tasks do not spawn TDM tasks.
- [ ] Ensure failed TDM tasks do not spawn ESM tasks.
- [ ] Convert every accepted solved topology into `HeatExchangerNetwork`.
- [ ] Keep raw solver arrays private and diagnostic-only.
- [ ] Use OpenPinch `ProblemTable` / targeting data only where HENS-04 parity
      already passed.
- [ ] Keep duplicate LMTD and costing formulas unchanged unless HENS-10 has
      already replaced them in a separate PR.
- [ ] Add focused tests for topology restriction requirements.
- [ ] Add focused tests for above/below-pinch structural fields.
- [ ] Add Four-stream solver-regression or shadow-run evidence for each moved
      sub-slice.

## General Standards That Apply

- [ ] Enforce the required OpenPinch workflow from `README.md`; deviations are
      not permitted.
- [ ] Move behavior first; refactor later.
- [ ] Do not widen tolerances in this task.
- [ ] Do not make raw arrays public.
- [ ] Do not introduce an OpenHENS-style public workflow root.
- [ ] Keep solver imports optional and lazy.

## Verification Checklist

- [ ] Fast task-graph tests pass.
- [ ] Structural PDM parity tests pass.
- [ ] Four-stream solver or shadow-run tests pass for the moved PDM slice.
- [ ] Four-stream solver or shadow-run tests pass for the moved TDM/ESM
      stagewise slice.
- [ ] Import smoke tests still prove normal OpenPinch imports are lightweight.
- [ ] Existing result-cache tests still pass.

## Definition of Done

- [ ] PDM, TDM, and ESM model construction is reachable only through the
      OpenPinch problem-rooted synthesis boundary.
- [ ] Downstream TDM/ESM construction receives the same solved parent/topology
      context as the source behavior.
- [ ] Task fan-out and deterministic task IDs are covered by tests.
- [ ] Benchmark or shadow evidence shows no unintended solver behavior drift for
      the moved slices.
- [ ] No helper cleanup, visualization, public aliasing, or CSV runtime support
      was mixed into this task.

## Out of Scope

- Stage-reduction and topology evolution logic unless required by moved model
  construction.
- Public service documentation.
- OpenHENS repository retirement.
- New HEN algorithms or multi-utility generalization.

## Implementation Notes

- 
