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

- [x] Move StageWise TDM/ESM equation construction into the OpenPinch synthesis
      models package.
- [x] Move the PDM coordinator into the OpenPinch synthesis models package.
- [x] Keep equations, variable names, constraints, objectives, and solver
      defaults unchanged.
- [x] Preserve above-pinch and below-pinch construction behavior.
- [x] Preserve manual stage selection behavior.
- [x] Preserve hot/cold threshold behavior where utility targets are zero.
- [x] Preserve topology restriction semantics from successful PDM outcomes into
      TDM tasks.
- [x] Preserve topology restriction semantics from successful TDM outcomes into
      ESM refinement tasks.
- [x] Preserve solved parent/topology/problem state needed by downstream
      TDM/ESM warm-start or restriction construction. Do not rebuild downstream
      model state from cold defaults unless parity evidence proves equivalence.
- [x] Ensure task IDs remain deterministic across PDM, TDM, and ESM fan-out.
- [x] Ensure failed PDM tasks do not spawn TDM tasks.
- [x] Ensure failed TDM tasks do not spawn ESM tasks.
- [x] Convert every accepted solved topology into `HeatExchangerNetwork`.
- [x] Keep raw solver arrays private and diagnostic-only.
- [x] Use OpenPinch `ProblemTable` / targeting data only where HENS-04 parity
      already passed.
- [x] Keep duplicate LMTD and costing formulas unchanged unless HENS-10 has
      already replaced them in a separate PR.
- [x] Add focused tests for topology restriction requirements.
- [x] Add focused tests for above/below-pinch structural fields.
- [x] Add Four-stream solver-regression or shadow-run evidence for each moved
      sub-slice.

## General Standards That Apply

- [x] Enforce the required OpenPinch workflow from `README.md`; deviations are
      not permitted.
- [x] Move behavior first; refactor later.
- [x] Do not widen tolerances in this task.
- [x] Do not make raw arrays public.
- [x] Do not introduce an OpenHENS-style public workflow root.
- [x] Keep solver imports optional and lazy.

## Verification Checklist

- [x] Fast task-graph tests pass.
- [x] Structural PDM parity tests pass.
- [x] Four-stream solver or shadow-run tests pass for the moved PDM slice.
- [x] Four-stream solver or shadow-run tests pass for the moved TDM/ESM
      stagewise slice.
- [x] Import smoke tests still prove normal OpenPinch imports are lightweight.
- [x] Existing result-cache tests still pass.

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

- Source verification: `rtk git -C /Users/ca107/Desktop/ahuora/OpenHENS rev-parse HEAD`
  returned `2afc14b7779482fc829edb1c3fa187b918d7fb19`, matching the expected
  source commit for parity comparison.
- Moved private model construction into
  `OpenPinch/services/heat_exchanger_network_synthesis/models/stagewise.py` and
  `OpenPinch/services/heat_exchanger_network_synthesis/models/pinch_decomposition.py`.
  The HENS-07 path keeps GEKKO/Pyomo loading behind solver factory calls, keeps
  raw arrays inside private problem/model objects, and intentionally gates
  stage reduction/topology evolution with `ModelSliceUnavailableError` for
  HENS-08.
- Restored the internal solve path through
  `InternalHeatExchangerNetworkProblem` and the private `LocalSynthesisExecutor`
  only behind the existing problem-rooted synthesis service boundary. The local
  executor preserves deterministic task fan-out, prevents failed PDM outcomes
  from spawning TDM tasks, prevents failed TDM outcomes from spawning ESM tasks,
  and carries the solved private parent problem into downstream TDM/ESM
  construction.
- Preserved source PDM construction semantics using the HENS-04 pinch snapshot:
  above/below pinch stream clipping, zero-utility threshold skips, manual
  above/below stage selection, utility target handoff, and topology restrictions
  are covered by Four-stream structural shadow tests against the source
  OpenHENS model.
- Re-review blocker resolution: added explicit Four-stream ESM StageWise
  structural shadow coverage against the source OpenHENS model. The ESM test
  constructs the moved non-isothermal `StageWiseModel` with
  `framework="ESM"`, `integers=False`, and
  `minimisation_goal="variable total cost"`, then compares the source dimensions,
  stream heat totals, `Q_max`, feasibility masks, GEKKO equation/objective
  counts, common recovery variables, ESM-only `X`/`Y` split variables, split
  outlet temperature variables, and variable-total-cost intermediates.
- Re-review blocker resolution: extended the local executor parent-preservation
  test through the full PDM -> TDM -> ESM task chain. The test now builds ESM
  tasks from successful TDM outcomes and asserts the private parent problem chain
  is `TDM parent == PDM` and `ESM parent == TDM`, covering the warm-start context
  handoff that source OpenHENS requires.
- Re-review docs blocker resolution: updated the `BaseHeatExchangerNetworkModel`
  docstring to describe the stable HENS-07 private contract. The base layer now
  documents ownership of guarded GEKKO setup, source-shaped solver-array
  normalization, inherited topology restrictions, common diagnostics, and shared
  helper equations for the moved private `PinchDecompModel` and
  `StageWiseModel`, while clearly leaving topology evolution and stage reduction
  to HENS-08.
- Verification commands and results for the docs blocker:
  `rtk .venv/bin/ruff check OpenPinch/services/heat_exchanger_network_synthesis/models/base.py`
  passed; `rtk .venv/bin/python -m py_compile OpenPinch/services/heat_exchanger_network_synthesis/models/base.py`
  passed; `rtk git diff --check -- . ':(exclude).DS_Store'` passed.
- Solver defaults now match the source model defaults for this moved slice:
  PDM/TDM use `couenne`; ESM uses `ipopt-pyomo`.
- Verification commands and results:
  `rtk .venv/bin/python -m py_compile OpenPinch/services/heat_exchanger_network_synthesis/models/base.py OpenPinch/services/heat_exchanger_network_synthesis/models/stagewise.py OpenPinch/services/heat_exchanger_network_synthesis/models/pinch_decomposition.py OpenPinch/services/heat_exchanger_network_synthesis/models/problem.py OpenPinch/services/heat_exchanger_network_synthesis/workflow.py`
  passed.
- Verification commands and results:
  `rtk .venv/bin/ruff format OpenPinch/services/heat_exchanger_network_synthesis/models/base.py OpenPinch/services/heat_exchanger_network_synthesis/models/stagewise.py OpenPinch/services/heat_exchanger_network_synthesis/models/pinch_decomposition.py OpenPinch/services/heat_exchanger_network_synthesis/models/problem.py OpenPinch/services/heat_exchanger_network_synthesis/models/__init__.py OpenPinch/services/heat_exchanger_network_synthesis/workflow.py OpenPinch/lib/config_metadata.py tests/test_heat_exchanger_network_synthesis_models.py`
  reformatted seven files; `rtk .venv/bin/ruff check --fix OpenPinch/services/heat_exchanger_network_synthesis/workflow.py tests/test_heat_exchanger_network_synthesis_models.py`
  fixed two import-order issues; final
  `rtk .venv/bin/ruff check OpenPinch/services/heat_exchanger_network_synthesis/models/base.py OpenPinch/services/heat_exchanger_network_synthesis/models/stagewise.py OpenPinch/services/heat_exchanger_network_synthesis/models/pinch_decomposition.py OpenPinch/services/heat_exchanger_network_synthesis/models/problem.py OpenPinch/services/heat_exchanger_network_synthesis/models/__init__.py OpenPinch/services/heat_exchanger_network_synthesis/workflow.py OpenPinch/lib/config_metadata.py tests/test_heat_exchanger_network_synthesis_models.py`
  passed.
- Verification commands and results:
  `rtk .venv/bin/pytest tests/test_heat_exchanger_network_synthesis_models.py -q`
  passed with `17 passed`; this includes Four-stream TDM StageWise structural
  parity, Four-stream ESM StageWise structural parity, Four-stream PDM
  above/below structural parity, manual PDM stage selection parity, missing PDM
  solver-binary handling, internal parent-context construction, full
  PDM -> TDM -> ESM local executor parent preservation, and result
  serialization without solver arrays.
- Verification commands and results:
  `rtk .venv/bin/pytest tests/test_heat_exchanger_network_synthesis_workflow.py -q`
  passed with `10 passed`; this includes deterministic task IDs, task fan-out,
  topology restriction handoff, failed-task gating, result-cache behavior, and
  API-boundary assertions.
- Verification commands and results:
  `rtk .venv/bin/pytest tests/test_synthesis_dependency_boundaries.py -q`
  passed with `3 passed`; `rtk .venv/bin/pytest tests/test_lib/test_synthesis_schemas.py tests/test_lib/test_config_enums.py tests/test_package_api_surface.py -q`
  passed with `34 passed`; `rtk .venv/bin/pytest tests/test_heat_exchanger_network_array_adapter.py -q`
  passed with `9 passed`; `rtk .venv/bin/pytest tests/test_heat_exchanger_network_pinch_parity.py -q`
  passed with `83 passed`.
- Lightweight import evidence:
  `rtk .venv/bin/python -c "import sys; import OpenPinch; from OpenPinch.services.heat_exchanger_network_synthesis.workflow import LocalSynthesisExecutor; from OpenPinch.services.heat_exchanger_network_synthesis.models import StageWiseModel, PinchDecompModel; forbidden=['gekko','pyomo','pyomo.environ','matplotlib','plotly','openpyxl']; print('loaded', [name for name in forbidden if name in sys.modules])"`
  printed `loaded []`.
- Solver-marked full solves were not run locally because `rtk which couenne`
  and `rtk which ipopt` both exited `1` with no binary on PATH. The added
  missing-binary test covers the required failure mode; rerun full solver tests
  in an environment with Couenne and IPOPT available.
- Whitespace verification:
  `rtk git diff --check -- . ':(exclude).DS_Store'` passed. The root
  `.DS_Store` was pre-existing/user-owned and was not edited by this task.
- Definition of Done checkboxes are left unchecked pending the requested
  adversarial review and the follow-up re-review.
