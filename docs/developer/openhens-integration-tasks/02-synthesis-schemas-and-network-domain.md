# HENS-02 Synthesis Schemas and Network Domain

## PRD Summary

Introduce OpenPinch-native HEN schema and domain objects before solver behavior
moves. This creates the public result model and validation shape that later
workflow and solver tasks must use.

## User Outcome

Users and maintainers get clear OpenPinch-native concepts for HEN design output:
`HeatExchanger`, `HeatExchangerNetwork`, synthesis tasks, task outcomes, and a
future `TargetOutput.design` payload. They do not need to understand OpenHENS
array positions or old OpenHENS public names.

## Scope

- New schema/domain files and fast validation tests.
- HEN configuration field definitions where needed for schema validation.
- No solver execution and no GEKKO/Pyomo imports.

## Plan Context

Read these sections before implementation:

- [Target Architecture](../../../OPENHENS_MIGRATION_PLAN.md#target-architecture)
- [Root Primitive Mandate and Parallel Workflow Purge](../../../OPENHENS_MIGRATION_PLAN.md#root-primitive-mandate-and-parallel-workflow-purge)
- [Heat Exchanger Network Domain Model](../../../OPENHENS_MIGRATION_PLAN.md#heat-exchanger-network-domain-model)
- [Labelled HEN Data Access](../../../OPENHENS_MIGRATION_PLAN.md#labelled-hen-data-access)
- [OpenHENS Source Disposition](../../../OPENHENS_MIGRATION_PLAN.md#openhens-source-disposition)
- [Phase 2: Synthesis Schemas and Network Domain in OpenPinch](../../../OPENHENS_MIGRATION_PLAN.md#phase-2-synthesis-schemas-and-network-domain-in-openpinch)

Settled decisions for this task:

- `HeatExchanger` and `HeatExchangerNetwork` are the public network model;
  raw solver arrays are private diagnostics only.
- Public names must be OpenPinch-native. Do not preserve `OpenHENS`,
  `CaseStudy`, `SynthesisStudy`, old field aliases, or old keyword shells.
- Do not expose option-owner classes such as
  `HeatExchangerNetworkDesignSpace`, `HeatExchangerNetworkMethodSequence`,
  `HeatExchangerNetworkSolveSetup`, or `HeatExchangerNetworkOutputs`. Those
  concepts are configuration fields or internal/serialization-only records.
- HEN schemas may describe tasks, outcomes, manifests, exports, and result
  payloads; they must not become alternate owners of streams, utilities,
  cases, variants, workspaces, or persistent configuration.
- Stable stream identity and labelled access are required before solver result
  extraction is migrated.
- The required workflow in `README.md` is mandatory; schema/domain objects must
  support that workflow and must not enable a parallel public workflow.

## Requirements Checklist

- [ ] Add OpenPinch-native synthesis schemas under
      `OpenPinch.lib.schemas.synthesis` or the agreed equivalent.
- [ ] Translate useful OpenHENS task/result concepts into OpenPinch-native
      names: synthesis task, task outcome, synthesis result, and optional
      manifest/export records.
- [ ] Translate useful OpenHENS design-space, method-sequence, solve-setup, and
      output-control concepts into `TargetInput.options` /
      `CONFIG_FIELD_SPECS`, not into public option-owner classes.
- [ ] Do not port `SynthesisStudy`, `CaseStudy`, `OpenHENS`, or old keyword
      option shells as public roots.
- [ ] Do not accept OpenHENS field aliases such as `min_dT_values`,
      `min_dqda_values`, `output_folder`, or `output_formats`.
- [ ] Do not add public option-owner classes for design space, method sequence,
      solve setup, or outputs.
- [ ] If an external JSON alias is proposed, require a named non-OpenHENS
      consumer, make it serialization-only, and update the plan/task docs in
      the same reviewed PR.
- [ ] Add HEN option keys to `CONFIG_FIELD_SPECS` when schemas need persistent
      controls, using OpenPinch-style names such as
      `HENS_APPROACH_TEMPERATURES`, `HENS_DERIVATIVE_THRESHOLDS`,
      `HENS_STAGE_SELECTION`, `HENS_METHOD_SEQUENCE`, `HENS_PDM_SOLVER`,
      `HENS_TDM_SOLVER`, `HENS_ESM_SOLVER`, `HENS_SOLVE_TOLERANCE`,
      `HENS_MAX_PARALLEL`, `HENS_LOG_LEVEL`, `HENS_OUTPUT_FOLDER`,
      `HENS_OUTPUT_FORMATS`, `HENS_RUN_ID`, and
      `HENS_BEST_SOLUTIONS_TO_SAVE`.
- [ ] Introduce `HeatExchanger` in `OpenPinch.classes` or the agreed domain
      module.
- [ ] Represent recovery exchangers as hot process stream -> cold process
      stream.
- [ ] Represent hot utility exchangers as hot utility stream -> cold process
      stream.
- [ ] Represent cold utility exchangers as hot process stream -> cold utility
      stream.
- [ ] Include stage, duty, area, active flag, approach temperatures,
      inlet/outlet temperatures, cost fields where available, and private
      solver/source metadata needed for migration traceability.
- [ ] Introduce `HeatExchangerNetwork` as an ordered collection of exchangers
      plus synthesis metadata and summary metrics.
- [ ] Add identity-based accessors for all exchangers involving one stream.
- [ ] Add identity-based accessors for the exchanger between a specific source
      stream, sink stream, and stage.
- [ ] Add totals by exchanger kind, stream, and stage.
- [ ] Add HEN labels following the `ProblemTableLabel` pattern, including
      recovery duty, hot utility duty, cold utility duty, recovery area, outlet
      temperatures, match active, and match allowed.
- [ ] Add type aliases for canonical label strings where useful.
- [ ] Keep stream axes labelled by stable OpenPinch stream identity.
- [ ] Store private axis-map metadata needed by future solver arrays without
      exposing array offsets as the public API.
- [ ] Define `HeatExchangerNetworkSynthesisResult` with a
      `HeatExchangerNetwork`, solver/task metadata, objective values, and
      optional diagnostic references.
- [ ] Enforce strict validation for positive grids, finite values, valid stage
      selections, valid output formats, positive tolerances, and valid run ids.
- [ ] Preserve deterministic task id generation.
- [ ] Add serialization round-trip tests for every new public schema/domain
      object.
- [ ] Add public export snapshot tests for intentionally exported names.
- [ ] Add negative API tests proving no `OpenHENS`, no `openhens` module path,
      no `OPENHENS_*` environment contract, no public study/case root, and no
      raw-input synthesis runner.
- [ ] Add negative API tests proving no public service/runner accepts
      `HeatExchangerNetworkDesignSpace`, `HeatExchangerNetworkSolveSetup`, or
      similar option-like records as configuration owners.
- [ ] Add row-reordering tests proving labelled exchanger links and duties are
      stable for the same OpenPinch stream identities.

## General Standards That Apply

- [ ] Enforce the required OpenPinch workflow from `README.md`; deviations are
      not permitted.
- [ ] New schema/domain objects must not own stream, utility, case, workspace,
      or economic state that belongs to `TargetInput`, `PinchProblem`, or
      `PinchWorkspace`.
- [ ] New objects must be constructible and serializable without optional solver
      dependencies.
- [ ] Public names must be OpenPinch-native.
- [ ] Raw solver arrays stay private migration details.

## Verification Checklist

- [ ] Fast schema validation tests pass.
- [ ] Domain serialization and labelled-access tests pass.
- [ ] Negative API tests for absent OpenHENS compatibility surfaces pass.
- [ ] Import smoke tests prove no synthesis solver dependency is imported.
- [ ] Existing OpenPinch API surface tests pass.

## Definition of Done

- [ ] `HeatExchanger` and `HeatExchangerNetwork` model HEN results by stable
      stream identity, not by raw array position.
- [ ] Synthesis schemas can express tasks, outcomes, result payloads, and export
      records with OpenPinch-native names.
- [ ] Validation rejects invalid grids, stage selections, tolerances, formats,
      run ids, and legacy public aliases.
- [ ] Tests prove no parallel problem/case/workspace root was introduced.
- [ ] The task can be completed with no GEKKO/Pyomo import or solver execution.

## Out of Scope

- Moving solver equations.
- Adding the `PinchProblem.design` execution path.
- Converting every OpenHENS fixture.
- Writing optional JSON/CSV export files.

## Implementation Notes

- 
