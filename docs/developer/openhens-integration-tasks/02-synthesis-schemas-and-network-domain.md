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

- [x] Add OpenPinch-native synthesis schemas under
      `OpenPinch.lib.schemas.synthesis` or the agreed equivalent.
- [x] Translate useful OpenHENS task/result concepts into OpenPinch-native
      names: synthesis task, task outcome, synthesis result, and optional
      manifest/export records.
- [x] Translate useful OpenHENS design-space, method-sequence, solve-setup, and
      output-control concepts into `TargetInput.options` /
      `CONFIG_FIELD_SPECS`, not into public option-owner classes.
- [x] Do not port `SynthesisStudy`, `CaseStudy`, `OpenHENS`, or old keyword
      option shells as public roots.
- [x] Do not accept OpenHENS field aliases such as `min_dT_values`,
      `min_dqda_values`, `output_folder`, or `output_formats`.
- [x] Do not add public option-owner classes for design space, method sequence,
      solve setup, or outputs.
- [x] If an external JSON alias is proposed, require a named non-OpenHENS
      consumer, make it serialization-only, and update the plan/task docs in
      the same reviewed PR.
- [x] Add HEN option keys to `CONFIG_FIELD_SPECS` when schemas need persistent
      controls, using OpenPinch-style names such as
      `HENS_APPROACH_TEMPERATURES`, `HENS_DERIVATIVE_THRESHOLDS`,
      `HENS_STAGE_SELECTION`, `HENS_METHOD_SEQUENCE`, `HENS_PDM_SOLVER`,
      `HENS_TDM_SOLVER`, `HENS_ESM_SOLVER`, `HENS_SOLVE_TOLERANCE`,
      `HENS_MAX_PARALLEL`, `HENS_LOG_LEVEL`, `HENS_OUTPUT_FOLDER`,
      `HENS_OUTPUT_FORMATS`, `HENS_RUN_ID`, and
      `HENS_BEST_SOLUTIONS_TO_SAVE`.
- [x] Introduce `HeatExchanger` in `OpenPinch.classes` or the agreed domain
      module.
- [x] Represent recovery exchangers as hot process stream -> cold process
      stream.
- [x] Represent hot utility exchangers as hot utility stream -> cold process
      stream.
- [x] Represent cold utility exchangers as hot process stream -> cold utility
      stream.
- [x] Include stage, duty, area, active flag, approach temperatures,
      inlet/outlet temperatures, cost fields where available, and private
      solver/source metadata needed for migration traceability.
- [x] Introduce `HeatExchangerNetwork` as an ordered collection of exchangers
      plus synthesis metadata and summary metrics.
- [x] Add identity-based accessors for all exchangers involving one stream.
- [x] Add identity-based accessors for the exchanger between a specific source
      stream, sink stream, and stage.
- [x] Add totals by exchanger kind, stream, and stage.
- [x] Add HEN labels following the `ProblemTableLabel` pattern, including
      recovery duty, hot utility duty, cold utility duty, recovery area, outlet
      temperatures, match active, and match allowed.
- [x] Add type aliases for canonical label strings where useful.
- [x] Keep stream axes labelled by stable OpenPinch stream identity.
- [x] Store private axis-map metadata needed by future solver arrays without
      exposing array offsets as the public API.
- [x] Define `HeatExchangerNetworkSynthesisResult` with a
      `HeatExchangerNetwork`, solver/task metadata, objective values, and
      optional diagnostic references.
- [x] Enforce strict validation for positive grids, finite values, valid stage
      selections, valid output formats, positive tolerances, and valid run ids.
- [x] Preserve deterministic task id generation.
- [x] Add serialization round-trip tests for every new public schema/domain
      object.
- [x] Add public export snapshot tests for intentionally exported names.
- [x] Add negative API tests proving no `OpenHENS`, no `openhens` module path,
      no `OPENHENS_*` environment contract, no public study/case root, and no
      raw-input synthesis runner.
- [x] Add negative API tests proving no public service/runner accepts
      `HeatExchangerNetworkDesignSpace`, `HeatExchangerNetworkSolveSetup`, or
      similar option-like records as configuration owners.
- [x] Add row-reordering tests proving labelled exchanger links and duties are
      stable for the same OpenPinch stream identities.

## General Standards That Apply

- [x] Enforce the required OpenPinch workflow from `README.md`; deviations are
      not permitted.
- [x] New schema/domain objects must not own stream, utility, case, workspace,
      or economic state that belongs to `TargetInput`, `PinchProblem`, or
      `PinchWorkspace`.
- [x] New objects must be constructible and serializable without optional solver
      dependencies.
- [x] Public names must be OpenPinch-native.
- [x] Raw solver arrays stay private migration details.

## Verification Checklist

- [x] Fast schema validation tests pass.
- [x] Domain serialization and labelled-access tests pass.
- [x] Negative API tests for absent OpenHENS compatibility surfaces pass.
- [x] Import smoke tests prove no synthesis solver dependency is imported.
- [x] Existing OpenPinch API surface tests pass.

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

- 2026-06-16: Added `OpenPinch.classes.HeatExchanger` and
  `OpenPinch.classes.HeatExchangerNetwork` as Pydantic domain models. Direction
  validation enforces recovery as process -> process, hot utility as utility ->
  process, and cold utility as process -> utility. Public serialization excludes
  private solver/source metadata.
- 2026-06-16: Added `HeatExchangerNetworkLabel` / `HEN` labels and
  `HeatExchangerNetworkLabelKey`, plus identity-based network accessors for
  stream involvement, source/sink/stage lookup, labelled values, and totals by
  kind, stream, and stage.
- 2026-06-16: Added `OpenPinch.lib.schemas.synthesis` with
  `HeatExchangerNetworkSynthesisTask`,
  `HeatExchangerNetworkSynthesisTaskOutcome`,
  `HeatExchangerNetworkSynthesisResult`,
  `HeatExchangerNetworkSynthesisManifest`, and
  `HeatExchangerNetworkSynthesisExportRecord`. `TargetOutput.design` now accepts
  the synthesis result payload.
- 2026-06-16: Added HEN controls to `CONFIG_FIELD_SPECS` under the `synthesis`
  group with OpenPinch names: `HENS_APPROACH_TEMPERATURES`,
  `HENS_DERIVATIVE_THRESHOLDS`, `HENS_STAGE_SELECTION`,
  `HENS_METHOD_SEQUENCE`, `HENS_PDM_SOLVER`, `HENS_TDM_SOLVER`,
  `HENS_ESM_SOLVER`, `HENS_SOLVE_TOLERANCE`, `HENS_MAX_PARALLEL`,
  `HENS_LOG_LEVEL`, `HENS_OUTPUT_FOLDER`, `HENS_OUTPUT_FORMATS`,
  `HENS_RUN_ID`, and `HENS_BEST_SOLUTIONS_TO_SAVE`.
- 2026-06-16: Added tests in
  `tests/test_classes/test_heat_exchanger_network.py` and
  `tests/test_lib/test_synthesis_schemas.py` covering serialization
  round-trips, deterministic task IDs, strict grids/stages/formats/tolerances/run
  IDs, public export snapshots, negative OpenHENS compatibility surfaces, absent
  option-owner config records, and row-reordering stability for labelled stream
  duties.
- 2026-06-16: Verification passed:
  `rtk uv run pytest tests/test_classes/test_heat_exchanger_network.py
  tests/test_lib/test_synthesis_schemas.py tests/test_package_api_surface.py
  tests/test_synthesis_dependency_boundaries.py tests/test_lib/test_config_enums.py
  tests/test_lib/test_io_schemas.py -q` -> `39 passed in 10.79s`.
- 2026-06-16: Verification passed:
  `rtk uv run ruff check OpenPinch/classes/__init__.py
  OpenPinch/classes/heat_exchanger.py
  OpenPinch/classes/heat_exchanger_network.py OpenPinch/lib/__init__.py
  OpenPinch/lib/config_metadata.py OpenPinch/lib/enums.py
  OpenPinch/lib/heat_exchanger_network_types.py
  OpenPinch/lib/schemas/__init__.py OpenPinch/lib/schemas/io.py
  OpenPinch/lib/schemas/synthesis.py
  tests/test_classes/test_heat_exchanger_network.py
  tests/test_lib/test_synthesis_schemas.py tests/test_package_api_surface.py` ->
  `All checks passed!`.
- 2026-06-16: Verification passed:
  `rtk git diff --check -- ':!.DS_Store'` -> no whitespace errors.
- 2026-06-16: No solver execution was added or run. Import smoke coverage came
  from `tests/test_synthesis_dependency_boundaries.py` in the passing pytest
  command, which asserts `import OpenPinch` does not import GEKKO, Pyomo, or
  other synthesis-only dependencies.
- 2026-06-16: Definition of Done remains unchecked pending adversarial review.
- 2026-06-16 re-review fix: Addressed the P1 review finding by adding HENS
  value validation to the canonical configuration path. `Configuration` now
  validates supported option values before assignment, and `TargetInput.options`
  validates the same option payloads during schema validation. The focused tests
  now reject invalid HENS approach-temperature grids, derivative-threshold grids,
  stage selections, output formats, solve tolerances, and run ids through both
  `Configuration(options=...)` and `TargetInput(..., options=...)`.
- 2026-06-16 re-review fix: Added valid canonical-option coverage proving HENS
  option values remain accepted and normalized on both `Configuration` and
  `TargetInput.options`.
- 2026-06-16 re-review fix verification passed:
  `rtk uv run pytest tests/test_lib/test_synthesis_schemas.py
  tests/test_lib/test_config_enums.py tests/test_lib/test_io_schemas.py -q` ->
  `37 passed in 1.84s`.
- 2026-06-16 re-review fix verification passed:
  `rtk uv run pytest tests/test_classes/test_heat_exchanger_network.py
  tests/test_lib/test_synthesis_schemas.py tests/test_package_api_surface.py
  tests/test_synthesis_dependency_boundaries.py tests/test_lib/test_config_enums.py
  tests/test_lib/test_io_schemas.py -q` -> `47 passed in 3.50s`.
- 2026-06-16 re-review fix verification passed:
  `rtk uv run ruff check OpenPinch/classes/__init__.py
  OpenPinch/classes/heat_exchanger.py
  OpenPinch/classes/heat_exchanger_network.py OpenPinch/lib/__init__.py
  OpenPinch/lib/config.py OpenPinch/lib/config_metadata.py
  OpenPinch/lib/enums.py OpenPinch/lib/heat_exchanger_network_types.py
  OpenPinch/lib/schemas/__init__.py OpenPinch/lib/schemas/io.py
  OpenPinch/lib/schemas/synthesis.py
  tests/test_classes/test_heat_exchanger_network.py
  tests/test_lib/test_synthesis_schemas.py tests/test_package_api_surface.py` ->
  `All checks passed!`.
- 2026-06-16 re-review fix verification passed:
  `rtk git diff --check -- ':!.DS_Store'` -> no whitespace errors.
- 2026-06-16 re-review fix for P2: root `.DS_Store` remains modified in the
  working tree but is pre-existing/user-owned and was not touched, staged, or
  included in the HENS-02 task slice. Evidence: `rtk git status --short` still
  lists `M .DS_Store`; `rtk git diff --name-only -- .DS_Store` lists only
  `.DS_Store`; `rtk git diff --cached --name-only` returns no staged files; and
  `rtk git diff --name-only -- ':!.DS_Store'` lists only HENS-02 tracked files.
- 2026-06-16: Definition of Done remains unchecked pending re-review.
