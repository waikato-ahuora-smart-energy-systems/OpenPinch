# OpenHENS to OpenPinch Migration Plan

## Purpose

OpenPinch should become the central library for heat utility, pinch, and heat
exchanger network algorithms. OpenHENS should be migrated into OpenPinch in a
way that keeps the heat exchanger network synthesis algorithms intact, reuses
OpenPinch classes and services where they are already stronger, and adds
regression coverage before any scientific behavior is changed.

This plan is intentionally staged. The early phases establish baselines,
package boundaries, internal adapters, and tests. The solver model move happens
only after there is enough validation to distinguish intentional structural
changes from numerical regressions.

## Current Codebase Inventory

### OpenPinch

OpenPinch is already a packaged library with a stable public surface:

- Root API: `OpenPinch.__init__`, `pinch_analysis_service`, `PinchProblem`,
  `PinchWorkspace`.
- Typed request/response models:
  `OpenPinch.lib.schemas.io.StreamSchema`, `UtilitySchema`, `TargetInput`,
  `TargetOutput`, and zone-tree schemas.
- Unit-aware runtime objects: `Value`, `Stream`, `StreamCollection`, `Zone`,
  and `ProblemTable`.
- Service boundaries:
  `OpenPinch.services.data_preprocessing_service`,
  `direct_heat_integration_service`, `indirect_heat_integration_service`,
  heat pump/refrigeration services, cogeneration, exergy, and area/cost
  targeting.
- Reusable low-level algorithms:
  `problem_table_analysis`, `utility_targeting`, `gcc_manipulation`,
  `capital_cost_and_area_targeting`, `utils.heat_exchanger`, and
  `utils.costing`.
- Existing regression coverage for problem tables, utility targeting, LMTD and
  exchanger helpers, costing helpers, schemas, API exports, docs, packaging,
  and sample workflows.
- Packaging currently targets Python `>=3.14` with a relatively small core
  dependency set and optional extras for dashboard, notebooks, and TESPy-backed
  Brayton-cycle tooling.

### OpenHENS

OpenHENS is a smaller synthesis package built around a PDM -> TDM -> ESM search
workflow:

- Public API and typed refactor surface:
  `OpenHENS`, `SynthesisStudy`, `CaseStudy`, `DesignSpace`, `MethodSequence`,
  `SolveSetup`, `StudyOutputs`, `SynthesisTask`, `TaskOutcome`,
  `NetworkSolution`, `StudyManifest`, and `StudyOutcome`.
- Input parsing:
  `CaseStudy.from_csv()` parses workbook-export CSV files with process streams,
  utilities, and exchanger economics rows. OpenHENS case temperatures are
  declared in K.
- Orchestration:
  `run_synthesis_workflow()` builds PDM tasks over `dTmin`, fans successful PDM
  topologies into TDM tasks over `min_dqda`, and fans successful TDM topologies
  into ESM refinement tasks.
- Solver boundary:
  `LocalSynthesisExecutor` translates task records into
  `HeatExchangerNetworkProblem` instances and runs local multiprocessing.
- Solver/model layer:
  `GenericHENModel`, `PinchDecompModel`, and `StageWiseModel` hold the current
  GEKKO/Pyomo equations, post-processing, stage reduction, and topology
  evolution.
- Verification and artifacts:
  `solution_verification.py` checks temperatures, utility costs, and area
  costs. `artifacts.py` writes manifest JSON, per-task JSON results, and CSV
  metrics, with optional Excel and plot outputs.
- Regression tests:
  fast tests cover public models, parsing, workflow, solver wrappers, and
  artifacts. Marked solver tests compare Four-stream and Nine-stream benchmark
  runs against saved workbook baselines.
- Packaging currently targets Python `>=3.12` and depends on heavier synthesis
  tools such as `pyomo`, `gekko`, `matplotlib`, `plotly`, `kaleido`,
  `openpyxl`, and `wakepy`.

## Design Principles

1. Baselines first, movement second.
   Before moving model code, capture the current OpenHENS behavior as executable
   tests and stable artifacts.

2. Reuse OpenPinch where it is the better abstraction.
   Use OpenPinch schemas, unit handling, stream collections, problem tables,
   LMTD/costing helpers, and service/export conventions where they match the
   synthesis need.

3. Do not force-fit equation models into targeting services.
   PDM/TDM/ESM synthesis is a new advanced service family. It should sit beside
   direct/indirect/HPR/cogeneration targeting, not inside those existing
   algorithms.

4. Put HEN operations under a design accessor.
   `PinchProblem` already exposes fundamental `target` and `plot` handles.
   HEN synthesis is a design workflow, not a target calculation, so public HEN
   operations should start from a new `PinchProblem.design` accessor.

5. Keep solver dependencies optional and lazy.
   OpenPinch core installs should not require Pyomo, GEKKO, solver binaries, or
   plotting stacks unless the user installs a synthesis extra.

6. Preserve behavior at each migration phase.
   Equation refactors, multi-utility synthesis, new solvers, distributed
   execution, or performance improvements should be separate follow-up work.

7. Prefer OpenPinch result contracts over live solver objects.
   OpenHENS' recent `SynthesisTask`, `TaskOutcome`, `NetworkSolution`, and
   artifact work should be carried into OpenPinch as typed result payloads
   attached to `PinchProblem._results`, rather than reverting to pickle,
   GEKKO-object persistence, or end-of-workflow disk writes.

8. Do not support OpenHENS runtime APIs.
   The migrated implementation should expose OpenPinch-native names only.
   OpenHENS import paths, `OpenHENS`-named facade classes, old keyword option
   shells, and OpenHENS-style public entry points should not be preserved inside
   OpenPinch.

9. Treat OpenHENS as source material, not as a contract.
   OpenHENS code, artifacts, and tests are migration inputs used to preserve the
   scientific algorithm. They are not a runtime API, object model, or naming
   scheme that OpenPinch must continue to support.

10. Start from OpenPinch problem primitives.
   `PinchProblem` and `PinchWorkspace` are the root primitives, not optional
   integration conveniences. HEN synthesis should begin from `PinchProblem` and
   be executable through `PinchWorkspace`. Do not introduce independent pinch
   model management, variant management, stream ownership, or targeting
   workflows for anything outside the private HEN solver internals.

11. Results first, exports second.
   The primary result of a HEN design run is an updated
   `PinchProblem._results` object. Disk artifacts are optional exports from
   `problem.results`, not the terminal step of the core workflow.

## Target Architecture

Add a synthesis package under OpenPinch without expanding the core import cost:

```text
OpenPinch/
  classes/
    heat_exchanger.py
    heat_exchanger_network.py
  lib/
    enums.py              # add HEN field/link labels beside existing labels
    heat_exchanger_network_types.py
    schemas/
      io.py               # extend TargetOutput with optional design field
      reporting.py        # add any typed HEN design payloads used by design
      synthesis.py
  services/
    heat_exchanger_network_synthesis/
      __init__.py
      exports.py          # optional JSON/CSV export from TargetOutput results
      array_adapter.py
      domain.py
      problem_adapter.py
      solvers.py
      workflow.py
      verification.py
      visualization.py        # optional/deferred; not part of the model core
      models/
        __init__.py
        base.py
        pinch_decomposition.py
        stagewise.py
```

Public API additions should be conservative:

- `OpenPinch.classes.HeatExchanger`
- `OpenPinch.classes.HeatExchangerNetwork`
- `OpenPinch.services.heat_exchanger_network_synthesis_service(...)`
- `OpenPinch.services.heat_exchanger_network_synthesis` as the advanced module
  namespace.
- A `PinchProblem.design.heat_exchanger_network_synthesis(...)` workflow method,
  backed by the service entry point.
- A `PinchWorkspace.solve_variant(..., workflow="heat_exchanger_network_synthesis")`
  path using the existing workspace workflow dispatcher, routed to the design
  accessor.
- An optional `design` field on `TargetOutput` beside `targets`. The preferred
  Python shape should follow OpenPinch's existing convention, for example
  `TargetOutput.design: HeatExchangerNetworkSynthesisResult | None = None`, with
  aliases only if an external JSON name is required.
- Schema exports from `OpenPinch.lib.schemas.synthesis` for synthesis tasks,
  outcomes, optional export contracts, and HEN result contracts. Persistent HEN
  controls stay in `TargetInput.options` / `Configuration`; economic inputs stay
  in `UtilitySchema` and OpenPinch costing configuration.
- Optional root exports only after the API shape is stable and tested.

The service entry point should accept a live `PinchProblem` and read persistent
HEN controls from that problem's `TargetInput.options` / prepared
`Configuration`. It should not accept raw CSV rows, raw stream lists,
`TargetInput`, a separate design-options object, or a new public study/case
object. Those inputs must first be loaded into `PinchProblem` or
`PinchWorkspace`.

The service must populate `PinchProblem._results` with a `TargetOutput` that
contains existing target results plus the HEN `design` payload. It should not make
filesystem artifact generation the required final output of synthesis.

Suggested public names:

- `HeatExchanger`
- `HeatExchangerNetwork`
- `HeatExchangerNetworkSynthesis`
- `HeatExchangerNetworkDesignSpace`
- `HeatExchangerNetworkMethodSequence`
- `HeatExchangerNetworkSolveSetup`
- `HeatExchangerNetworkOutputs`
- `HeatExchangerNetworkSynthesisTask`
- `HeatExchangerNetworkSynthesisTaskOutcome`
- `HeatExchangerNetworkSynthesisResult`
- `HeatExchangerNetworkSynthesisManifest`

Do not add an `OpenHENS` facade, import-path shims, or old keyword-option
entry points. The user-visible migration should be explicit: new code imports
the OpenPinch synthesis service and OpenPinch-native schema names.

## Reuse Map

| OpenHENS area | OpenPinch target | Migration direction |
| --- | --- | --- |
| OpenHENS example case data | OpenPinch-compatible JSON using `TargetInput`, `StreamSchema`, `UtilitySchema`, OpenPinch costing configuration, HEN options in `TargetInput.options`, then `PinchProblem` | Convert source CSV examples into standard OpenPinch JSON fixtures once. The primary workflow should load JSON/`TargetInput` through `PinchProblem`, not CSV. |
| `HotStream`, `ColdStream`, `HotUtility`, `ColdUtility` | OpenPinch `StreamSchema`/`UtilitySchema`; extend those schemas if generally useful stream/utility fields are missing | Avoid duplicate stream models. Do not put stream/utility identity, temperatures, HTC, heat loads, or utility prices in HEN side objects. |
| Source OpenHENS case array export | `array_adapter.py` fed by a prepared `PinchProblem` | Keep an internal solver-array adapter only while moved equation models still expect arrays. Do not expose it as public API. |
| OpenHENS positional network arrays | `HeatExchangerNetwork`, `HeatExchanger`, and enum-labelled HEN table/tensor accessors | Convert solver arrays into labelled stream-link records immediately after solution extraction. Preserve positional arrays only behind private solver boundaries. |
| `PinchDecompModel.calculate_pinch()` and `pinch_classes` | `ProblemTable`, direct integration/utility targeting helpers | Replace only after parity tests pass for every OpenHENS example and `dTmin` grid. |
| LMTD calculations | `OpenPinch.utils.heat_exchanger.compute_LMTD_from_dts` | Replace hand-coded LMTD post-processing in controlled slices. |
| Capital cost formulas | `OpenPinch.utils.costing` where formulas match | Reuse common helpers, but keep OpenHENS annual cost conventions explicit. |
| `DesignSpace`, `MethodSequence`, `SolveSetup`, `StudyOutputs` | New `CONFIG_FIELD_SPECS` entries consumed through `TargetInput.options` / `Configuration` | Port and rename deliberately as configuration options only; do not make them case/study/workflow roots. |
| `workflow.py` | New synthesis workflow module invoked from `PinchProblem.design` and `PinchWorkspace.solve_variant` | Port mostly as-is first, but use existing OpenPinch workflow entry points rather than a separate case/study manager. |
| `solvers.py` | New synthesis solver module | Port with lazy optional imports and clearer missing-solver errors. |
| `artifacts.py` | Optional synthesis export module | Do not make artifact writing the terminal workflow step. Port only the useful JSON/CSV export contracts as exports from `TargetOutput.design`, keyed by OpenPinch problem or workspace variant identity. |
| `GenericHENModel`, `PinchDecompModel`, `StageWiseModel` | `models/` under synthesis service | Move after adapter and regression gates exist. |
| `solution_verification.py` | `verification.py` | Port early, then add stronger invariants. |

## OpenPinch Reuse Commitments

The migration should actively use the existing OpenPinch model stack instead of
creating parallel equivalents.

| OpenPinch component | Required synthesis use |
| --- | --- |
| `Value` and `ValueWithUnit` | All user-facing synthesis inputs with units should enter through explicit value/unit payloads. Kelvin case data must be explicit rather than relying on OpenPinch default temperature units. |
| `StreamSchema` and `UtilitySchema` | Process and utility data from converted OpenHENS examples should be represented in these schemas before any solver-array export happens. |
| `Stream` | Prepared synthesis stream data should use OpenPinch's derived `CP`, shifted temperatures, `dt_cont_act`, HTC, and heat-flow behavior for validation and pinch-target parity checks. |
| `StreamCollection` | Hot/cold process and utility groupings should use OpenPinch collection helpers rather than list-of-array handling outside the internal solver adapter. |
| `Zone` | Synthesis cases should prepare a single process zone first. Multi-zone support should be deferred until the single-zone OpenHENS behavior is reproduced. |
| `PinchProblem` | All public HEN synthesis solves should start from a validated `PinchProblem`, regardless of whether the original input was OpenPinch-native JSON, a `TargetInput`, or another existing `PinchProblem` source. CSV is not part of the primary HEN workflow. |
| `PinchWorkspace` | Multi-case and variant execution should use existing workspace payload storage and `solve_variant(..., workflow="heat_exchanger_network_synthesis")`, not a synthesis-specific case manager for non-HENS concerns. |
| `PinchProblem.design` | HEN synthesis should be exposed through a new design accessor parallel to the existing `target` and `plot` handles. `target` remains for heat/utility targeting; `design` owns network synthesis/design workflows. |
| `prepare_problem(...)` | Synthesis must build from OpenPinch's preparation service for validation, stream creation, and shared workflows before any solver-array export happens. |
| `ProblemTable` | Pinch temperature, hot/cold utility target, heat recovery target, and above/below-pinch split checks should be based on `ProblemTable` parity before replacing OpenHENS' adapted pinch code. |
| `ProblemTableLabel` pattern | HEN solution data should get enum-labelled accessors so heat duties, areas, temperatures, and match binaries can be requested by semantic label rather than by raw array column/axis positions. |
| `problem_table_analysis` and `direct_heat_integration_entry` | These should become the authoritative targeting implementation for PDM pinch decomposition once parity is proven. |
| `utils.heat_exchanger` | LMTD helper replacement should use existing OpenPinch formulas only after endpoint/tolerance parity tests pass. |
| `utils.costing` | Capital-cost helper replacement should use OpenPinch utilities where formulas match; HEN synthesis TAC conventions stay explicit in synthesis code. |
| `TargetOutput` / `TargetResults` | HEN synthesis should populate `PinchProblem._results` by extending `TargetOutput` with an optional `design` field beside `targets`. Targeting results and design outputs then share the same problem-owned result cache. |
| OpenPinch service/export/docs patterns | Synthesis should follow lazy service imports, optional extras, documented API exports, optional JSON/CSV export helpers, docs coverage, and public API snapshot tests. |

The only permitted array-style data boundary is the internal adapter that feeds
the moved equation models while they still require NumPy arrays. That adapter
must stay private and should shrink as model code is rewritten around OpenPinch
objects.

## Root Primitive Mandate and Parallel Workflow Purge

The migration should explicitly purge OpenHENS-style parallel workflow
ownership. The final OpenPinch shape should have one public owner for loaded
thermal problem state and one public owner for multi-case orchestration:

- `PinchProblem` is the only public owner of prepared stream, utility, zone,
  targeting, and HEN design state for one problem.
- `PinchWorkspace` is the only public owner of multi-case, variant, comparison,
  and workflow execution state.
- `PinchProblem.design` is the public HEN design entry point.
- `PinchWorkspace.solve_variant(..., workflow="heat_exchanger_network_synthesis")`
  is the multi-case entry point and must dispatch to the active variant's
  `PinchProblem.design` path.
- `HeatExchangerNetworkSynthesis` may exist only as a thin design runner bound
  to a `PinchProblem`; it must not own or reload streams, utilities, variants,
  cases, or workspace state.

Explicitly purge these parallel surfaces:

- no public `CaseStudy`, `SynthesisStudy`, or replacement object that owns a
  problem payload.
- no public `run_synthesis_workflow(case_or_study, ...)` entry point.
- no service that accepts raw stream lists, raw utility lists, CSV rows, or
  `TargetInput` directly for synthesis.
- no HEN-specific workspace, project, scenario, variant, or case registry.
- no duplicated validation path for stream/utility semantics outside
  `TargetInput`, `PinchProblem.load(...)`, and `prepare_problem(...)`.
- no public solver-array model management; positional arrays are private
  implementation details.
- no artifact manifest that identifies a result by an OpenHENS-style case or
  study id instead of OpenPinch problem/workspace variant identity.
- no workflow whose canonical output is a directory of artifacts rather than
  `PinchProblem._results`.

Allowed HEN-specific objects are task records, solver records, optional export
manifests, and typed payloads stored through `TargetOutput.design`. They may
reference a `PinchProblem`, workspace variant name, stream identities, and
exchanger-network identities; they must not become alternate owners of
process/utility model state, duplicate OpenPinch costing inputs, or carry
persistent HEN configuration outside `TargetInput.options`.

## TargetInput Boundary for HEN

`TargetInput` represents the OpenPinch problem payload:

- `streams`: process stream zone, name, supply/target temperatures, heat flow,
  `dt_cont`, HTC, and active flag.
- `utilities`: utility name, hot/cold type, temperatures, optional heat flow,
  `dt_cont`, HTC, utility price, and active flag.
- `options` and `zone_tree`: runtime options and zone hierarchy used by
  OpenPinch preparation, targeting, design, and workspace workflows.

These fields should not be duplicated in HEN-specific objects. Existing
OpenHENS CSV examples should be converted into standard OpenPinch JSON files
that populate `TargetInput`; the primary synthesis workflow should not read
CSVs.

HEN design/search controls should be added to `TargetInput.options`, not passed
as separate design objects. The `options` field is the payload representation
of `OpenPinch.lib.config.Configuration`; every new HEN option should be defined
in `CONFIG_FIELD_SPECS` so it is validated, documented, grouped for workspace
metadata, and available on the prepared `Zone.config`.

Add HEN configuration fields for:

- approach-temperature grid (`dTmin` values).
- derivative/minimum-dQdA thresholds.
- stage selection.
- PDM/TDM/ESM method order.
- solver choices.
- tolerance, local parallelism, and log level.
- artifact folder/formats, run id, and number of best solutions to persist.

Use explicit, OpenPinch-style option keys, for example
`HENS_APPROACH_TEMPERATURES`, `HENS_DERIVATIVE_THRESHOLDS`,
`HENS_STAGE_SELECTION`, `HENS_METHOD_SEQUENCE`, `HENS_PDM_SOLVER`,
`HENS_TDM_SOLVER`, `HENS_ESM_SOLVER`, `HENS_SOLVE_TOLERANCE`,
`HENS_MAX_PARALLEL`, `HENS_LOG_LEVEL`, `HENS_OUTPUT_FOLDER`,
`HENS_OUTPUT_FORMATS`, `HENS_RUN_ID`, and `HENS_BEST_SOLUTIONS_TO_SAVE`.

Economic inputs should reuse existing OpenPinch structures:

- OpenHENS hot-utility and cold-utility operating costs (`hu_cost`, `cu_cost`)
  map to `UtilitySchema.price` and the existing prepared `Stream.utility_cost`
  path.
- Heat-exchanger capital-cost coefficients should first map to existing
  OpenPinch costing configuration and helpers:
  `Configuration.FIXED_COST`, `Configuration.VARIABLE_COST`,
  `Configuration.COST_EXP`, `Configuration.DISCOUNT_RATE`,
  `Configuration.SERV_LIFE`, and `OpenPinch.utils.costing`.
- If HEN synthesis needs per-exchanger-kind capital-cost coefficients that the
  existing OpenPinch costing configuration cannot express, extend OpenPinch's
  general costing configuration/schema. Do not create a HEN-only economics
  object for values already covered by `UtilitySchema` or OpenPinch costing.

There is one migration audit item before changing schemas: OpenHENS also has
process stream cost fields (`h_cost`, `c_cost`) in its solver arrays, while
OpenPinch `StreamSchema` does not currently expose process stream price/cost
even though `Stream` has a `price` attribute. The migration should determine
whether those fields are used by the algorithm. If they are needed, prefer a
general `StreamSchema.price` extension or another existing OpenPinch costing
path, not a parallel HEN case object.

Use precise names in the migrated API and configuration:

- Prefer `TargetInput.options` / `Configuration` keys for HEN search, method,
  solver, and output controls.
- Prefer existing `UtilitySchema.price`, OpenPinch costing configuration, and
  OpenPinch costing helpers for economic inputs.
- Avoid separate `HeatExchangerNetworkDesignOptions` or HEN economics objects
  for data that belongs in `TargetInput.options`, `UtilitySchema`, or OpenPinch
  costing configuration.

## End-to-End Flow Comparison

This section defines the intended full-flow replacement. The migration is not
just a package move; it changes ownership so OpenPinch primitives become the
root of the workflow.

### Before Migration: OpenHENS-Owned Flow

```text
Workbook-export CSV or hand-built OpenHENS domain objects
  -> CaseStudy.from_csv(...) or CaseStudy(...)
  -> SynthesisStudy(case=CaseStudy, design_space, methods, solving, outputs)
  -> run_synthesis_workflow(study)
  -> generate PDM tasks over dTmin
  -> LocalSynthesisExecutor builds HeatExchangerNetworkProblem from CaseStudy
  -> OpenHENS PinchDecompModel calculates pinch with OpenHENS pinch_classes
  -> solve PDM and extract topology from positional Q_r arrays
  -> generate TDM tasks by fanning successful PDM topologies over min_dqda
  -> solve TDM with StageWiseModel and positional restrictions
  -> generate ESM refinement tasks from successful TDM topologies
  -> solve ESM/evolution path
  -> extract NetworkSolution with Q_r, Q_h, Q_c, temperature, area arrays
  -> verify solution using OpenHENS case/solver arrays
  -> write StudyManifest, per-task JSON, CSV metrics, optional Excel/plots
```

Ownership in the current OpenHENS flow:

- `CaseStudy` owns process streams, utilities, exchanger economics, and case
  identity.
- `SynthesisStudy` owns the case plus design-space, method, solve, and output
  configuration.
- `run_synthesis_workflow(...)` is the public orchestration root.
- `HeatExchangerNetworkProblem` and model classes rebuild solver-ready state
  from OpenHENS-owned stream/case data.
- Network structure is primarily represented by positional arrays:
  `Q_r[i][j][k]`, `Q_h[j]`, `Q_c[i]`, `z`, `z_allowed`, temperatures, and
  areas.
- Artifacts identify the run by OpenHENS study/case/task records.

This is the flow to preserve scientifically but not architecturally.

### After Migration: OpenPinch-Owned Flow

```text
OpenPinch-compatible JSON, TargetInput, or existing PinchProblem source
  -> PinchProblem(source=JSON path, TargetInput, or OpenPinch-native payload)
  -> PinchProblem.load(...) validation and canonicalization
  -> prepare_problem(...) builds Zone, StreamCollection, Stream, ProblemTable
  -> problem.design.heat_exchanger_network_synthesis()
  -> heat_exchanger_network_synthesis_service(problem)
  -> ProblemTable/OpenPinch targeting parity path supplies pinch decomposition data
  -> private problem_to_solver_arrays(...) only while moved equations need arrays
  -> generate PDM tasks over dTmin from problem-rooted settings
  -> solve PDM and extract topology by mapping private arrays back to stream identity
  -> generate TDM tasks from successful PDM HeatExchangerNetwork topology
  -> solve TDM with private restrictions derived from HeatExchangerNetwork
  -> generate ESM refinement tasks from successful TDM HeatExchangerNetwork topology
  -> solve ESM/evolution path
  -> extract HeatExchangerNetworkSynthesisResult
  -> expose HeatExchangerNetwork of HeatExchanger source/sink links
  -> verify solution using OpenPinch streams, labelled network data, and private arrays only where still necessary
  -> convert the accepted network outcome into the TargetOutput.design payload
  -> update PinchProblem._results as TargetOutput(..., targets=[...], design=...)
  -> optional export writes JSON/CSV artifacts from problem.results when requested
```

Ownership in the target OpenPinch flow:

- `PinchProblem` owns one loaded problem's streams, utilities, zones,
  preparation, targeting state, and design state.
- `PinchWorkspace` owns multi-case variants and dispatches design workflows to
  the active variant's `PinchProblem`.
- HEN configuration in `TargetInput.options` / `Configuration` configures
  synthesis but does not own stream, utility, zone, case, variant, or economic
  state.
- `HeatExchangerNetworkSynthesis` may coordinate one design run only while
  bound to a live `PinchProblem`.
- Private solver arrays may exist only behind `problem_to_solver_arrays(...)`
  and solution extraction while equation models are being migrated.
- Public network results are `HeatExchangerNetwork` objects containing
  `HeatExchanger` links with stable OpenPinch stream identities.
- `PinchProblem._results` is the canonical result store. Its `TargetOutput`
  payload carries normal `TargetResults` plus optional `design` data for HEN
  outcomes.
- Optional exports identify the run by OpenPinch problem/workspace variant
  identity, not by an OpenHENS-style study or case id.

### Flow Delta

| Flow concern | Before migration | After migration |
| --- | --- | --- |
| Root single-case object | `CaseStudy` / `SynthesisStudy` | `PinchProblem` |
| Root multi-case object | OpenHENS study/output folders | `PinchWorkspace` |
| Public execution call | `run_synthesis_workflow(study)` | `problem.design.heat_exchanger_network_synthesis(...)` or `workspace.solve_variant(..., workflow="heat_exchanger_network_synthesis")` |
| Stream/utility validation | OpenHENS row models and case validators | `TargetInput`, `PinchProblem.load(...)`, `prepare_problem(...)`, `Stream`, `StreamCollection` |
| Pinch decomposition source | OpenHENS `pinch_classes` and model-local arrays | OpenPinch `ProblemTable` and targeting services after parity gates |
| Solver input shape | OpenHENS case arrays | Private arrays derived from prepared `PinchProblem` |
| Topology representation | Positional `Q_r`, `Q_h`, `Q_c` restrictions | `HeatExchangerNetwork` topology with private array conversion only at solver boundaries |
| Result representation | `NetworkSolution` carrying positional arrays, plus artifact files | `PinchProblem._results` as `TargetOutput` containing `TargetResults` and optional `design` data that carries `HeatExchangerNetwork` / `HeatExchanger` links |
| Stream identity stability | Depends on array axis order | Stable OpenPinch stream identities and enum-labelled accessors |
| Optional export identity | OpenHENS study/case/task metadata | OpenPinch problem/workspace variant identity plus task metadata, exported from `problem.results` only when requested |
| Allowed bypasses | Direct CaseStudy/SynthesisStudy workflow | None; all synthesis starts from `PinchProblem` or `PinchWorkspace` |

The acceptance criterion for this section is strict: after migration, the old
left-hand flow should no longer be expressible as public OpenPinch API. It may
exist only as frozen reference behavior in migration tests or as private,
temporary solver internals during the equation-model move.

## Canonical Synthesis Problem Contract

The synthesis path must make OpenPinch ownership enforceable:

```text
OpenPinch-compatible JSON, TargetInput, or OpenPinch-native payload
  -> PinchProblem(source=JSON path, TargetInput, or OpenPinch-native payload)
  -> PinchProblem.load(...) validation and canonicalization
  -> prepare_problem(...) through PinchProblem's execution path
  -> single prepared Zone with StreamCollection-backed streams/utilities
  -> PinchProblem.design.heat_exchanger_network_synthesis(...)
  -> HeatExchangerNetwork configuration read from TargetInput.options / Configuration
  -> private problem_to_solver_arrays(...) only while moved equations require arrays
  -> synthesis workflow tasks
  -> HeatExchangerNetwork result with HeatExchanger stream-link records
```

Rules:

- Do not add any public synthesis case object or equivalent independent case
  owner.
  HEN-specific schemas may carry task metadata, solver outcomes, optional
  export manifests, the `TargetOutput.design` payload, and read-only snapshots
  of resolved execution configuration. Persistent HEN controls remain in
  `TargetInput.options` / `Configuration`; utility economics remain in
  `UtilitySchema`; process and utility stream ownership remains with
  `PinchProblem`.
- Existing OpenHENS CSV examples should be converted into OpenPinch-compatible
  JSON fixtures before migration. Runtime synthesis should not read CSV inputs.
  Utility economics must be represented in `UtilitySchema`; other capital-cost
  data must use existing or generalized OpenPinch costing configuration.
- The internal solver-array adapter must derive arrays from the prepared
  `PinchProblem` execution `Zone` or from data proven equivalent to that
  prepared `Zone`.
- Tests must fail if a new code path builds solver arrays directly from
  converted fixture rows, raw payloads, or a synthesis schema while bypassing
  `PinchProblem`, `TargetInput`,
  `prepare_problem(...)`, `Stream`, or `StreamCollection`.
- `PinchWorkspace` integration must use the existing `run_problem_workflow`
  dispatcher, extended so design workflows dispatch to `problem.design`.
  HEN synthesis can add a workflow name and design accessor method; it should
  not create separate workspace, variant, or project management.
- Multi-zone synthesis is out of scope until the single-zone OpenHENS behavior
  is reproduced through this contract.
- The adapter-removal target is explicit: once equation models accept
  OpenPinch-native stream, utility, and costing objects,
  `problem_to_solver_arrays(...)` should be deleted.

## Heat Exchanger Network Domain Model

Introduce `HeatExchanger` and `HeatExchangerNetwork` before moving the solver
models so the migrated workflow has an OpenPinch-native result model from the
start.

`HeatExchanger` should represent one heat-transfer link:

- `kind`: recovery, hot utility, or cold utility.
- `source_stream`: the stream providing heat.
- `sink_stream`: the stream receiving heat.
- `stage`: synthesis stage for recovery exchangers, optional for utility
  exchangers where the current OpenHENS model has no stage axis.
- `duty`, `area`, `active`, `approach_temperatures`, inlet/outlet temperatures,
  and cost fields where available.
- solver/source metadata sufficient to trace back to `Q_r[i][j][k]`, `Q_h[j]`,
  or `Q_c[i]` during migration, without exposing those positional arrays as the
  public API.

The stream-link direction is explicit:

- recovery exchanger: hot process stream -> cold process stream.
- hot utility exchanger: hot utility stream -> cold process stream.
- cold utility exchanger: hot process stream -> cold utility stream.

`HeatExchangerNetwork` should own an ordered collection of `HeatExchanger`
records plus synthesis metadata and summary metrics. It should provide
identity-based accessors such as:

- all exchangers involving one stream, regardless of current stream ordering.
- the exchanger between a specific hot/source stream and cold/sink stream at a
  stage.
- totals by exchanger kind, stream, and stage.
- conversion to private solver arrays only inside the synthesis service.

The model should be usable by verification, `TargetOutput.design`, optional exports,
and future visualization. Solver arrays should be treated as an implementation
detail feeding into and out of this OpenPinch-native network object.

## Result Envelope Model

HEN synthesis should use the same problem-owned result cache as targeting:
`PinchProblem._results`.

Current OpenPinch targeting writes `TargetOutput` into `PinchProblem._results`;
`TargetOutput` currently contains `targets: list[TargetResults]`. The migration
should extend that envelope rather than creating an artifact-only or
workflow-local result store:

- Add an optional `design` field beside `targets` in `TargetOutput`.
- The `design` field should represent the HEN design outcome for a
  problem/state/workflow run. It should carry the accepted
  `HeatExchangerNetwork`, objective values such as TAC and utility/capital
  costs, solver status, method/stage metadata, state id, and references to
  optional task/outcome records where needed.
- `problem.design.heat_exchanger_network_synthesis(...)` should update
  `problem._results` and return either the updated `TargetOutput` or the
  `TargetOutput.design` convenience payload, matching the style chosen for
  existing `PinchProblem.target` methods. The cache is the canonical state in
  either case.
- If targeting has not yet been run, the design workflow may compute the target
  data needed for HEN synthesis and populate both `targets` and `design` in the
  same `TargetOutput`. If targeting has already been run, it should preserve
  existing `TargetResults` and replace or refresh only `design` according to
  documented workflow semantics.
- JSON/CSV files are export views generated from `problem.results`; they are
  not the primary persistence model for an in-process HEN solve.

## Labelled HEN Data Access

OpenHENS fundamentally stores solved HEN data in three duty arrays:

- `Q_r[i][j][k]`: recovery duty from hot process stream `i` to cold process
  stream `j` at stage `k`.
- `Q_h[j]`: hot utility duty into cold process stream `j`.
- `Q_c[i]`: cold utility duty removing heat from hot process stream `i`.

The moved code also carries parallel arrays such as `z_allowed`, `z`, `z_hu`,
`z_cu`, `T_h_out_x`, `T_c_out_y`, `area_r`, `area_hu`, `area_cu`, and `dqda`.
Those positional arrays must not become the long-term public data model.

Add enum-labelled HEN access following the existing `ProblemTable` pattern:

- Add labels such as `HeatExchangerNetworkLabel.RECOVERY_DUTY`,
  `HOT_UTILITY_DUTY`, `COLD_UTILITY_DUTY`, `RECOVERY_AREA`,
  `HOT_RECOVERY_OUTLET_TEMPERATURE`, `COLD_RECOVERY_OUTLET_TEMPERATURE`,
  `MATCH_ACTIVE`, and `MATCH_ALLOWED`.
- Add type aliases mirroring `ProblemTableColumnKey` where useful so accessors
  accept either enum values or their canonical string labels.
- Keep stream axes labelled by stable OpenPinch stream identity from the
  prepared `StreamCollection`, not only by solver positions `i` and `j`.
- Store the private axis maps needed by the solver adapter:
  hot process stream key -> `i`, cold process stream key -> `j`, stage -> `k`,
  hot utility key -> utility index, and cold utility key -> utility index.
- Require tests showing that reordering input stream rows changes solver array
  positions if necessary but does not change labelled network access,
  exchanger links, or extracted duties for the same stream identities.

This gives OpenHENS' array-heavy algorithm a migration bridge while preserving
OpenPinch's higher-level label-driven access style.

## OpenHENS Source Disposition

Every material OpenHENS source area should have a deliberate outcome:

| OpenHENS source area | Disposition in OpenPinch |
| --- | --- |
| `OpenHENS`, `OpenHensOptions`, keyword-option construction | Drop. Replace with OpenPinch-native service and schema names only. |
| `SynthesisStudy` | Drop as a public owner. Its useful fields become `TargetInput.options` entries consumed through `PinchProblem.design`; no replacement object may own streams, utilities, or cases. |
| `DesignSpace`, `MethodSequence`, `SolveSetup`, `StudyOutputs` | Translate into OpenPinch-native `CONFIG_FIELD_SPECS` entries under a HEN/synthesis configuration group; do not keep OpenHENS class names or make them workflow roots. |
| `HotStream`, `ColdStream`, `HotUtility`, `ColdUtility` | Replace with OpenPinch `StreamSchema`/`UtilitySchema` where fields match. Audit OpenHENS process stream cost fields before deciding whether to extend `StreamSchema` or use an existing/generalized OpenPinch costing path. Keep row-context parser errors. |
| `CaseStudy.from_csv()` | Do not port as runtime API. Convert existing source CSV examples into OpenPinch-compatible JSON fixtures. Utility economics go into `UtilitySchema.price`; capital-cost coefficients go through existing or generalized OpenPinch costing configuration. |
| OpenHENS array-export and apply-to-model helpers | Replace with private `problem_to_solver_arrays(...)` and remove once equation models consume OpenPinch objects directly. |
| `SynthesisTask`, `TaskOutcome`, `StudyManifest`, `StudyOutcome` | Translate useful pieces into OpenPinch-native task/outcome schemas, `TargetOutput.design` payloads, and optional export manifests because these are good data boundaries, not because their OpenHENS names are public contracts. They must reference `PinchProblem`/workspace variant identity rather than own a case. |
| `NetworkSolution` and solved arrays | Translate into `HeatExchangerNetworkSynthesisResult` containing a `HeatExchangerNetwork`; keep raw arrays only as private migration/diagnostic fields while parity tests still need them. |
| `workflow.py` task fan-out | Translate into OpenPinch synthesis workflow code invoked through `PinchProblem.design` and `PinchWorkspace.solve_variant`; keep PDM -> TDM -> ESM semantics and deterministic task IDs. |
| `solvers.py` | Translate into optional synthesis solver backend code with lazy GEKKO/Pyomo imports and explicit missing-binary errors. |
| `artifacts.py` | Translate only useful JSON/CSV export contracts into optional OpenPinch export helpers sourced from `problem.results`; keep pickle and required end-of-workflow artifact writes out of the primary path. |
| `solution_verification.py` | Translate and broaden with OpenPinch object-level invariants. |
| `GenericHENModel`, `PinchDecompModel`, `StageWiseModel`, `HeatExchangerNetworkProblem` | Move internally in one behavior-preserving slice, then progressively rename/internalize around OpenPinch-native concepts once solver regressions are green. |
| `pinch_classes` | Replace with OpenPinch `ProblemTable` and targeting services after Phase 4 parity. |
| `grid_diagram.py`, plots, `open_best.py`, `run.py` | Rebuild as OpenPinch examples/docs or optional visualization utilities only where needed. Do not keep OpenHENS command surfaces. |
| OpenHENS tests and workbook baselines | Use as migration evidence and regression inputs. Long-term tests should live under OpenPinch and assert OpenPinch-native behavior. |

## Migration Phases

### Phase 0: Baseline Freeze and Acceptance Matrix

Scope:

- No production code moves.
- Add or update documentation only.
- Capture current OpenPinch and OpenHENS test commands, dependency constraints,
  solver assumptions, and benchmark artifacts.

Actions:

- Record the OpenHENS benchmark baselines used by solver tests:
  Four-stream and Nine-stream workbook rows, best ESM TAC, quartiles, solved
  counts, stage counts, unit counts, best `dTmin`, and best `min_dQ`.
- Define fixture policy before moving code:
  source OpenHENS CSV examples should be converted into OpenPinch-compatible
  JSON fixtures and then treated as the source fixtures for migrated tests;
  generated solver artifacts stay test-only unless they are curated sample
  outputs; large workbooks should not enter wheels unless explicitly approved.
- Add fast structural snapshots for every converted OpenHENS example JSON:
  parsed stream/utility counts, economics rows, design-grid task counts,
  internal solver-array shapes, stream-axis identity maps, and OpenPinch
  `PinchProblem`/`Zone` preparation success.
- Add order-invariance fixtures before implementation:
  at least one case with stream rows reordered should produce the same labelled
  stream identities, exchanger-link expectations, and target values even if
  private solver array axes move.
- Add a migration acceptance matrix in OpenPinch docs or tests describing which
  metrics must remain invariant.
- Run OpenPinch fast suite and docs build.
- Run OpenHENS fast suite and collect the solver tests.
- Run OpenHENS solver tests on a machine with Couenne/IPOPT available and save
  the generated JSON/CSV artifacts as migration reference artifacts.

Exit criteria:

- The team knows the exact pre-migration scientific baseline.
- Any failing pre-existing test or missing solver is documented before code is
  moved.

Suggested commands:

```bash
rtk uv run pytest
rtk uv run scripts/build_docs.py
rtk uv run pytest -m "not solver"
rtk uv run pytest -m solver
```

### Phase 1: Dependency and Runtime Viability Spike

Scope:

- Packaging and import behavior only.
- No synthesis equations moved yet.

Actions:

- Decide the Python version policy. OpenPinch currently requires `>=3.14`;
  OpenHENS currently requires `>=3.12`. Test Pyomo, GEKKO, Kaleido, and Wakepy
  under OpenPinch's Python target before merging dependencies.
- Add a `synthesis` optional extra to OpenPinch only if dependency resolution
  succeeds.
- Decide whether `synthesis` is included in the `full` extra, and update
  packaging metadata tests accordingly.
- Keep synthesis solver packages out of core and unrelated extras.
- Update dev dependency policy, CI marker policy, and packaging tests for the
  new marked solver tests.
- Keep solver imports lazy so `import OpenPinch` and existing OpenPinch tests do
  not import GEKKO/Pyomo.
- Keep Pyomo/GEKKO imports inside backend modules or functions. Do not repeat
  OpenHENS' eager `SolverFactory` import pattern at package import time.
- Add import smoke tests proving core OpenPinch imports do not load synthesis
  solvers.

Exit criteria:

- `pip install .` or `uv sync` for OpenPinch core remains lightweight.
- `uv sync --extra synthesis` installs the synthesis dependency set, or there is
  a documented version/dependency decision before proceeding.

### Phase 2: Synthesis Schemas and Network Domain in OpenPinch

Scope:

- New schema/domain files and fast validation tests.
- No solver execution.

Actions:

- Translate OpenHENS design-space, method, solve, output, task, and result
  concepts into `OpenPinch.lib.schemas.synthesis` with OpenPinch-native naming.
  Do not port `SynthesisStudy` as a public root and do not port old public
  class names or keyword option shells.
- Introduce `HeatExchanger` and `HeatExchangerNetwork` in `OpenPinch.classes`
  as the public network model before any solver move.
- Add HEN-specific enum labels and type aliases following the `ProblemTable`
  `ProblemTableLabel`/`ProblemTableColumnKey` pattern.
- Implement labelled accessors for exchanger duty, area, activity, stream-link
  endpoints, stage, and utility/recovery kind without requiring users to know
  solver array offsets.
- Define how `HeatExchangerNetworkSynthesisResult` embeds a
  `HeatExchangerNetwork` plus solver/task metadata.
- Do not port OpenHENS Pydantic aliases such as `min_dT_values`,
  `min_dqda_values`, `output_folder`, or `output_formats` unless an
  OpenPinch-native name is intentionally chosen and documented.
- Keep validation strict: positive grids, finite values, valid stage selection,
  valid output formats, positive tolerances, valid run IDs.
- Preserve model serialization behavior and deterministic task ID generation.
- Add export snapshot tests so public names are intentionally added to
  `OpenPinch.lib` and `OpenPinch.services` only when desired.
- Add negative API tests proving there is no `OpenHENS` export, no `openhens`
  package/module path, no `OPENHENS_*` environment-variable contract, and no
  OpenHENS field aliases accepted by the OpenPinch-native schemas.
- Add negative root-primitive tests proving there is no public study/case owner,
  no top-level synthesis runner accepting raw problem inputs, and no HEN
  service overload that bypasses `PinchProblem`.
- Add domain tests proving recovery links are represented as hot process stream
  -> cold process stream, hot utility links as hot utility -> cold process
  stream, and cold utility links as hot process stream -> cold utility.
- Add labelled-access tests proving stream row reordering does not change
  extracted exchanger links or duties for the same OpenPinch stream identities.

Exit criteria:

- New synthesis configuration fields, tasks, results, `HeatExchanger`, and
  `HeatExchangerNetwork` objects can be constructed, serialized, and validated
  without importing GEKKO/Pyomo.
- No new object can be used as a parallel problem/case/workspace root.
- Existing OpenPinch public API tests still pass.

### Phase 3: JSON Fixture Conversion, Problem Adapter, and Unit Bridge

Scope:

- One-time fixture conversion, unit validation, and solver-array adapters.
- Still no solver model move.
- No runtime CSV import support.

Actions:

- Convert existing OpenHENS CSV example cases into standard OpenPinch JSON
  files using `TargetInput` structure.
- Put process stream data in `StreamSchema`, utility data and utility prices in
  `UtilitySchema`, and HEN controls in `TargetInput.options`.
- Keep conversion tooling outside the public runtime API. It can be a one-time
  migration script or checked-in data-prep note, but OpenPinch HEN synthesis
  should not expose CSV loading.
- Load the converted JSON through `PinchProblem` and use the prepared execution
  `Zone` as the source of process and utility stream data.
- Map hot-utility and cold-utility operating costs to `UtilitySchema.price`.
- Map exchanger capital-cost coefficients to existing OpenPinch costing
  configuration where possible. If per-exchanger-kind coefficients are required,
  extend OpenPinch's general costing configuration rather than adding a
  HEN-only economics schema.
- Add all new HEN option keys to `CONFIG_FIELD_SPECS` before converted fixtures
  use them, so `Configuration(options=target_input.options)` validates and
  exposes them on the prepared zone config.
- Preserve the source OpenHENS solver behavior where missing temperature
  contributions become `dTmin / 2` in the internal solver-array adapter.
- Create `problem_to_solver_arrays(problem, dTmin)` as an
  internal bridge so the moved equation models receive exactly the arrays they
  currently expect, including HEN option values, utility prices, and costing
  coefficients sourced from OpenPinch objects/configuration.
- Store labelled axis maps next to the private arrays so any solver result can
  be reconstructed as a `HeatExchangerNetwork` by stream identity.
- Add tests over every converted OpenHENS example JSON fixture.

Exit criteria:

- Every current OpenHENS example case exists as an OpenPinch-compatible JSON
  fixture.
- Array snapshots match the source OpenHENS solver-array payload exactly for
  the same case and `dTmin`.
- The public path for the same data starts with `PinchProblem`; tests fail if
  private arrays are built directly from converted fixture rows, `TargetInput`,
  or a HEN result schema.
- Bad row values fail with row and field context.

### Phase 4: Pinch Target Parity and Replacement Plan

Scope:

- Parity tests around pinch targets and split logic.
- Minimal production changes.

Actions:

- Add a test harness that runs OpenHENS' current `PinchDecompModel.calculate_pinch()`
  logic and OpenPinch's current problem-table/direct-integration logic for the
  same prepared `PinchProblem`.
- Compare hot utility target, cold utility target, heat recovery target, hot
  pinch, cold pinch, shifted pinch temperature, and above/below active stream
  masks across the full OpenHENS `dTmin` grid.
- Confirm `ProblemTable` access uses `ProblemTableLabel` or canonical string
  labels, not raw row/column assumptions that break when stream order changes.
- Include structural PDM parity for `z_i_active`, `z_j_active`, clipped hot and
  cold stream temperatures, `S`, `K`, manual stage selection, and hot/cold
  threshold cases where `HU_target == 0` or `CU_target == 0`.
- Identify any unit or `dt_cont` convention differences explicitly.
- Only after parity passes, replace OpenHENS' private `pinch_classes` usage in
  the migrated `PinchDecompModel` with OpenPinch `ProblemTable` and targeting
  helpers.

Exit criteria:

- The source OpenHENS-adapted pinch implementation can be removed because
  OpenPinch's native implementation proves equivalent for OpenHENS examples.
- Parity covers target values and the structural fields that shape downstream
  TDM/ESM tasks.
- Any non-equivalence is either fixed in an adapter or documented as a blocked
  algorithm decision.

### Phase 5: Workflow, Solver Metadata, Result Cache, and Optional Export Modules

Scope:

- Port OpenHENS orchestration and durable result records into OpenPinch's
  problem-owned result cache.
- Solver execution may still be fake in tests.

Actions:

- Translate the OpenHENS workflow, solver metadata, result, optional export, and
  verification logic into the new OpenPinch synthesis service package with
  OpenPinch-native module and class names.
- Extend `TargetOutput` with an optional `design` field adjacent to `targets`.
- Add a `PinchProblem.design` accessor, implemented in the same spirit as the
  existing `target` and `plot` handles.
- Add a `PinchProblem.design.heat_exchanger_network_synthesis(...)` method that
  calls `heat_exchanger_network_synthesis_service(problem)`.
- Keep `heat_exchanger_network_synthesis_service(...)` internal-facing and
  problem-rooted: its first required argument is the live `PinchProblem`.
- Make the design service update `problem._results` with a `TargetOutput`
  containing preserved or generated `TargetResults` plus the new
  `design` payload for the HEN run.
- Register `heat_exchanger_network_synthesis` as an advanced
  `PinchWorkspace` workflow in the existing workspace workflow dispatcher.
- Extend `run_problem_workflow` so workflow families can dispatch to
  `problem.design` without pretending design workflows are target methods.
- Keep persistent HEN search/method/solver/output controls in
  `TargetInput.options` / `Configuration`; keep utility economics, costing
  configuration, case, and variant ownership in `PinchProblem` and
  `PinchWorkspace`.
- Do not port a public equivalent of OpenHENS `run_synthesis_workflow(...)`.
  The moved workflow module is implementation detail behind `problem.design`
  and workspace dispatch.
- Keep `LocalSynthesisExecutor` behind the optional synthesis dependency path.
- Treat JSON/CSV as optional exports generated from `problem.results` when
  requested. Useful export contracts include `manifest.json`,
  `results/<task_id>.json`, `metrics/solution_metrics.csv`, and
  `metrics/run_summary.csv`, but writing them is not the terminal step of the
  core workflow.
- Have fake executor outputs return `HeatExchangerNetworkSynthesisResult`
  objects containing `HeatExchangerNetwork`, then convert accepted outcomes into
  `TargetOutput.design` payloads stored in `problem._results`.
- Add fake-executor workflow tests in OpenPinch mirroring OpenHENS coverage:
  failed PDM does not spawn TDM, failed TDM does not spawn ESM, task IDs are
  deterministic, topology restrictions are required, and outcomes serialize
  without live solver objects.
- Add workspace tests proving
  `PinchWorkspace.solve_variant(..., workflow="heat_exchanger_network_synthesis")`
  dispatches through `problem.design.heat_exchanger_network_synthesis(...)` on
  the same live `PinchProblem` path as direct problem calls.
- Add result-cache tests proving direct problem and workspace design runs
  populate `problem.results` / `problem._results` as `TargetOutput` with
  `design`, without requiring artifact writes.

Exit criteria:

- OpenPinch can generate, execute with a fake executor, and store synthesis task
  outcomes in `PinchProblem._results` without importing or running GEKKO/Pyomo.
- `TargetOutput` validates with existing `TargetResults` and optional HEN
  `design`; optional export round-trip tests pass only for the export helper
  path.
- HEN synthesis is reachable through `PinchProblem` and `PinchWorkspace`
  without adding independent non-HENS case/workspace management.
- No HEN synthesis method is exposed from `PinchProblem.target`.
- There is no public synthesis runner that accepts anything other than a
  `PinchProblem`-rooted operation.

### Phase 6: Move Existing Equation Models Behind the New Service Boundary

Scope:

- Move `HeatExchangerNetworkProblem`, `GenericHENModel`, `PinchDecompModel`,
  and `StageWiseModel` into OpenPinch.
- Preserve equations and solver defaults.
- Do not move visualization code as part of the model core.

Actions:

- Split the move into reviewable sub-slices:
  1. equation-kernel base and backend setup,
  2. stagewise TDM/ESM equation construction,
  3. PDM coordinator and above/below-pinch construction,
  4. stage-reduction logic,
  5. topology evolution logic.
- Move each sub-slice into
  `OpenPinch/services/heat_exchanger_network_synthesis/models` with private or
  OpenPinch-native class names.
- Update imports to use OpenPinch-owned schemas, adapters, solver wrappers, and
  verification modules.
- Keep all GEKKO variable names, objective formulas, stage-reduction logic, and
  evolution logic unchanged in this phase.
- Convert solved `Q_r`, `Q_h`, `Q_c`, temperature, area, and binary arrays into
  `HeatExchangerNetwork` immediately at the solution extraction boundary.
- Keep raw solver arrays in task outcomes only as private diagnostic/parity
  data until benchmark and shadow tests no longer need them.
- Replace duplicate LMTD and costing code only where tests prove exact or
  tolerance-equivalent behavior.
- Add solver import guards with actionable errors for missing optional
  dependencies or missing external solver binaries.
- Defer `grid_diagram.py` and plot rendering to an optional visualization
  module after core synthesis parity is green.

Exit criteria:

- OpenPinch synthesis solver tests reproduce the same Four-stream and
  Nine-stream baselines as OpenHENS.
- Each model sub-slice has focused tests or solver-regression evidence before
  the next sub-slice lands.
- Existing OpenPinch tests still pass without the synthesis extra.
- The moved model code does not leak GEKKO/Pyomo imports into normal OpenPinch
  imports.

### Phase 7: Public Service and Documentation

Scope:

- Expose the new user-facing synthesis workflow from OpenPinch.
- Provide a clear, direct migration guide from OpenHENS concepts to
  OpenPinch-native names.

Actions:

- Add a service entry point such as
  `heat_exchanger_network_synthesis_service(problem) ->
  HeatExchangerNetworkSynthesisResult`.
- If an object-oriented workflow is desired, use an OpenPinch-native class name
  such as `HeatExchangerNetworkSynthesis(problem).solve()`, not `OpenHENS`;
  this class must require a live `PinchProblem`, read HEN configuration from the
  problem, and must not accept raw input data.
- Document the new OpenPinch workflow and the explicit name mapping from
  OpenHENS concepts to OpenPinch synthesis classes.
- Add examples converted from the current OpenHENS README.
- Show the primary examples as:
  `PinchProblem(...) -> problem.design.heat_exchanger_network_synthesis(...)`
  and
  `PinchWorkspace(...).solve_variant(..., workflow="heat_exchanger_network_synthesis")`.

Exit criteria:

- A user can express the current OpenHENS README example through the
  OpenPinch-native synthesis API.
- The result exposes a `HeatExchangerNetwork` whose exchangers describe
  source/sink stream links such as hot process stream -> cold process stream.
- Public API tests prove only the intended OpenPinch-native names are exported.
- Documentation may include a name mapping from OpenHENS to OpenPinch, but it
  must not imply import aliases, field aliases, or command parity.

### Phase 8: Replace Duplicate Helpers Incrementally

Scope:

- Small, reviewable reductions of migrated OpenHENS source code.
- Each replacement has focused parity tests.

Candidate replacements:

- Replace `pinch_classes` with OpenPinch problem-table/direct-integration
  services once Phase 4 parity is green.
- Replace hand-coded LMTD post-processing with
  `OpenPinch.utils.heat_exchanger.compute_LMTD_from_dts` where denominator and
  tolerance behavior match.
- Replace annualized exchanger cost helpers with `OpenPinch.utils.costing`
  where formulas match. Keep OpenHENS' TAC convention explicit if it differs
  from OpenPinch area/cost targeting.
- Replace ad hoc output tables with OpenPinch export helpers only when the
  synthesis artifact CSV contract remains unchanged.

Exit criteria:

- Less duplicate code with no TAC, topology, utility, or artifact drift.
- Each helper replacement is independently revertible.

### Phase 9: Expand Regression Coverage

Scope:

- Strong validation after the migration path exists.

Actions:

- Keep the Four-stream and Nine-stream solver regression tests as mandatory
  solver baselines.
- Add marked solver tests for more OpenHENS example cases in tiers:
  small cases first, then medium cases, then large benchmark cases.
- For every solver baseline, compare:
  best ESM TAC, quartiles, within-2/5/10 percent counts, attempted jobs, solved
  ESM count, best stage count, recovery/CU/HU unit counts, best `dTmin`, best
  derivative threshold, hot/cold utility loads, and verification failures.
- Compare `HeatExchangerNetwork` content by stream identity:
  recovery exchanger source/sink/stage/duty/area, hot utility links, cold
  utility links, utility loads, and unit counts. Do not rely only on raw
  positional arrays.
- Add numerical invariant tests that do not depend on benchmark workbooks:
  stream heat balances, stage heat balances, approach-temperature feasibility,
  utility cost recomputation, area cost recomputation, no negative areas, and
  no active exchanger with impossible temperature ordering.
- Add temporary shadow-run tests during the transition: run the frozen OpenHENS
  reference implementation and OpenPinch synthesis on the same case and compare
  artifacts. Remove those tests once OpenPinch owns the implementation and
  benchmark baselines.

Exit criteria:

- Solver regressions are broad enough that equation changes are low risk.
- Fast tests still cover all non-solver orchestration and serialization logic.

### Phase 10: Retire OpenHENS as an Independent Implementation

Scope:

- Repository and documentation cleanup after OpenPinch has proven parity.

Actions:

- Archive or otherwise freeze the OpenHENS repository after the OpenPinch
  synthesis API has proven parity.
- Do not maintain a thin wrapper package. Users should migrate to the
  OpenPinch-native API directly.
- Move documentation, examples, and benchmark data that should live long-term
  into OpenPinch.
- Add deprecation notes to OpenHENS README and package metadata only after the
  OpenPinch release is available.

Exit criteria:

- There is one maintained implementation of HEN synthesis.
- Users have a documentation-only mapping from source OpenHENS names to
  OpenPinch-native imports, with no import aliases.

## Validation Strategy

### Fast Tests

These should run in default OpenPinch CI without solver binaries:

- Schema construction and validation failures.
- Negative public API checks for absent OpenHENS names, aliases, import paths,
  and environment-variable contracts.
- Root primitive checks proving HEN synthesis cannot run without a
  `PinchProblem`, and multi-case HEN synthesis cannot bypass `PinchWorkspace`.
- Parallel workflow purge checks proving there is no public synthesis
  case/study owner, raw-input synthesis runner, or HEN-specific workspace
  registry.
- Converted JSON fixture validation and unit conversion.
- Structural snapshots for every converted OpenHENS example JSON fixture.
- Solver-array adapter snapshots.
- `HeatExchanger` and `HeatExchangerNetwork` serialization and label-based
  access.
- Stream order-invariance tests for labelled HEN duties, areas, and source/sink
  links.
- Canonical case-path tests proving JSON and native payloads flow through
  `TargetInput`, `PinchProblem`, `prepare_problem(...)`, `Zone`, `Stream`, and
  `StreamCollection`.
- PinchWorkspace dispatch tests proving HEN synthesis runs through
  `run_problem_workflow` to `problem.design` on a live `PinchProblem`.
- Task graph generation and deterministic task IDs.
- Fake-executor workflow behavior.
- Artifact writing/loading.
- Verification functions against fake solved models.
- Import smoke tests proving synthesis dependencies are lazy.

### Solver Tests

These should be marked separately because they require solver binaries and can
be slow:

- Four-stream and Nine-stream workbook baseline regressions.
- Temporary shadow comparisons against the frozen OpenHENS reference
  implementation during migration.
- Additional example-case regressions added in tiers.

### Regression Tolerances

Start from OpenHENS' existing constants:

- `TAC_REL_TOL = 1e-4`
- `TAC_ABS_TOL = 1.0`
- `MAX_REGRESSION_REL_TOL = 1e-2`
- one-count allowance only for within-threshold buckets when a solution is
  numerically tied at the threshold boundary.

Do not widen tolerances as part of migration unless the change is reviewed as a
scientific decision.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Python version mismatch between OpenPinch and OpenHENS dependencies | Run Phase 1 dependency/runtime viability spike before adding dependencies. Keep synthesis optional and lazy. |
| Unit mismatch between OpenHENS K inputs and OpenPinch canonical units | Use explicit unit payloads and array snapshot tests. Never silently assume defaults. |
| Input adapters bypass OpenPinch's preparation pipeline | Enforce the canonical synthesis problem contract with tests from JSON/native payload to `TargetInput`, `PinchProblem`, prepared `Zone`, and private solver arrays. |
| HEN code creates parallel non-HENS case/workspace management | Treat `PinchProblem` and `PinchWorkspace` as mandatory roots. Add negative API tests for public study/case/workspace roots and raw-input synthesis runners. Keep persistent HEN controls in `TargetInput.options` / `Configuration` and design outputs in `TargetOutput.design`; only transient execution metadata and optional exported files should live outside the problem result cache. |
| Stream row ordering changes network interpretation | Use enum-labelled HEN accessors and stable OpenPinch stream identities. Add row-reordering fixtures that compare `HeatExchangerNetwork` links, not only raw arrays. |
| Solver availability differs across machines | Keep solver tests marked. Provide actionable missing-solver errors and collect-only CI checks. |
| OpenPinch multi-utility concepts exceed current OpenHENS single-utility solver arrays | Preserve single-HU/single-CU synthesis behavior first. Treat multi-utility HEN synthesis as a later feature. |
| Moving equation code changes numerical behavior accidentally | Move equations unchanged first, with shadow tests and workbook baselines. Refactor only in small later phases. |
| Core OpenPinch import becomes heavy | Use optional extras, lazy imports, and import smoke tests. |
| Optional export contract drifts | Keep JSON/CSV export schemas snapshot-tested and compare metrics against OpenHENS baselines, while separately testing that the canonical in-memory result is `TargetOutput` with `design`. |

## Non-Goals for the First Migration

- Rewriting GEKKO equations into pure Pyomo.
- Adding distributed execution or remote task queues.
- Adding new HEN synthesis algorithms.
- Generalizing HEN synthesis to multiple utilities beyond current OpenHENS
  behavior.
- Changing solver defaults, tolerances, objective formulas, or evolution
  heuristics.
- Making pickle persistence a primary workflow.

## Recommended Review Slices

1. Baseline and docs only.
2. Optional dependency scaffolding and lazy import tests.
3. Schema shell, `HeatExchanger`, `HeatExchangerNetwork`, and label-access
   validation tests, including root-primitive negative tests.
4. JSON fixture conversion, unit validation, `PinchProblem` adapter, axis-map,
   and array snapshot tests.
5. Pinch target parity tests.
6. `PinchProblem.design`/`PinchWorkspace` workflow, artifact, and verification
   translation with fake executor tests and parallel-workflow purge checks.
7. Existing equation model base move.
8. Stagewise and PDM model moves with solver regressions.
9. Stage-reduction/evolution moves with solver regressions.
10. Public service and docs.
11. Duplicate-helper replacements, one helper family at a time.
12. OpenHENS archive cleanup.

This order keeps each pull request understandable and keeps the scientific
result protected before the riskiest code is moved.
