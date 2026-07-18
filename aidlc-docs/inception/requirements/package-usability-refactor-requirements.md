# Package Usability Refactor Requirements

## Intent Analysis

- **Request**: Refactor the OpenPinch package and expand its tutorial suite as
  needed so normal engineering workflows are simple, intuitive, and taught
  through the supported public API.
- **Request type**: User-facing API refactor, defect correction, and tutorial
  redesign.
- **Scope**: Multiple components spanning application accessors, workspace
  orchestration, result presentation, HEN selection/plotting, multiperiod
  aggregation, documentation, notebooks, and tests.
- **Complexity**: High. Seven notebooks currently fail against the live package,
  and several failures reveal missing or ambiguous public workflow operations.
- **Compatibility**: Clean break. No legacy aliases, positional compatibility,
  or forwarding methods are required. `pinch_analysis_service` is retiring and
  is not a protected contract, tutorial path, or acceptance dependency.

## Product Principles

1. A first-time user should need only
   `from OpenPinch import PinchProblem, PinchWorkspace` for stateful workflows.
2. `PinchProblem` owns targeting workflow selection. Every common analysis has
   an explicit, discoverable `problem.target.*` method; input configuration
   contains numerical assumptions, not booleans that secretly select analyses.
3. Tutorials must never import private or concrete implementation owners.
4. Common workflow methods accept friendly, validated keyword arguments and
   never require an OpenPinch-owned closed string answer; flat internal option
   keys remain an advanced escape hatch.
5. User-facing results expose descriptive scalar fields and stable collection
   operations without private conversion or selection helpers.
6. Every tutorial is executable from a clean kernel under its declared optional
   dependency profile.
7. The tutorial suite is capability-complete for the supported process-engineer
   surface, including selected-period and multiperiod workflows.

## Functional Requirements

### FR-1: Canonical package boundary

- Keep the package root limited to `PinchProblem` and `PinchWorkspace`.
- Retire `pinch_analysis_service` from supported imports, documentation,
  tutorials, capability matrices, and contract tests; all process-engineer
  workflows use `PinchProblem` or `PinchWorkspace`.
- Remove all tutorial imports from `OpenPinch.application`,
  `OpenPinch.analysis`, `OpenPinch.domain`, `OpenPinch.presentation`, and
  private underscore modules.

### FR-2: Explicit targeting vocabulary

- Replace the ambiguous `problem.target()` spelling with
  `problem.target.all_heat_integration(...)` while retaining an efficient
  single traversal of the zone hierarchy.
- `all_heat_integration(...)` runs only direct and indirect/Total Site heat
  integration. Heat pumps, refrigeration, cogeneration, exergy, energy
  transfer, and area/cost remain explicit methods and are never launched by the
  bulk heat-integration call.
- The bulk operation runs direct heat integration on every structurally
  applicable zone and indirect heat integration on aggregate zones after their
  child prerequisites are available. Applicability comes from the zone tree,
  not targeting-enable flags.
- Preserve selected-period capabilities, result recording, ordering, return
  shape, and numerical output. Implement one dependency-aware traversal rather
  than repeated public-method calls.
- Adopt descriptive canonical methods:
  - `problem.target.direct_heat_integration(...)`
  - `problem.target.indirect_heat_integration(...)`
  - `problem.target.total_site_heat_integration(...)`
  - `problem.target.all_heat_integration(...)`
  - `problem.target.carnot_heat_pump(...)`
  - `problem.target.carnot_refrigeration(...)`
  - `problem.target.vapour_compression_heat_pump(...)`
  - `problem.target.vapour_compression_refrigeration(...)`
  - `problem.target.brayton_heat_pump(...)`
  - `problem.target.brayton_refrigeration(...)`
  - `problem.target.mvr_heat_pump(...)`
  - `problem.target.heat_exchanger_area_and_cost(...)`
  - existing descriptive methods for energy transfer, cogeneration, and
    exergy where no clearer name adds value.
- `indirect_heat_integration(...)` and
  `total_site_heat_integration(...)` are deliberate public peers over one
  focused indirect/Total Site implementation. They have identical signatures,
  target recording, return types, and numerical behavior and never call each
  other.
- Remove superseded method names in the same clean-break change after all
  internal callers, docs, and tests migrate.
- Do not add ambiguous `direct()`, `indirect()`, `all()`, `area_cost()`, or
  configuration-dependent `configured_analyses()` shorthands.
- Remove `_TargetAccessor.__call__`; `problem.target()` is not retained as a
  compatibility alias.

### FR-3: Friendly workflow configuration

- Remove all ten analysis-selection options and the runtime `config.targeting`
  group:
  - `TARGETING_DIRECT_SITE_ENABLED`
  - `TARGETING_DIRECT_OPERATION_ENABLED`
  - `TARGETING_INDIRECT_PROCESS_ENABLED`
  - `TARGETING_PROCESS_HP_ENABLED`
  - `TARGETING_PROCESS_RFRG_ENABLED`
  - `TARGETING_UTILITY_HP_ENABLED`
  - `TARGETING_UTILITY_RFRG_ENABLED`
  - `TARGETING_TURBINE_ENABLED`
  - `TARGETING_EXERGY_ENABLED`
  - `TARGETING_AREA_COST_ENABLED`
- Reject removed selector keys in new input rather than ignoring them or
  translating them through a compatibility layer.
- Keep thermal, utility, costing, solver, and algorithm parameters as
  configuration because they describe engineering assumptions rather than
  workflow selection.
- Make every explicit `PinchProblem.target.*` method establish its own
  prerequisites without mutating configuration flags or relying on a hidden
  preliminary `problem.target()` run.
- `heat_exchanger_area_and_cost()` explicitly computes its area/cost fields;
  direct heat integration does not conditionally add them through a selector.
- `exergy()` explicitly enriches its selected base target; direct and indirect
  services do not conditionally attach exergy from configuration.
- Heat-pump and refrigeration algorithms use dedicated callables for Carnot,
  vapour-compression, Brayton, and MVR families. Booleans are limited to
  independent binary decisions such as utility placement and cascade versus
  parallel topology.
- HPR loading accepts exactly one effective named form: `load_fraction`,
  `load_duty`, or `period_loads`. It does not accept a load-mode string.
- Normal methods do not accept OpenPinch-owned cycle, target-type, workflow,
  aggregation-mode, graph-type, or HEN-method strings. Strings remain valid for
  user identities and external resources such as zones, periods, cases,
  streams, refrigerants, solvers, paths, and filenames.
- Common HEN choices use method arguments rather than tutorial-owned
  `HENS_*` dictionaries.
- Raw `update_options(...)` remains available for advanced numerical
  configuration but accepts no targeting-selection keys and is not part of the
  beginner path.
- Validation errors name the public argument and list accepted values.

### FR-4: Intuitive workspace scenarios

- Make `workspace.scenario(...)` the canonical scenario-creation operation.
- `scenario(...)` creates and returns a problem but does not accept `solve` or
  a workflow selector.
- `workspace.cases(names)` returns a batch view whose `target`, `design`,
  reporting, and export surfaces mirror the single-problem vocabulary.
- Workspace execution dispatches the selected `PinchProblem.target.*` method
  directly and must not call `problem.target()` as a hidden prerequisite.
- Keep comparisons explicit and deterministic.
- Remove `copy_case(...)` from tutorials and decide whether it remains an
  internal primitive or is removed entirely.
- Use case vocabulary consistently. Retire public `list_variants`,
  `get_variant_input`, `input_view`, `validate_variant`, `set_variant_input`,
  `solve_variant`, `compare_variants`, and `configuration_field_metadata`
  after migrating retained process-engineer functionality.
- Retire redundant `PinchWorkspace.from_json(...)`; the constructor accepts a
  mapping. Keep `load(...)`, `load_bundle(...)`, and `save_bundle(...)` for case
  and workspace lifecycle operations.
- Keep `list_cases`, `case`, `use_case`, `active_case_name`, `scenario`,
  `cases`, `compare_cases`, and `compare_to` as the canonical multi-case study
  surface; retire string-dispatched `run_cases`.
- Make workspace project and baseline identities read-only after construction or
  bundle loading; case activation changes only through `use_case(...)` or an
  explicit activation argument.
- Keep active-case `target`, `plot`, `problem_data`, `problem_filepath`,
  `results`, and `master_zone` forwarding, and teach when explicit
  `workspace.case(name)` access is clearer.
- Keep active-case validation, configuration, summary, metric, report, export,
  and dashboard forwarding with the same state and no-hidden-execution rules as
  `PinchProblem`.
- Use `workspace.to_problem_json(case_name=...)` as the canonical case-input
  serializer and retire redundant `get_case_input(...)` from the supported
  surface.

### FR-5: Public result operations

- User examples use descriptive fields such as `hot_utility_target`,
  `cold_utility_target`, and `heat_recovery_target`, not mixed `Qh`/`Qc`/`Qr`
  spellings.
- Scalar selected-period values must not require
  `domain._value.resolution.get_scalar_value`.
- Multiperiod summaries must support selected, all-period, weighted-average,
  and all-plus-weighted modes without failing on optional non-aggregable fields.
- The aggregation policy must explicitly distinguish additive, weighted,
  maximum-design, identity, and optional diagnostic fields.

### FR-6: Public HEN design experience

- Provide application-owned operations to retrieve the best design, the top
  `n` outcomes, a network by one-based rank, and a grid figure by rank.
- Keep Pydantic result schemas as data contracts rather than attaching mutable
  application behavior to them.
- Eliminate tutorial imports of presentation services and internal selection
  helpers.

### FR-7: Tutorial redesign

- Expand beyond the existing ten notebooks where separation improves
  learnability; give every notebook one primary learning outcome and a declared
  level: core, intermediate, or advanced.
- Maintain a version-controlled tutorial coverage manifest mapping every
  supported lifecycle, target, component, design, result, plot, report, export,
  dashboard, serialization, comparison, and advanced inspection operation to at
  least one notebook.
- Add a dedicated multi-segment stream tutorial that constructs or loads
  piecewise stream data, validates segment ordering and continuity, inspects the
  prepared segments, and uses them in heat-integration analysis.
- Provide distinct executable tutorials for multiperiod heat integration,
  multiperiod heat-pump analysis, multiperiod cogeneration, and multiperiod HEN
  synthesis. Each must show ordered period results and the appropriate
  aggregate or shared-design interpretation.
- Move `dt_cont` scenario sensitivity out of the first-solve core path.
- Reframe the VC+MVR material around the supported application workflow instead
  of direct internal thermodynamic classes.
- Use packaged sample names directly unless a tutorial explicitly teaches file
  copying.
- Remove broad exception swallowing, repeated target scans, repeated solves,
  unnecessary temporary-file setup, stale cross-references, and committed
  outputs/execution counts.

### FR-8: Executable tutorial contracts

- Replace substring assertions that preserve obsolete syntax with semantic
  execution and import-boundary tests.
- Execute each notebook from a clean kernel under its declared dependency and
  runtime class.
- Separate base, slow HPR, and solver-backed HEN tutorial gates without
  pretending an unexecuted notebook passed.
- Compare the live public-surface inventory with the tutorial coverage manifest;
  fail when a supported method is unmapped or a tutorial references a retired
  operation.
- Track documented semantic modes that change workflow behavior, including
  source type, zone scope, configuration precedence, placement, period scope,
  aggregation, workspace selection, HEN method, and plot return/export mode.
- Execute multiperiod examples with at least two periods and assert stable
  period ordering, preserved invocation arguments, and no mutation of stored
  configuration.
- Publish the canonical feature-to-tutorial map on Read the Docs from the same
  machine-readable manifest used by CI; do not maintain an independent copied
  table.
- Link the RTD coverage page from the Examples index, notebook series,
  `PinchProblem` API, `PinchWorkspace` API, and overview capability matrix.

### FR-9: Complete `PinchProblem` interaction contract

#### Lifecycle and input

- `PinchProblem(source, project_name=...)` is the canonical constructor for
  packaged names, paths, mappings, and `TargetInput`; construction loads,
  validates, and prepares the zone model but never runs an analysis.
- `problem.load(source)` reloads or replaces the source, clears derived results
  and designs, and does not target. Calling `load()` without a source is only a
  reload when a reusable source path exists; otherwise it raises clearly.
- `problem.validate()` returns validated input and
  `problem.validation_report()` returns non-raising structured diagnostics;
  neither performs analysis.
- Remove redundant `PinchProblem.from_json(...)` because the constructor already
  accepts a mapping. Make `problem.to_problem_json()` return the canonical
  JSON-compatible input mapping, remove the redundant
  `canonical_problem_json()`, and treat raw `problem_data` access as advanced
  inspection.

#### Configuration and argument precedence

- Expose the prepared effective numerical configuration through a read-only
  `problem.config` view. Keep `problem.update_options(...)` as the explicit
  persistent advanced mutation path; rebuilding invalidates derived state.
- Resolve every configurable `problem.*.*(...)` setting in this order:
  1. explicitly supplied named method kwarg;
  2. explicitly supplied advanced `options` entry;
  3. stored problem configuration;
  4. library default.
- Use an internal omitted-value sentinel rather than treating `None` as omitted,
  so explicit `False`, `0`, empty collections, and nullable `None` values are
  honored.
- Explicit method overrides apply only to that invocation and do not mutate
  stored configuration. Conflicting aliases and unknown kwargs fail clearly.
- Record the resolved effective arguments and their source in run/design
  metadata so results can be reproduced.
- Configuration may control numerical assumptions, algorithm choices, output
  detail, and solver behavior, but it must never decide whether a core target,
  design, component, report, plot, or export method executes.
- Core workflow selectors such as a target method, cycle algorithm,
  all-period replay method, or HEN design method are encoded by dedicated
  callables, never string arguments or config fallback fields. Genuine binary
  engineering decisions may use booleans.
  Remove `HENS_METHOD_SEQUENCE` along with targeting selectors because it stores
  which public design methods to run; retain HEN numerical and solver settings.

#### Targeting and periods

- `problem.target.*` owns all targeting execution and method-specific
  prerequisites. The public surface includes the five heat-integration methods
  in FR-2 plus heat pump, refrigeration, cogeneration, exergy, and energy
  transfer.
- Replace top-level `problem.target_all_periods()` with mirrored methods under
  `problem.target.all_periods`, such as
  `problem.target.all_periods.carnot_heat_pump(...)`. There is no method string or
  default configured target.
- A focused target method returns its selected target. A bulk or all-period
  method returns a structured collection and updates explicit cached-result
  state without changing stored configuration.
- `target.all_periods.all_heat_integration(...)`,
  `target.all_periods.carnot_heat_pump(...)`, and
  `target.all_periods.cogeneration(...)` are supported, documented replay paths
  with the same friendly arguments and precedence as focused methods.
- All-period methods are absent for unsupported backends rather than accepting
  a string that fails later. `workers=1` is serial and values above one use the
  documented process backend.

#### Components and design

- Replace the awkward `problem.add_component.process_mvr(...)` teaching surface
  with `problem.components.add_process_mvr(...)`; adding or changing a component
  invalidates dependent targets and designs but does not run them.
- Make `problem.design.heat_exchanger_network(...)` the canonical HEN entry.
  Common named kwargs override HEN configuration, omitted kwargs use HEN
  configuration, and generic `options` remains advanced.
- Shorten advanced method names to `open_hens(...)`, `pinch_design(...)`,
  `thermal_derivative(...)`, and `network_evolution(...)`; remove redundant
  `_method` suffixes.
- Provide `enhanced_heat_exchanger_network(...)` and
  `multiperiod_heat_exchanger_network(...)` as distinct callables instead of
  quality-tier aliases or `periods="all"` selectors.
- A design method may establish documented required heat-integration targets,
  but the prerequisite is fixed by that method and never selected by a config
  boolean. It must not invoke the removed generic target callable.
- Keep design-result ranking, network selection, grid rendering, and selected
  network metrics on application-owned views.
- Support explicit multiperiod HEN synthesis through
  `design.multiperiod_heat_exchanger_network(...)`; the design consumes
  prepared ordered period targets, records the period set and weights, and does
  not select periods or a synthesis method from configuration.

#### Results, plots, reports, and exports

- `problem.results`, `summary_frame()`, `metrics()`, `report()`, `compare_to()`,
  all `problem.plot.*` methods, `export_excel()`, and `show_dashboard()` consume
  existing results; they never select or run a target method implicitly.
- Remove `solve=True` defaults from reporting operations. Missing results raise
  an actionable error naming the target method the user should call.
- Remove callable `problem.plot()`; use `problem.plot.catalog()` for inventory
  and `problem.plot.data()` for serialized graph records, with named plot methods
  for figures or selected graph data. Plot access never falls back to
  `problem.target()`.
- All file writes and dashboard launches remain explicit side effects.
- `export_excel(output_dir, ...)` receives its destination explicitly; remove
  mutable `problem.results_dir` as a second hidden export configuration path.
- Multiperiod summaries consume explicitly cached all-period results; requesting
  uncached periods does not silently replay an analysis.
- Replace aggregation-mode and format strings on summaries, metrics, reports,
  and Excel export with `detailed`, `include_periods`, and
  `include_weighted_average` booleans.
- Plot kind is selected by a named plot method. Plot export accepts plot method
  references, or omission for all plots, rather than a graph-type string.

#### Inspection and state

- Keep `project_name`, `period_ids`, streams, utilities, prepared zones, input
  data, and process-component collections as read-only or clearly mutating
  inspection surfaces, classified as core or advanced in the API reference.
- Make `problem.project_name` read-only after construction/load rather than
  retaining an assignment side effect that can leave result labels stale.
- Document state transitions for load/configure/component/target/design calls
  and test that invalidation prevents stale results, graphs, reports, or designs.

## Non-Functional Requirements

- **Learnability**: The first complete solve should fit in five logical lines
  after imports and source definition.
- **Discoverability**: Public method names and signatures must be visible through
  normal IDE completion without private-module knowledge.
- **Consistency**: The same vocabulary is used in Python APIs, DataFrame column
  labels, docs, and notebooks.
- **Maintainability**: Notebook checks validate behavior and supported imports,
  not incidental source strings.
- **Performance**: Core notebooks complete quickly; expensive notebooks declare
  their expected runtime and optional extras.
- **Formatting**: Notebook code passes Ruff, contains no trailing whitespace,
  and is formatted for an 88-character teaching width.
- **Determinism**: Tutorial outputs use fixed configuration and seeds where
  optimization is involved.
- **Predictability**: Reading, reporting, plotting, comparing, or exporting
  never chooses or executes a core analysis. Explicit kwargs always have the
  same precedence over stored configuration across accessors.
- **Coverage completeness**: Every supported process-engineer operation is
  taught in at least one executable tutorial, while expensive or interactive
  operations use honest solver, slow, or opt-in execution profiles.

## Acceptance Criteria

1. Every existing and newly added notebook compiles and executes successfully
   under its declared environment.
2. No notebook imports a private or unsupported concrete OpenPinch owner.
3. No notebook uses positional `copy_case`, removed HEN result methods, or the
   private scalar resolver.
4. The real shared-HPR weighted summary scenario has a regression test and no
   longer raises on partially missing optional pinch fields.
5. The core quickstart and notebook 01 use only the two package-root workflow
   classes plus ordinary presentation libraries when needed, and teach
   `problem.target.all_heat_integration()` as the concise complete
   heat-integration path.
6. Notebook Ruff, clean-kernel execution, optional-dependency, Sphinx,
   architecture, and complete non-solver gates pass.
7. No accepted input, sample case, runtime configuration object, public method,
   workspace dispatcher, or test depends on `TARGETING_*_ENABLED` selectors or
   `HENS_METHOD_SEQUENCE` core-method selection.
8. Every public `PinchProblem` method and accessor is classified in the
   interaction contract, and configuration-fallback, state-invalidation, and
   no-hidden-execution tests cover each applicable surface.
9. The tutorial coverage manifest accounts for every supported lifecycle,
   `target`, `components`, `design`, result, `plot`, report, comparison,
   serialization, export, dashboard, and advanced inspection operation.
10. Executable examples demonstrate multiperiod heat integration, heat pumps,
    cogeneration, and HEN synthesis with at least two ordered periods.
11. Signature and stale-symbol checks prove that normal workflows expose no
    OpenPinch-owned closed string selector and that specialized HPR, HEN,
    cogeneration, all-period, plot, and workspace-batch methods are complete.
11. A dedicated example uses multi-segment streams end to end without private
    imports or flattening the streams into a single linear segment.
12. The live `PinchProblem`, `PinchWorkspace`, target, component, design,
    selected-network, and plot inventories match the feature-to-tutorial
    manifest, and every supported operation has executable tutorial coverage.
13. Variant-oriented and other retiring workspace methods are absent before
    they are excluded from the 100 percent coverage denominator.
14. The executable tutorial report shows 100 percent canonical operation and
    required semantic-mode coverage; Markdown-only mentions and skipped runtime
    profiles do not count as executed coverage.
15. Warning-free Sphinx builds publish the RTD coverage map from the canonical
    manifest and fail for stale operations, missing tutorial owners, unsupported
    claimed coverage, or broken cross-references.

## Extension Compliance

- **Security Baseline**: Disabled; N/A to this usability refactor.
- **Resiliency Baseline**: Disabled; N/A to this local library refactor.
- **Partial Property-Based Testing**: Applicable only to pure normalization and
  multiperiod aggregation policies introduced by the implementation.
