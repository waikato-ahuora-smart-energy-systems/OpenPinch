# Package Architecture Modernization Code Generation Plan

## Status and Authority

This document is the detailed, dependency-ordered implementation checklist for
the approved package architecture modernization. It is the single source of
truth for Code Generation once the user approves it.

- [x] Convert the approved architecture plan into an executable checklist.
- [x] Preserve the completed owner-oriented helper reorganization as baseline.
- [x] Preserve `OpenPinch/main.py` as the sole current external Python contract.
- [x] Place reusable optimisation at package level rather than under heat pumps.
- [x] Record quality targets and blocking test gates.
- [x] Obtain explicit approval before changing application code.

## Locked Decisions

- [x] Keep `OpenPinch/main.py` at its current path.
- [x] Preserve the exact public signature:
  `pinch_analysis_service(data: Any, project_name: str = "Project") -> TargetOutput`.
- [x] Preserve `main.py` validation order, return shape, exceptions,
  serialization, ordering, and numerical behaviour.
- [x] Limit `main.py` source changes to owner imports and a stale documentation
  cross-reference unless a failing contract test proves another change is
  necessary.
- [x] Keep `__all__ = ["pinch_analysis_service"]` in `main.py`.
- [x] Treat `from OpenPinch.main import pinch_analysis_service` as the only
  compatibility-protected import.
- [x] Do not preserve root imports, deep imports, advanced service imports,
  internal schema paths, or Python pickle paths.
- [x] Remove compatibility facades, forwarding aliases, migration modules, and
  pickle shims rather than relocating them.
- [x] Keep public input/output behaviour through `main.py` unchanged even when
  the concrete model owners move.
- [x] Keep internal runtime records private to their parent or service owner.
- [x] Do not introduce mixins, service locators, dependency-injection
  frameworks, mutable global registries, or arbitrary line-count limits.
- [x] Do not add dependencies, data migrations, deployment changes, or
  speculative algebraic-solver abstractions.
- [x] Use release version `0.5.0` for the intentional pre-1.0 structural break.
- [x] Keep Security and Resiliency extensions disabled.
- [x] Enforce the selected partial Property-Based Testing rules: PBT-02,
  PBT-03, PBT-07, PBT-08, and PBT-09.

## Architectural Boundaries

- [x] `domain` owns business state, invariants, arithmetic, indexing, and
  parent-owned records; it performs no filesystem, plotting, dashboard, or
  solver-backend work.
- [x] `contracts` owns external input/output schemas used by `main.py`; contract
  models do not import application, analysis, adapters, or presentation.
- [x] `optimisation` owns reusable scalar numerical optimisation mechanics and
  imports only the standard library, NumPy, and SciPy.
- [x] `application` owns use-case orchestration, workflow state, caches, and
  coordination; it delegates calculations and I/O.
- [x] `analysis` owns deterministic engineering calculations and service-level
  algorithms, including heat-pump and HEN analysis.
- [x] `adapters` owns third-party and infrastructure translation, including
  file formats and optional solver or plotting integration.
- [x] `presentation` owns reports, tables, diagrams, dashboards, and exports.
- [x] Dependencies point inward toward contracts/domain or sideways through
  explicit typed values; lower layers never import application or presentation.
- [x] Optional dependencies are imported only in their owning adapter or
  presentation leaf and fail with the package's standard actionable error.
- [x] Cross-package calls use explicit function arguments or immutable result
  models rather than hidden globals or parent-barrel imports.

## Target Package Map and Planning Estimates

Line estimates are physical lines, including imports and docstrings, with an
expected variance of approximately 20 percent. They guide cohesion reviews and
work sequencing; they are not acceptance limits.

### Root and Contracts

- [x] `OpenPinch/__init__.py` - package marker only, about 5 lines.
- [x] `OpenPinch/main.py` - sole external contract, about 47 lines.
- [x] `OpenPinch/__main__.py` - command entry orchestration, about 90 lines.
- [x] `OpenPinch/resources.py` - packaged-resource lookup, about 280 lines.
- [x] `OpenPinch/contracts/input.py` - `TargetInput` and nested input contracts,
  about 520 lines.
- [x] `OpenPinch/contracts/output.py` - `TargetOutput` and result contracts,
  about 420 lines.
- [x] `OpenPinch/contracts/synthesis/common.py` - shared synthesis contract
  values, about 120 lines.
- [x] `OpenPinch/contracts/synthesis/topology.py` - topology models, about 210
  lines.
- [x] `OpenPinch/contracts/synthesis/method.py` - synthesis methods/options,
  about 160 lines.
- [x] `OpenPinch/contracts/synthesis/task.py` - synthesis task contracts, about
  210 lines.
- [x] `OpenPinch/contracts/synthesis/result.py` - synthesis result contracts,
  about 260 lines.

### Domain

- [x] `OpenPinch/domain/configuration.py` - configuration state and invariants,
  about 300 lines.
- [x] `OpenPinch/domain/value.py` - `Value` state and arithmetic, about 520
  lines.
- [x] `OpenPinch/domain/_value/coercion.py` - accepted value coercion, about 150
  lines.
- [x] `OpenPinch/domain/_value/units.py` - unit normalization and conversion,
  about 230 lines.
- [x] `OpenPinch/domain/stream.py` - `Stream` aggregate and mutation API, about
  720 lines.
- [x] `OpenPinch/domain/_stream/segment.py` - private `StreamSegment`, about 190
  lines.
- [x] `OpenPinch/domain/_stream/segments.py` - attachment and transaction logic,
  about 330 lines.
- [x] `OpenPinch/domain/_stream/profile.py` - temperature/heat profiles, about
  210 lines.
- [x] `OpenPinch/domain/_stream/thermodynamics.py` - stream thermal
  calculations, about 250 lines.
- [x] `OpenPinch/domain/_stream/value_state.py` - owned immutable value views,
  about 180 lines.
- [x] `OpenPinch/domain/stream_collection.py` - collection behaviour, about 620
  lines.
- [x] `OpenPinch/domain/_stream_collection/filters.py` - domain filters, about
  180 lines.
- [x] `OpenPinch/domain/_stream_collection/sorting.py` - stable ordering, about
  130 lines.
- [x] `OpenPinch/domain/_stream_collection/numeric_view.py` - numeric
  projection, about 190 lines.
- [x] `OpenPinch/domain/_stream_collection/serialization.py` - domain data
  representation, about 170 lines.
- [x] `OpenPinch/domain/problem_table.py` - `ProblemTable` aggregate and public
  operations, about 650 lines.
- [x] `OpenPinch/domain/_problem_table/constants.py` - interval constants, about
  80 lines.
- [x] `OpenPinch/domain/_problem_table/equality.py` - tolerant equality rules,
  about 170 lines.
- [x] `OpenPinch/domain/_problem_table/intervals.py` - interval insertion engine,
  about 680 lines.
- [x] `OpenPinch/domain/zone.py` - zone state and targeting invariants, about 540
  lines.
- [x] `OpenPinch/domain/heat_exchanger.py` - `HeatExchanger` aggregate, about
  650 lines.
- [x] `OpenPinch/domain/_heat_exchanger/period_state.py` - private
  `HeatExchangerPeriodState`, about 170 lines.
- [x] `OpenPinch/domain/_heat_exchanger/area.py` - private
  `HeatExchangerAreaSlice`, about 160 lines.
- [x] `OpenPinch/domain/heat_exchanger_network.py` - network state and queries,
  about 820 lines.
- [x] `OpenPinch/domain/targets.py` - runtime targeting result values, about 580
  lines.

### Shared Optimisation

- [x] `OpenPinch/optimisation/models.py` - problem, options, candidate, and
  result models, about 180 lines.
- [x] `OpenPinch/optimisation/errors.py` - optimisation-specific failures, about
  40 lines.
- [x] `OpenPinch/optimisation/candidates.py` - candidate validation, ranking,
  clustering, and polishing, about 320 lines.
- [x] `OpenPinch/optimisation/execution.py` - deterministic serial/parallel run
  coordination, about 180 lines.
- [x] `OpenPinch/optimisation/service.py` - public internal minimisation entry,
  about 220 lines.
- [x] `OpenPinch/optimisation/backends/protocol.py` - backend callable protocol,
  about 50 lines.
- [x] `OpenPinch/optimisation/backends/dual_annealing.py` - SciPy dual annealing,
  about 150 lines.
- [x] `OpenPinch/optimisation/backends/cma_es.py` - CMA-ES implementation, about
  290 lines.
- [x] `OpenPinch/optimisation/backends/bayesian.py` - Bayesian implementation,
  about 410 lines.
- [x] `OpenPinch/optimisation/backends/rbf.py` - radial-basis implementation,
  about 320 lines.

### Application

- [x] `OpenPinch/application/problem.py` - `PinchProblem` use-case facade, about
  760 lines.
- [x] `OpenPinch/application/_problem/accessors/component.py` - component
  access, about 180 lines.
- [x] `OpenPinch/application/_problem/accessors/design.py` - design access,
  about 210 lines.
- [x] `OpenPinch/application/_problem/accessors/plot.py` - presentation request
  shaping, about 140 lines.
- [x] `OpenPinch/application/_problem/accessors/target.py` - target access,
  about 190 lines.
- [x] `OpenPinch/application/_problem/input/loading.py` - contract-to-domain
  construction, about 300 lines.
- [x] `OpenPinch/application/_problem/input/semantics.py` - cross-object
  semantic checks, about 620 lines.
- [x] `OpenPinch/application/_problem/input/validation.py` - validation report
  assembly, about 360 lines.
- [x] `OpenPinch/application/_problem/output/reporting.py` - report request
  orchestration, about 220 lines.
- [x] `OpenPinch/application/_problem/output/result_extraction.py` - result
  assembly, about 300 lines.
- [x] `OpenPinch/application/_problem/periods/aggregation.py` - multiperiod
  aggregation, about 340 lines.
- [x] `OpenPinch/application/_problem/periods/execution.py` - per-period
  execution, about 300 lines.
- [x] `OpenPinch/application/_problem/targeting/dispatch.py` - target dispatch,
  about 210 lines.
- [x] `OpenPinch/application/_problem/targeting/execution.py` - target
  execution, about 430 lines.
- [x] `OpenPinch/application/_problem/targeting/plan.py` - immutable target-run
  specification, about 180 lines.
- [x] `OpenPinch/application/workspace.py` - `PinchWorkspace` use-case facade,
  about 560 lines.
- [x] `OpenPinch/application/_workspace/case_inputs.py` - case normalization,
  about 260 lines.
- [x] `OpenPinch/application/_workspace/comparison.py` - variant comparison,
  about 300 lines.
- [x] `OpenPinch/application/_workspace/execution.py` - variant solving, about
  310 lines.
- [x] `OpenPinch/application/_workspace/state.py` - cache and active-case state,
  about 260 lines.
- [x] `OpenPinch/application/_workspace/views/input.py` - input view shaping,
  about 160 lines.
- [x] `OpenPinch/application/_workspace/views/graph.py` - graph request shaping,
  about 170 lines.
- [x] `OpenPinch/application/_workspace/views/problem_table.py` - table request
  shaping, about 180 lines.
- [x] `OpenPinch/application/_workspace/views/comparison.py` - comparison view
  shaping, about 180 lines.

### Analysis, Adapters, and Presentation

- [x] `OpenPinch/analysis/targeting/` - cascade, composite-curve, GCC, utility,
  and total-site calculations in cohesive 150-500 line modules.
- [x] `OpenPinch/analysis/graphs/specifications.py` - private graph
  specifications, about 220 lines.
- [x] `OpenPinch/analysis/graphs/composite.py` - composite builders, about 420
  lines.
- [x] `OpenPinch/analysis/graphs/grand_composite.py` - GCC builders, about 400
  lines.
- [x] `OpenPinch/analysis/graphs/metadata.py` - graph metadata, about 190 lines.
- [x] `OpenPinch/analysis/energy_transfer/selection.py` - transfer selection,
  about 210 lines.
- [x] `OpenPinch/analysis/energy_transfer/cascade.py` - cascade transformation,
  about 360 lines.
- [x] `OpenPinch/analysis/energy_transfer/diagram.py` - diagram data, about 380
  lines.
- [x] `OpenPinch/adapters/io/json.py` - JSON loading/writing, about 180 lines.
- [x] `OpenPinch/adapters/io/csv.py` - CSV input/output, about 180 lines.
- [x] `OpenPinch/adapters/optional_dependencies.py` - dependency guards, about
  130 lines.
- [x] `OpenPinch/presentation/reporting/problem_table.py` - table export, about
  260 lines.
- [x] `OpenPinch/presentation/reporting/stream_collection.py` - stream export,
  about 220 lines.
- [x] `OpenPinch/presentation/reporting/results.py` - result report assembly,
  about 340 lines.
- [x] `OpenPinch/presentation/graphs/plotly.py` - Plotly conversion, about 420
  lines.
- [x] `OpenPinch/presentation/dashboard/dependencies.py` - Streamlit guards,
  about 80 lines.
- [x] `OpenPinch/presentation/dashboard/state.py` - private dashboard state,
  about 180 lines.
- [x] `OpenPinch/presentation/dashboard/rendering.py` - dashboard orchestration,
  about 520 lines.
- [x] `OpenPinch/presentation/dashboard/exports.py` - dashboard exports, about
  220 lines.
- [x] `OpenPinch/presentation/network_grid/geometry.py` - grid layout, about 420
  lines.
- [x] `OpenPinch/presentation/network_grid/labels.py` - labels, about 230 lines.
- [x] `OpenPinch/presentation/network_grid/temperatures.py` - temperature
  mapping, about 260 lines.
- [x] `OpenPinch/presentation/network_grid/plotly.py` - Plotly adapter, about 360
  lines.

### Heat Pumps and Heat Exchanger Networks

- [x] `OpenPinch/analysis/heat_pumps/optimisation_adapter.py` - HPR objective,
  penalty, and result translation, about 140 lines.
- [x] `OpenPinch/analysis/heat_pumps/direct_mvr/` - public models plus private
  thermodynamics, units, and execution modules, generally 150-450 lines each.
- [x] `OpenPinch/analysis/heat_pumps/process_mvr.py` - public component/factory,
  about 520 lines.
- [x] `OpenPinch/analysis/heat_pumps/_process_mvr/` - private selection,
  replacement-stream construction, work accounting, membership, and record
  state modules, generally 120-320 lines each.
- [x] `OpenPinch/analysis/heat_pumps/_multiperiod/` - private period
  preparation, period cases, shared-design execution, and aggregation modules,
  generally 180-420 lines each.
- [x] `OpenPinch/analysis/heat_pumps/cycles/` - cohesive cycle models retained
  unless duplicated orchestration is discovered.
- [x] `OpenPinch/analysis/heat_exchanger_networks/models/base.py` - base model
  coordination, about 620 lines.
- [x] `OpenPinch/analysis/heat_exchanger_networks/models/_base/parameters.py` -
  parameter loading, about 340 lines.
- [x] `OpenPinch/analysis/heat_exchanger_networks/models/_base/piecewise.py` -
  piecewise equations, about 520 lines.
- [x] `OpenPinch/analysis/heat_exchanger_networks/models/_base/approach.py` -
  approach constraints, about 360 lines.
- [x] `OpenPinch/analysis/heat_exchanger_networks/models/_base/execution.py` -
  solver execution, about 310 lines.
- [x] `OpenPinch/analysis/heat_exchanger_networks/models/stagewise.py` -
  stagewise coordination, about 650 lines.
- [x] `OpenPinch/analysis/heat_exchanger_networks/models/_stagewise/setup.py` -
  axes and model setup, about 470 lines.
- [x] `OpenPinch/analysis/heat_exchanger_networks/models/_stagewise/equations.py`
  - equations in preserved order, about 760 lines.
- [x] `OpenPinch/analysis/heat_exchanger_networks/models/_stagewise/warm_start.py`
  - warm starts, about 430 lines.
- [x] `OpenPinch/analysis/heat_exchanger_networks/models/_stagewise/evolution.py`
  - evolution and private records, about 420 lines.
- [x] `OpenPinch/analysis/heat_exchanger_networks/models/_stagewise/objectives.py`
  - objective construction, about 280 lines.
- [x] `OpenPinch/analysis/heat_exchanger_networks/models/_stagewise/postprocess.py`
  - post-processing, about 430 lines.
- [x] `OpenPinch/analysis/heat_exchanger_networks/models/_stagewise/verification.py`
  - result verification, about 300 lines.
- [x] `OpenPinch/analysis/heat_exchanger_networks/models/pinch_decomposition.py`
  - pinch-design coordination, about 560 lines.
- [x] `OpenPinch/analysis/heat_exchanger_networks/models/_pinch_design/preprocessing.py`
  - period-native preparation, about 620 lines.
- [x] `OpenPinch/analysis/heat_exchanger_networks/models/_pinch_design/equations.py`
  - decomposition equations, about 720 lines.
- [x] `OpenPinch/analysis/heat_exchanger_networks/models/_pinch_design/amalgamation.py`
  - period result amalgamation, about 520 lines.
- [x] `OpenPinch/analysis/heat_exchanger_networks/models/_pinch_design/postprocess.py`
  - network post-processing, about 390 lines.
- [x] `OpenPinch/analysis/heat_exchanger_networks/extraction/recovery.py` -
  recovery matches, about 340 lines.
- [x] `OpenPinch/analysis/heat_exchanger_networks/extraction/utility.py` - utility
  matches, about 320 lines.
- [x] `OpenPinch/analysis/heat_exchanger_networks/extraction/period_state.py` -
  period-state extraction, about 260 lines.
- [x] `OpenPinch/analysis/heat_exchanger_networks/extraction/segment_area.py` -
  area-slice extraction, about 260 lines.
- [x] `OpenPinch/analysis/heat_exchanger_networks/extraction/metadata.py` -
  solver/design metadata, about 180 lines.
- [x] `OpenPinch/analysis/heat_exchanger_networks/controllability.py` - network
  controllability analysis, about 420 lines.

Expected package size after the move is approximately 65,000 physical Python
lines: root 420; contracts 1,400; domain 11,200; optimisation 2,200;
application 6,400; foundational analysis 7,300; heat pumps 11,900; HEN 16,900;
adapters 1,200; and presentation 6,100.

## Part 2 - Dependency-Ordered Generation Checklist

### Step 1 - Freeze the External Contract and Baselines

- [x] Record the clean Git baseline and preserve all unrelated user changes.
- [x] Add `tests/e2e/test_main.py` as the authoritative external contract suite.
- [x] Assert the `OpenPinch.main` import path and exact function signature.
- [x] Assert `main.py` accepts every supported input form currently accepted.
- [x] Assert representative valid inputs produce exact `TargetOutput` model
  types, field ordering, dumps, and numerical values.
- [x] Assert invalid inputs preserve exception classes, messages, locations,
  and validation order.
- [x] Assert `project_name` default and explicit handling remain unchanged.
- [x] Assert a fresh process can import `OpenPinch.main` with optional dashboard,
  heat-pump, and HEN dependencies absent.
- [x] Capture deterministic single-period, multiperiod, segmented-stream, HPR,
  and HEN baselines before moving their owners.
- [x] Capture the current `OpenPinch.main` import-module count and wall-time as a
  diagnostic baseline.
- [x] Run the main contract suite before and after every subsequent unit.
- [x] Update the checklist, state, and audit immediately when Step 1 completes.

### Step 2 - Create Shared Optimisation

- [x] Create immutable `OptimisationProblem`, options, candidate, and result
  models without importing HPR concepts.
- [x] Require the generic objective to return a scalar `float`.
- [x] Make bounds, constraints, initial points, seed, and backend selection
  explicit inputs.
- [x] Implement candidate validation and deterministic ordering.
- [x] Implement candidate clustering and polishing without global registries.
- [x] Implement deterministic multistart execution with serial fallback.
- [x] Move dual annealing, CMA-ES, Bayesian, and RBF implementations into the
  shared backend package.
- [x] Preserve seed expansion, start ordering, tie-breaking, tolerances, and
  backend-specific options exactly.
- [x] Propagate unexpected programming errors instead of converting them to
  feasibility penalties.
- [x] Keep backend availability errors specific and actionable.
- [x] Add backend contract tests using non-HPR scalar objectives.
- [x] Add generated bounds, ordering, result-shape, and reproducibility
  invariants.
- [x] Add a known convex objective for cross-backend correctness checks.
- [x] Verify `optimisation` imports no domain, contracts, application, analysis,
  adapters, or presentation package.
- [x] Run focused optimisation tests, the main contract suite, and Ruff.
- [x] Update the checklist, state, and audit immediately when Step 2 completes.

### Step 3 - Establish Contracts and Domain Owners

- [x] Move `TargetInput` and its nested input schemas to `contracts/input.py`.
- [x] Move `TargetOutput` and nested output schemas to `contracts/output.py`.
- [x] Move synthesis schemas to concrete contract owners with no parent-barrel
  reverse imports.
- [x] Keep contract model names, validation, JSON fields, and main-service dumps
  unchanged.
- [x] Move `Value`, `Stream`, `StreamCollection`, `ProblemTable`, `Zone`,
  `HeatExchanger`, `HeatExchangerNetwork`, `Configuration`, and runtime target
  values to `domain`.
- [x] Keep `StreamSegment`, `HeatExchangerPeriodState`, and
  `HeatExchangerAreaSlice` private and parent-owned.
- [x] Preserve mapping/schema normalization at parent boundaries.
- [x] Preserve immutable owned-value views and transactional parent mutations.
- [x] Preserve interval insertion ordering, equality tolerances, and repeated
  insertion behaviour.
- [x] Remove export, dataframe, plotting, dashboard, and solver methods from
  domain owners; replace their internal callers with presentation or analysis
  functions.
- [x] Ensure domain serialization helpers return plain domain data and perform
  no filesystem I/O.
- [x] Add contract round-trip properties for supported input/output values.
- [x] Add domain transaction, ordering, conservation, ownership, copy, pickle,
  and serialization properties using seed `20260715`.
- [x] Add AST dependency tests proving domain and contracts have no outward
  layer imports.
- [x] Run focused domain/contract tests, the main contract suite, and Ruff.
- [x] Update the checklist, state, and audit immediately when Step 3 completes.

### Step 4 - Move Application Orchestration

- [x] Move `PinchProblem` to `application/problem.py` without changing its
  internal caller-visible behaviour during the migration.
- [x] Move `PinchWorkspace` to `application/workspace.py`.
- [x] Keep use-case methods as thin coordinators over domain and analysis.
- [x] Split problem accessors by component, design, plot request, and target.
- [x] Split loading, semantic validation, and validation report assembly.
- [x] Split result extraction from report request orchestration.
- [x] Split period execution from period aggregation.
- [x] Represent target execution intent with an immutable private plan value.
- [x] Split target dispatch from target execution.
- [x] Split workspace case inputs, variant execution, comparison, cache/state,
  and view shaping.
- [x] Pass dependencies and state explicitly; do not use hidden module globals.
- [x] Preserve cache invalidation, rollback, active-case identity, and exception
  restoration on success and failure.
- [x] Keep application free of concrete Plotly, Streamlit, filesystem, and
  solver-backend imports.
- [x] Add caller-level tests for targeting, workspace variants, cache
  invalidation, failed replay, and result extraction.
- [x] Run focused application tests, the main contract suite, and Ruff.
- [x] Update the checklist, state, and audit immediately when Step 4 completes.

### Step 5 - Separate Foundational Analysis, Adapters, and Presentation

- [x] Move cascade, composite-curve, GCC, utility, and total-site calculations
  to deterministic analysis modules.
- [x] Split graph specifications, builders, and metadata; keep graph runtime
  records private.
- [x] Split energy-transfer selection, cascade transformation, and diagram-data
  construction.
- [x] Move JSON and CSV filesystem work into adapters.
- [x] Move optional dependency guards to adapter/presentation leaves.
- [x] Move `ProblemTable` and `StreamCollection` exports to presentation
  reporting modules.
- [x] Move Plotly conversion out of analysis and application.
- [x] Split Streamlit dependency guards, state, dashboard rendering, and exports.
- [x] Split network-grid geometry, labels, temperature mapping, and Plotly
  adaptation.
- [x] Ensure deterministic analysis outputs are plain values that can be tested
  without plotting or dashboard dependencies.
- [x] Preserve graph coordinates, labels, metadata, JSON structures, table
  ordering, and exported values.
- [x] Add adapter contract tests at filesystem or library boundaries.
- [x] Add presentation snapshot/semantic tests that assert rendered data rather
  than private helper calls.
- [x] Add cold-import tests with Plotly, Streamlit, and optional solvers absent.
- [x] Run focused analysis/adapter/presentation tests, main contract tests,
  notebook parsing, and Ruff.
- [x] Update the checklist, state, and audit immediately when Step 5 completes.

### Step 6 - Reorganize Heat-Pump Analysis Around Shared Optimisation

- [x] Keep direct-MVR settings, output units, stage results, stream solve
  results, and solve functions at their concrete heat-pump owners.
- [x] Separate direct-MVR models, thermodynamics, unit conversion, and execution.
- [x] Keep `ProcessMVRComponent` and its factory as concrete service-level
  owners.
- [x] Keep Process MVR membership and stream records private.
- [x] Split Process MVR selection, replacement-stream construction, work
  accounting, membership, and record state.
- [x] Move private HPR period preparation, period cases, shared-design
  execution, and aggregation into `_multiperiod`.
- [x] Add `optimisation_adapter.py` for HPR-specific dictionary adaptation,
  objective interpretation, feasibility penalties, and result translation.
- [x] Ensure only the HPR adapter knows HPR-specific `obj`, cost-breakdown, and
  penalty conventions.
- [x] Preserve weighted operating cost, maximum capital, feasibility penalty,
  and annualized-total policies.
- [x] Preserve deterministic starts, backend selection, candidate ranking, and
  exact single/multiperiod numerical results.
- [x] Propagate unexpected errors through the HPR adapter.
- [x] Retain cohesive cycle-model files unless analysis proves duplicated
  orchestration.
- [x] Add adapter tests for penalty conversion, unexpected error propagation,
  fallback objective handling, and result translation.
- [x] Add exact HPR fixture parity tests before deleting old paths.
- [x] Run focused HPR tests, available heat-pump integrations, main contract
  tests, and Ruff.
- [x] Update the checklist, state, and audit immediately when Step 6 completes.

### Step 7 - Decompose HEN Models and Extraction

- [x] Move HEN analysis beneath `analysis/heat_exchanger_networks` without
  adding old-path facades.
- [x] Keep base, stagewise, and pinch-decomposition classes as small concrete
  coordinators at their new owners.
- [x] Extract base parameter loading, piecewise equations, approach
  constraints, and solver execution as explicit composition functions.
- [x] Extract stagewise setup, equations, warm starts, evolution, objectives,
  post-processing, and verification.
- [x] Keep stagewise evolution records private to `_stagewise`.
- [x] Extract pinch-design preprocessing, equations, amalgamation, and
  post-processing.
- [x] Split result extraction into recovery, utility, period state, segment
  area, and metadata modules.
- [x] Pass model state explicitly to every composition helper.
- [x] Preserve solver axes, index sets, parameter loading order, equation order,
  warm-start values, tolerances, piecewise mappings, and numerical ordering.
- [x] Preserve period-native targets, utility `dt_cont`, later-period-only
  matches, explicit branch temperatures, and typed period states.
- [x] Keep GEKKO/Pyomo construction and result extraction HEN-owned.
- [x] Do not create `optimisation.algebraic` until a second real consumer proves
  a shared boundary.
- [x] Add equation-order and index-set structural tests only where they protect
  solver semantics.
- [x] Prefer domain-result and exact numerical regression tests over helper-call
  assertions.
- [x] Run focused extraction/preprocessing tests after each extraction group.
- [x] Run available solver tests and canonical HEN tier 0/1 fixtures after each
  equation-bearing move.
- [x] Run the main contract suite and Ruff after each HEN sub-unit.
- [x] Update the checklist, state, and audit immediately when Step 7 completes.

### Step 8 - Retire Old Package Structure and Compatibility Machinery

- [x] Update production imports to concrete `domain`, `contracts`,
  `optimisation`, `application`, `analysis`, `adapters`, and `presentation`
  owners.
- [x] Update tests, scripts, notebooks, examples, and documentation to concrete
  owners or the external `OpenPinch.main` contract as appropriate.
- [x] Reduce `OpenPinch/__init__.py` to a package marker with no re-exports.
- [x] Remove `OpenPinch.classes`.
- [x] Remove `OpenPinch.lib`.
- [x] Remove `OpenPinch.services`.
- [x] Remove `OpenPinch.utils`.
- [x] Remove `OpenPinch.streamlit_webviewer`.
- [x] Remove stale compatibility modules, aliases, `__getattr__` barrels, and
  pickle-path accommodations.
- [x] Remove public-looking internal record names from documentation and
  `__all__` declarations.
- [x] Confirm `StreamSegmentSchema` remains available through the main input
  contract where required, while runtime segment classes remain private.
- [x] Search source, tests, docs, notebooks, built artifacts, and distributions
  for every retired path.
- [x] Assert retired imports fail rather than silently resolving.
- [x] Update `main.py` owner imports and stale doc reference only after all new
  owners are importable.
- [x] Run fresh-process imports and the main contract suite immediately after
  deleting the old packages.
- [x] Update the checklist, state, and audit immediately when Step 8 completes.

### Step 9 - Behavioural Test Architecture

- [x] Organize tests by observable layer: `tests/e2e`, `tests/application`,
  `tests/domain`, `tests/analysis`, `tests/optimisation`, `tests/adapters`, and
  narrowly scoped architecture tests.
- [x] Make `tests/e2e/test_main.py` the authoritative compatibility suite.
- [x] Remove tests whose only assertion is a private helper path, forwarding
  call, import alias, or internal implementation detail.
- [x] Keep private-module tests only for mathematical kernels, solver equation
  order, architecture rules, or failure localization that cannot be expressed
  through a stable owner.
- [x] Replace internal monkeypatching with explicit dependency seams or
  caller-visible fixtures where practical.
- [x] Preserve exact regression fixtures for numerical algorithms and critical
  business scenarios.
- [x] Add AST tests for allowed dependency directions and forbidden parent
  barrel imports.
- [x] Add fresh-process cold-import tests for the root marker, main contract,
  contracts, domain, optimisation, application, dashboard leaves, HPR, and HEN.
- [x] Add wheel-installed tests proving only the main contract is compatibility
  protected.
- [x] Add `Value` and contract serialization round-trip properties (PBT-02).
- [x] Add segment normalization, transaction, interval insertion, candidate
  ordering, and conservation invariants (PBT-03).
- [x] Centralize realistic domain strategies and constrained solver strategies
  (PBT-07).
- [x] Keep Hypothesis shrinking enabled and use/log seed `20260715` in CI
  (PBT-08).
- [x] Retain Hypothesis as the documented Python PBT framework (PBT-09).
- [x] Convert each shrunk PBT defect into a permanent example regression; no
  shrunk defect was produced by this unit's seeded runs.
- [x] Measure statement and branch coverage without making implementation-line
  tests the means of satisfying coverage.
- [x] Update the checklist, state, and audit immediately when Step 9 completes.

### Step 10 - Documentation, Packaging, and Release Cleanup

- [x] Update architecture documentation with layer responsibilities and allowed
  dependency directions.
- [x] Document `OpenPinch.main.pinch_analysis_service` as the sole current
  external Python contract.
- [x] Remove documentation for root aliases, deep imports, compatibility
  facades, and internal runtime records.
- [x] Update examples and notebooks to use the main contract when demonstrating
  supported package use.
- [x] Keep advanced internal examples explicitly labelled unsupported if they
  remain necessary for development.
- [x] Update release notes with the `0.5.0` clean-break scope and no-migration
  policy.
- [x] Update package discovery and distribution manifests for the new owners.
- [x] Build wheel and sdist in isolation.
- [x] Inspect wheel and sdist contents for retired packages and missing owners.
- [x] Install the wheel in a clean environment and run the external contract
  suite.
- [x] Update AI-DLC application design, requirements, implementation summary,
  build/test evidence, audit, state, and checkboxes in the same interaction as
  each completed item.
- [x] Update the checklist, state, and audit immediately when Step 10 completes.

## Blocking Quality Gates

### Quality Review

- [x] Score Ease of Change at least 9/10 with evidence that a new service can
  reuse optimisation without importing heat-pump code.
- [x] Score Simplicity at least 8/10 with evidence that extracted modules have a
  cohesive owner and no speculative protocols or pass-through layers.
- [x] Score Behavioural Tests at least 9/10 with evidence that the main contract
  and domain outcomes dominate over helper-call assertions.
- [x] Score Clear Boundaries at least 9/10 with passing dependency-direction
  checks.
- [x] Score Low Coupling at least 9/10 with explicit inputs, isolated optional
  dependencies, and no global registries.
- [x] Score Project Coherence at least 9/10 with consistent owner, function,
  file, error, and test naming.
- [x] Record an overall quality score of at least 8.8/10 and explain every
  deduction.
- [x] Review `PinchProblem` specifically for composition-hub overload; extract
  only responsibilities with independent reasons to change.
- [x] Review each package for over-splitting; merge modules that are only
  pass-throughs with no independent ownership or test boundary.

### Test and Build Gates

- [x] Main external contract suite passes after every unit.
- [x] All focused unit and integration tests pass.
- [x] Complete non-solver suite passes with no unexpected warnings.
- [x] Statement coverage is at least 95 percent and does not regress from the
  recorded baseline.
- [x] Branch coverage does not regress from the recorded baseline.
- [x] Partial PBT rules PBT-02, PBT-03, PBT-07, PBT-08, and PBT-09 are compliant.
- [x] All available supported solver tests pass; unavailable solvers are
  reported explicitly rather than treated as successful coverage.
- [x] Canonical HPR and HEN numerical fixtures match their approved tolerances.
- [x] HEN tier 0/1 outcomes preserve success, timeout, and skip classifications.
- [x] Ruff lint passes.
- [x] Ruff format check passes.
- [x] Warning-free Sphinx documentation build passes.
- [x] Every packaged notebook parses; executable notebook gates run where their
  dependencies are available.
- [x] Wheel and sdist build in isolation and contain the intended package tree.
- [x] Clean wheel installation passes the `OpenPinch.main` contract suite.
- [x] Cold-import and optional-dependency tests pass.
- [x] Retired-path, forbidden-name, reverse-import, and compatibility-facade
  searches return no unintended matches.
- [x] `git diff --check` passes.
- [x] No unrelated user changes are modified or removed.
- [x] Record a Test Gates score of at least 9/10 and explain every deduction.

## Completion Definition

- [x] All implementation steps and immediate checkbox updates are complete.
- [x] Only `OpenPinch.main.pinch_analysis_service` is treated as an external
  compatibility contract.
- [x] The new package tree has no forwarding compatibility facades.
- [x] Domain, contracts, shared optimisation, application, analysis, adapters,
  and presentation obey their documented dependency boundaries.
- [x] Shared optimisation is demonstrated by a non-HPR test consumer and the HPR
  adapter.
- [x] Main-contract behaviour and approved numerical baselines are unchanged.
- [x] Quality and Test Gates meet their minimum scores.
- [x] Build, test, documentation, notebook, packaging, stale-path, and patch
  evidence is recorded.
- [x] Release notes and all AI-DLC artifacts are current.
- [x] Present generated code for explicit user review before closing Code
  Generation.

## Post-Review Correction - HEN Result Source Tracking

- [x] Reproduce the clean-checkout import failure reported during review.
- [x] Confirm the HEN `results` package exists locally but is hidden by the
  repository-wide `results/` ignore rule.
- [x] Add a narrow `.gitignore` exception for the concrete Python source owner.
- [x] Make `assembly.py`, `selection.py`, `seeds.py`, and `__init__.py` visible
  to version-control discovery.
- [x] Keep this approved architecture checklist visible despite the separate
  generated-solver `plans/` ignore rule.
- [x] Add a repository gate rejecting any Git-ignored Python package source.
- [x] Require HEN result assembly in wheel and sdist content checks.
- [x] Import context, OpenHENS, PDM, TDM, and network-grid result dependants.
- [x] Run a cache-independent Sphinx build with `-E -W`.
- [x] Run focused HEN, presentation, architecture, and packaging tests.
- [x] Run repository Ruff lint/format and `git diff --check`.
- [x] Reproduce the reported five Ruff `I001` failures in a clean Git-index
  snapshot that omits the ignored result package.
- [x] Prove the same snapshot passes Ruff after adding only the result package
  and corrected ignore policy, without suppressions or import workarounds.
- [x] Update implementation, Build and Test, state, audit, and plan evidence in
  the same interaction.

## PBT Compliance for This Planning Artifact

- **PBT-02 Round trips**: compliant in plan; contract and serialization
  round-trip properties are explicit implementation requirements.
- **PBT-03 Invariants**: compliant in plan; domain transactions, ordering,
  conservation, and optimisation result invariants are explicit.
- **PBT-07 Generator quality**: compliant in plan; reusable domain-specific and
  constrained solver strategies are required.
- **PBT-08 Reproducibility**: compliant in plan; shrinking remains enabled and
  seed `20260715` is required.
- **PBT-09 Framework**: compliant; Hypothesis remains the selected Python
  framework.
- **Security Baseline**: N/A because the extension is disabled.
- **Resiliency Baseline**: N/A because the extension is disabled.
