# Package-Wide Owner-Oriented Reorganization Code Generation Plan

## Part 1 - Approved Planning

- [x] Preserve the completed `OpenPinch.classes` refactor as the baseline.
- [x] Inventory thin helper forwarding, schema/barrel inversion, service-owned
  records, optional-dependency imports, and large HEN solver responsibilities.
- [x] Confirm the curated compatibility policy and explicitly removed names.
- [x] Confirm four dependency-ordered implementation units.
- [x] Obtain explicit approval through the user's implementation request.

## Part 2 - Generation

### Unit 1 - Complete Existing Class Extractions

- [x] Move semantic checks into `input/semantics.py`, retaining schema/report
  formatting and assembly in `input/validation.py`.
- [x] Move the complete ProblemTable interval insertion engine into its owner
  helper while keeping public parent delegates.
- [x] Move Stream segment transactions and attachment into owner helpers while
  retaining parent normalization and mutation APIs.
- [x] Move Value coercion and unit normalization/serialization into owner
  helpers while preserving parent arithmetic and state.
- [x] Split workspace views into input, graph, problem-table, and variant
  comparison responsibilities.
- [x] Add/update deterministic and seeded generated tests for Unit 1.
- [x] Run focused Unit 1 tests and Ruff checks.

### Unit 2 - Schemas and Package Barrels

- [x] Move synthesis definitions into concrete common, topology, method, task,
  and result modules without reverse imports from the barrel.
- [x] Preserve public synthesis imports, schemas, dumps, names, and old public
  barrel pickle paths.
- [x] Convert `classes`, `lib`, and `lib.schemas` to typed lazy barrels with
  unchanged `__all__`; keep the root package eager.
- [x] Add structural, API, pickle, and fresh-process cold-import tests.
- [x] Run focused Unit 2 tests and Ruff checks.

### Unit 3 - Service-Owned Helpers and Runtime Records

- [x] Make Process MVR own private membership/stream records and split
  selection, replacement construction, accounting, and state.
- [x] Separate direct-MVR public models, thermodynamics, unit conversion, and
  solver execution behind existing public facades.
- [x] Move multiperiod HPR preparation and shared-design execution into
  `_multiperiod`, privatizing period cases.
- [x] Split graph data and energy-transfer processing by construction,
  metadata, selection/transformation, diagram, and serialization ownership.
- [x] Split Streamlit graphing and network-grid rendering by dependency,
  conversion/adapters, layout, labels, temperatures, dashboard, and exports.
- [x] Remove the approved service/runtime record aliases and documentation.
- [x] Add ownership, copy/pickle, ordering, optional-dependency, API, and seeded
  generated tests for Unit 3.
- [x] Run focused Unit 3 tests and Ruff checks.

### Unit 4 - HEN Equation and Solver Internals

- [x] Extract base-model parameter, piecewise-equation, approach-constraint,
  and solver-execution composition helpers.
- [x] Split stagewise setup, equations, warm starts, evolution, objectives,
  post-processing, and verification; move evolution records to `_stagewise`.
- [x] Split pinch-design preprocessing, equations, amalgamation, and
  post-processing into `_pinch_design`.
- [x] Split solver extraction into recovery, utility, period-state,
  segment-area, and metadata modules.
- [x] Remove approved unit-model internal names from public barrels/docs while
  preserving documented model classes and numerical contracts.
- [x] Add structural, equation-order, extraction, warm-start, API, and solver
  regression tests for Unit 4.
- [x] Run focused Unit 4 tests, available solver tests, and Ruff checks.

### Package-Wide Build and Test

- [x] Run the complete seeded non-solver suite with coverage.
- [x] Run all available solver tests.
- [x] Run Ruff lint and format checks.
- [x] Build warning-free Sphinx documentation and parse notebooks.
- [x] Build wheel and sdist in isolation and inspect public package contents.
- [x] Run stale-path, reverse-import, forbidden-name, public-API, cold-import,
  and `git diff --check` checks.
- [x] Update release notes, implementation summary, Build and Test evidence,
  audit, state, and all remaining checkboxes.
