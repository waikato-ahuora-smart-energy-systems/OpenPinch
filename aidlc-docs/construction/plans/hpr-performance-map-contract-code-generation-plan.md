# Unit 1 HPR Performance-Map Contract Code Generation Plan

This plan is the single source of truth for Unit 1 Code Generation. Production
code and tests remain in the workspace root; only the implementation summary is
written under `aidlc-docs/`.

## Unit context

- **Unit**: HPR Performance-Map Contract and Golden Fixtures.
- **Project**: brownfield Python library rooted at
  `/Users/timothyw/Github_Local/OpenPinch`.
- **Stories**: user stories were skipped; Unit 1 owns FR-1 through FR-4 and
  FR-9, supports FR-7, and owns acceptance criteria 1, 2, 4, 6, and 9 through
  13.
- **Dependencies**: existing Python 3.14, Pydantic 2, JSON/resource, pytest,
  Hypothesis, Ruff, Hatchling, build, and package-test owners. Development-only
  `jsonschema` is added explicitly if needed.
- **Downstream units**: Unit 2 consumes the request/map contracts and resources;
  Unit 3 exposes them through current HPR targeting.
- **External boundary**: OpenUtility consumes plain JSON fixtures and field
  semantics only. OpenUtility, Pyomo, HiGHS, CoolProp, and TESPy are forbidden
  Unit 1 production imports.
- **Database entities**: none.
- **Network/API layer**: none. The specialist Python contract and plain JSON
  resources are the complete interface.
- **Repository/infrastructure/frontend/deployment layers**: N/A. Package
  resources and distribution verification use existing owners.

## Expected interfaces

- `OpenPinch.contracts.hpr_performance_map.JsonValue`
- `OpenPinch.contracts.hpr_performance_map.HprPerformanceMapRequest`
- `OpenPinch.contracts.hpr_performance_map.HprPerformanceMapUnits`
- `OpenPinch.contracts.hpr_performance_map.HprPerformancePoint`
- `OpenPinch.contracts.hpr_performance_map.HprPerformanceMap`
- narrow `OpenPinch.resources` helpers to list/read the schema and golden
  fixtures through `importlib.resources`
- explicit development generator/check command for canonical schema and fixture
  resources

No new `OpenPinch` package-root export is added.

## Exact file scope

### Create

- `OpenPinch/contracts/hpr_performance_map.py`
- `OpenPinch/data/contracts/__init__.py`
- `OpenPinch/data/contracts/hpr_performance_map/__init__.py`
- `OpenPinch/data/contracts/hpr_performance_map/schema-1.0.json`
- `OpenPinch/data/contracts/hpr_performance_map/heat-pump-1.0.json`
- `OpenPinch/data/contracts/hpr_performance_map/refrigeration-1.0.json`
- `scripts/generate_hpr_performance_map_contract.py`
- `tests/contracts/test_hpr_performance_map.py`
- `tests/contracts/test_hpr_performance_map_properties.py`
- `tests/strategies/hpr_performance_maps.py`
- `aidlc-docs/construction/hpr-performance-map-contract/code/code-summary.md`

### Modify in place

- `OpenPinch/resources.py`
- `pyproject.toml` and `uv.lock` only if direct development declaration of
  `jsonschema` changes resolved metadata
- `tests/architecture/test_dependency_rules.py`
- `tests/packaging/test_resources.py`
- `tests/packaging/test_release_artifacts.py`
- `scripts/artifact_install_smoke.py`
- `docs/api/schemas-and-config.rst`
- `docs/guides/input-formats-and-validation.rst`
- `docs/reference/api-heat-pump.rst`

Existing files are modified in place; no `_new`, `_modified`, or duplicate
contract/resource file is permitted.

## Detailed generation sequence

### Step 1: RED example contract tests

- [x] Create `tests/contracts/test_hpr_performance_map.py` with one valid
  heat-pump and one valid refrigeration example.
- [x] Pin the exact map field set, units, mode-specific capacity/COP semantics,
  immutable behavior, detached JSON mapping, and JSON round trip.
- [x] Add table-driven invalid examples for unknown/extra fields, version,
  units, finiteness/ranges, modes/conventions, energy, capacity, COP,
  provenance, coordinates, curve temperatures, and load ordering.
- [x] Run only the new example test module and record the expected RED failure
  caused by the absent contract module.

**Traceability**: BR-001 through BR-020 and BR-023 through BR-027; FR-1 through
FR-4 and FR-9; acceptance 1 and 12.

### Step 2: RED domain strategies and property tests

- [x] Create reusable constrained strategies in
  `tests/strategies/hpr_performance_maps.py` for canonical requests, recursive
  JSON provenance, points, curves, and complete heat-pump/refrigeration maps.
- [x] Create `tests/contracts/test_hpr_performance_map_properties.py` for JSON
  round trips, request permutation equivalence, detached serialization,
  physical/range/order invariants, and canonical point ordering.
- [x] Retain Hypothesis shrinking and use the repository's fixed seed/replay
  convention.
- [x] Run the property module and record the expected RED failure caused by the
  absent contract module.

**PBT**: PBT-01, PBT-02, PBT-03, PBT-07, PBT-08, PBT-09, and PBT-10. PBT-04
and PBT-06 remain N/A. JSON Schema comparison in Step 6 supplies PBT-05.

### Step 3: Implement the strict contract module

- [x] Create `OpenPinch/contracts/hpr_performance_map.py` with the recursive
  JSON value type, strict frozen base, request, units, point, and map models.
- [x] Implement finite/range/identity/provenance guards and canonical request
  coordinate ordering.
- [x] Implement mode/capacity/COP, energy, useful-duty, point identity,
  coordinate uniqueness, fixed-temperature curve, ascending-load, and global
  canonical-order validation in linear time.
- [x] Preserve exact alpha `1.0` external fields, including separate
  `energy_balance_tolerance` and `temperature_match_tolerance`.
- [x] Run Steps 1 and 2 tests to GREEN and refactor only with those tests green.

### Step 4: RED resource-generation and drift tests

- [x] Extend the example test module with canonical JSON byte-policy, schema,
  heat-pump fixture, refrigeration fixture, resource-size, and regeneration
  drift expectations.
- [x] Pin at least three nonconstant-COP ordered breakpoints in each fixture.
- [x] Run the focused resource tests and record RED due to absent generator and
  resources.

**Traceability**: BR-021, BR-022, BR-025 through BR-027; NFR-U1-006,
NFR-U1-008, NFR-U1-015, NFR-U1-019; acceptance 6 and 11 through 13.

### Step 5: Implement canonical schema and fixture resources

- [x] Create the importable package-resource directories and initializers.
- [x] Create `scripts/generate_hpr_performance_map_contract.py` with explicit
  check and write modes, resolved fixed targets, canonical UTF-8 JSON, and one
  trailing newline.
- [x] Generate `schema-1.0.json`, `heat-pump-1.0.json`, and
  `refrigeration-1.0.json` exclusively from authoritative valid contract
  values.
- [x] Re-run the resource generator in check mode and require byte equality.
- [x] Run focused contract/resource tests to GREEN.

### Step 6: Add independent schema conformance

- [x] Add `jsonschema` as an explicit development dependency without changing
  runtime or optional dependency profiles, and update `uv.lock` consistently.
- [x] Validate the committed schema against its meta-schema and both fixtures as
  plain parsed JSON without OpenPinch model decoding.
- [x] Add structural generated-payload comparison where JSON Schema can serve
  as an independent oracle.
- [x] Run the example/property suite to GREEN.

**PBT**: implements PBT-05 and completes Unit 1 PBT-02 through PBT-10 coverage.

### Step 7: Add package-resource access

- [x] Modify `OpenPinch/resources.py` in place with a closed catalog and
  list/read helpers for the HPR schema and fixtures.
- [x] Extend `tests/packaging/test_resources.py` for source-tree reads, unknown
  names, detached parsed data, exact catalog, and schema/fixture validation.
- [x] Prove resource access uses `importlib.resources` and does not require an
  unpacked checkout.
- [x] Run focused resource tests to GREEN.

### Step 8: Enforce architecture and backward compatibility

- [x] Modify `tests/architecture/test_dependency_rules.py` in place to forbid
  CoolProp, TESPy, runtime HPR target modules, OpenUtility, Pyomo, and HiGHS from
  Unit 1 production owners.
- [x] Add a subprocess cold-import test with TESPy and external optimizer
  imports blocked, while statically forbidding a direct CoolProp import.
- [x] Assert the package-root export inventory is unchanged and existing HPR
  contract/target tests remain green.
- [x] Run focused architecture and existing contract/HPR regressions.

**Traceability**: NFR-U1-003, NFR-U1-005, NFR-U1-010 through NFR-U1-013;
acceptance 2 and 7.

### Step 9: Integrate distribution and installed-wheel checks

- [x] Modify `tests/packaging/test_release_artifacts.py` in place to require all
  three contract resources in wheel and source distribution.
- [x] Modify `scripts/artifact_install_smoke.py` in place to read and validate
  the installed schema and both fixtures without checkout imports.
- [x] Add source/wheel byte-equality and resource-size checks where owned by
  existing packaging tests.
- [x] Run focused packaging tests that do not require a full distribution
  build; defer the complete clean build/install gate to Step 12 and Build and
  Test.

### Step 10: Publish Unit 1 documentation

- [x] Update `docs/api/schemas-and-config.rst` with specialist contract types,
  exact map/point fields, alpha version policy, and validation behavior.
- [x] Update `docs/guides/input-formats-and-validation.rst` with canonical units,
  useful-duty/COP equations, tolerances, and plain JSON resource consumption.
- [x] Update `docs/reference/api-heat-pump.rst` with the physical-map boundary,
  zero-load exclusion, fixed-capacity semantics, and downstream ownership.
- [x] Include explicit statements that Unit 1 performs no simulation and has no
  OpenUtility/Pyomo/HiGHS dependency.
- [x] Run the warning-strict Sphinx build.

### Step 11: Validate correctness and bounded performance

- [x] Run all new example and property tests with seed/replay output retained.
- [x] Run the deterministic 10,000-point validation performance check and
  confirm the 2.0-second ceiling.
- [x] Confirm at least 95 percent branch coverage for the new contract/resource
  helpers without weakening repository thresholds.
- [x] Run focused existing contract, architecture, packaging-resource, and HPR
  regression tests.
- [x] Run Ruff lint/format checks and `git diff --check` on the complete Unit 1
  patch.

### Step 12: Build, install, and summarize Unit 1

- [x] Run the repository distribution build/package validation and dependency
  audit applicable to the current environment.
- [x] Inspect source and wheel archives for all three byte-identical resources.
- [x] Run the isolated installed-wheel smoke including the HPR contract
  resources.
- [x] Create
  `aidlc-docs/construction/hpr-performance-map-contract/code/code-summary.md`
  listing created/modified files, implemented rules, test evidence, PBT
  compliance, limitations, and deferred Unit 2/3 work.
- [x] Verify no duplicate brownfield files, no unrelated user changes modified,
  and no Unit 2/3 behavior implemented early.

## Completion and approval tracking

- [x] Code Generation Part 1 analyzed the unit, reverse-engineered code
  structure, dependencies, interfaces, and brownfield owners.
- [x] Exact production, test, resource, documentation, and summary paths are
  listed.
- [x] The twelve-step RED-GREEN-REFACTOR sequence covers every Unit 1 FR,
  acceptance criterion, NFR, and applicable PBT rule.
- [x] Database, repository, frontend, network API, infrastructure, and deployment
  artifact generation are explicitly N/A.
- [x] Log and obtain explicit approval of this complete plan before Step 1.
- [x] Complete all Step 1 through Step 12 checkboxes during generation.
- [x] Obtain explicit approval of generated Unit 1 code before Unit 2 begins.

## Content validation

This plan contains no Mermaid, ASCII diagram, embedded JSON/YAML document, or
executable code block. Markdown headings, tables, paths, inline code, and every
plan/substep checkbox were validated before file creation.
