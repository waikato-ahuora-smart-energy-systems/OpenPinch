# Unit 2 HPR Point Simulators and Map Generation Code Generation Plan

This plan is the single source of truth for Unit 2 Code Generation. Production
code, tests, package metadata, scripts, and CI workflows remain in the workspace
root. Only the implementation summary is written under `aidlc-docs/`.

## Unit Context

- **Unit**: HPR Point Simulators and Map Generation.
- **Project**: brownfield Python library rooted at
  `/Users/timothyw/Github_Local/OpenPinch`.
- **Stories**: user stories were skipped. Unit 2 owns FR-6 and FR-7, owns the
  numerical and optional-dependency portions of FR-2 through FR-4 and FR-10,
  and supports the Unit 3 integration portions of FR-5, FR-8, and FR-9.
- **Acceptance ownership**: primary acceptance criteria 3 through 5, with
  supporting evidence for criteria 2, 8, and 12.
- **Upstream dependency**: completed Unit 1
  `OpenPinch.contracts.hpr_performance_map` request, point, map, schema, and
  fixture contract.
- **Existing engine dependency**: the current
  `VapourCompressionCycle` and CoolProp fluid-state helper are the default
  physical oracle and parser foundation.
- **Optional engine dependency**: TESPy is an explicitly selected leaf and
  never an import of the package initializer, coordinator, Unit 1, or root
  facade.
- **Downstream dependency**: Unit 3 will create `HprTargetMapBasis`, select the
  backend through current HPR methods, and call the Unit 2 service.
- **External boundary**: OpenUtility consumes only Unit 1 plain mappings or
  exported JSON. OpenUtility, Pyomo, and HiGHS remain forbidden Unit 2 imports.
- **Database, repository, network API, frontend, infrastructure, and deployment
  entities**: N/A. This is a synchronous in-process analysis component with a
  packaged static JSON resource.

## Expected Interfaces

The specialist package exposes only the engine-neutral values and operations
needed by Unit 3:

- `HprTargetMapBasis`, `HprWorkingFluidSpec`, `HprMapGenerationContext`,
  `HprOperatingPoint`, `HprPointSimulation`, `HprSimulatorMetadata`, and
  `HprSimulationDiagnostic`;
- `HprMapGenerationError`;
- structural `HprPointSimulator` and injectable factory callable;
- `build_hpr_map_generation_context`;
- `iter_hpr_operating_points`;
- `get_hpr_point_simulator`; and
- `generate_hpr_performance_map`.

The returned success value is the Unit 1 `HprPerformanceMap`. No raw CoolProp
state, TESPy network, temporary path, partial point list, or consumer object
crosses this boundary. No new `OpenPinch` package-root export is added.

## Exact File Scope

### Create

- `OpenPinch/analysis/heat_pumps/performance_maps/__init__.py`
- `OpenPinch/analysis/heat_pumps/performance_maps/models.py`
- `OpenPinch/analysis/heat_pumps/performance_maps/context.py`
- `OpenPinch/analysis/heat_pumps/performance_maps/fluids.py`
- `OpenPinch/analysis/heat_pumps/performance_maps/points.py`
- `OpenPinch/analysis/heat_pumps/performance_maps/protocols.py`
- `OpenPinch/analysis/heat_pumps/performance_maps/factory.py`
- `OpenPinch/analysis/heat_pumps/performance_maps/generation.py`
- `OpenPinch/analysis/heat_pumps/performance_maps/errors.py`
- `OpenPinch/analysis/heat_pumps/performance_maps/provenance.py`
- `OpenPinch/analysis/heat_pumps/performance_maps/settings.py`
- `OpenPinch/analysis/heat_pumps/performance_maps/resources.py`
- `OpenPinch/analysis/heat_pumps/performance_maps/adapters/__init__.py`
- `OpenPinch/analysis/heat_pumps/performance_maps/adapters/coolprop.py`
- `OpenPinch/analysis/heat_pumps/performance_maps/adapters/tespy.py`
- `OpenPinch/data/heat_pumps/__init__.py`
- `OpenPinch/data/heat_pumps/performance_maps/__init__.py`
- `OpenPinch/data/heat_pumps/performance_maps/openpinch-single-stage-compressor-v1.json`
- `tests/analysis/heat_pumps/hpr_map_fakes.py`
- `tests/analysis/heat_pumps/test_hpr_map_generation.py`
- `tests/analysis/heat_pumps/test_hpr_map_generation_properties.py`
- `tests/analysis/heat_pumps/test_hpr_simulator_stateful.py`
- `tests/analysis/heat_pumps/test_hpr_coolprop_simulator.py`
- `tests/analysis/heat_pumps/test_hpr_tespy_simulator.py`
- `tests/strategies/hpr_map_generation.py`
- `aidlc-docs/construction/hpr-point-simulators-map-generation/code/code-summary.md`

### Modify In Place

- `pyproject.toml`
- `uv.lock`
- `pytest.ini`
- `.github/workflows/ci-pull-request.yml`
- `.github/workflows/ci-develop.yml`
- `.github/workflows/ci-publish.yml`
- `scripts/optional_install_smoke.py`
- `scripts/artifact_install_smoke.py`
- `tests/architecture/test_dependency_rules.py`
- `tests/packaging/test_packaging_metadata.py`
- `tests/packaging/test_resources.py`
- `tests/packaging/test_release_artifacts.py`

Existing files are modified in place. No `_new`, `_modified`, duplicate facade,
parallel resource loader, or second public map model is permitted. Unit 3
target accessors and public documentation are not changed early.

## Detailed Generation Sequence

### Step 1: RED Dependency and Profile Contract Tests

- [x] Extend packaging metadata tests to require base `CoolProp>=8`, neutral
  `tespy` and compatible `brayton_cycle` extras with
  `tespy>=0.10.1.post2`, the complete `full` aggregate, and the compatible
  development dependency.
- [x] Extend optional-install and workflow assertions for isolated core and
  explicit TESPy surfaces and a blocking marked real-TESPy profile.
- [x] Assert the package/root and Unit 1 contract remain cold-importable when
  TESPy is blocked.
- [x] Run the focused tests and record RED against current metadata and profile
  definitions.

**Traceability**: NFR-U2-001 through NFR-U2-005, NFR-U2-023,
NFR-U2-026 through NFR-U2-028; FR-10.

### Step 2: Update Dependencies and Prove the CoolProp 8 Baseline

- [x] Change the required runtime dependency to `CoolProp>=8`, add the neutral
  TESPy extra, align `brayton_cycle`, `full`, and development requirements, and
  regenerate `uv.lock` without weakening unrelated constraints.
- [x] Make the smallest packaging-script/profile changes needed to satisfy the
  dependency tests while retaining base installs without TESPy.
- [x] Run the complete existing CoolProp, stream, vapour-compression, MVR, and
  HPR targeting regression boundary before adding new adapter behavior.
- [x] Run Step 1 tests to GREEN and record the installed CoolProp and TESPy
  versions used as the implementation baseline.

### Step 3: RED Internal Values, Fluid, Context, and Point Tests

- [x] Create constrained Unit 2 strategies for target bases, pure fluids,
  registered blends, explicit binary/ternary and bounded N-component molar
  mixtures, requests, simulation results, and diagnostics.
- [x] Create examples for frozen values, capacity/mode conventions, external
  approach translation, unsupported targets, invalid lift, REFPROP rejection,
  and source-input immutability.
- [x] Add properties for deterministic fluid normalization, preserved component
  order/fractions, dew/bubble anchors, canonical Cartesian size/order/identity,
  and load-fraction induction.
- [x] Run the new context/property modules and record RED because the Unit 2
  package is absent.

**Traceability**: BR-U2-001 through BR-U2-023 and BR-U2-061 through
BR-U2-073; NFR-U2-006 through NFR-U2-013 and NFR-U2-016.

### Step 4: Implement Values, Fluid Resolution, Context, and Lazy Points

- [x] Create frozen slotted internal values in `models.py` without duplicating
  Unit 1 Pydantic contracts.
- [x] Implement `fluids.py` using the existing CoolProp construction semantics,
  explicit REFPROP rejection, capability checks, normalized molar composition,
  and TESPy token formatting without importing TESPy.
- [x] Implement the pure context builder in `context.py` with explicit mode,
  target, capacity, approach, tolerance, and positive-lift validation.
- [x] Implement the lazy ordinal-based Cartesian iterator in `points.py`
  without materializing a second full grid.
- [x] Run Step 3 tests to GREEN and refactor only while those tests remain
  green.

### Step 5: RED Coordinator, Diagnostics, Provenance, and Lifecycle Tests

- [x] Create a deterministic fake simulator with observable lifecycle events,
  configurable preparation, point-local, fatal, invalid-output, and cleanup
  failures.
- [x] Add examples for exact selection, one prepare/close, one call per valid
  coordinate, internal-lift diagnostics, no retry/fallback, ordered aggregation,
  session-unavailable fill, cleanup ordering, and no partial map.
- [x] Add corrupt-result examples for signs, finiteness, useful duty, energy
  closure, and derived COP, including values just inside and outside named
  tolerances.
- [x] Add generated properties for complete-map physical invariants, input
  immutability, deterministic provenance, JSON round trips, and arbitrary
  failed-coordinate sets.
- [x] Add the Hypothesis fresh/prepared/point-local-failed/fatal/closed state
  machine with fixed CI seed and normal shrinking.
- [x] Run the focused modules and record RED against the absent coordinator.

**Traceability**: BR-U2-041 through BR-U2-060; NFR-U2-011 through
NFR-U2-021 and NFR-U2-024 through NFR-U2-027; PBT-02, PBT-03, PBT-06 through
PBT-10. PBT-04 remains N/A.

### Step 6: Implement the Engine-Neutral Lifecycle and Map Coordinator

- [x] Create the structural protocol and context-managed closed factory with
  local concrete imports and one fresh session per request.
- [x] Implement stable typed diagnostics, bounded sanitization, point-local and
  session-fatal private failures, cause chaining, and immutable aggregate error.
- [x] Implement orchestration with preparation at most once, lazy traversal,
  independent numerical checks, atomic failure, and close exactly once.
- [x] Implement deterministic provenance assembly and Unit 1 map construction
  only after total success, retaining native finite floats without quantization.
- [x] Run Step 5 examples, properties, and state-machine tests to GREEN.

### Step 7: RED CoolProp Adapter Oracle Tests

- [x] Add direct `VapourCompressionCycle` comparison examples for heat-pump and
  refrigeration nominal points, including pure fluid and registered blend.
- [x] Add explicit binary and ternary mixture examples accepted by the installed
  CoolProp 8 capability boundary, with correct dew/bubble anchors.
- [x] Add generated fixed-temperature properties proving useful duty and power
  scale with load while COP remains invariant within named tolerance.
- [x] Add adapter failure classification and watts-to-kilowatts boundary tests.
- [x] Run the focused CoolProp module and record RED because the adapter is
  absent.

**Traceability**: BR-U2-024 through BR-U2-030 and BR-U2-061 through
BR-U2-071; NFR-U2-007 through NFR-U2-015 and NFR-U2-024; PBT-05.

### Step 8: Implement the Default CoolProp Adapter

- [x] Implement one session adapter that delegates each point to the existing
  `VapourCompressionCycle` and passes all declared cycle assumptions.
- [x] Apply mode-specific useful duty, convert watts to kilowatts once, expose
  no raw state, and translate engine failures to the internal typed boundary.
- [x] Preserve default steady-state part-load physics without cycling loss,
  PLF, minimum-load penalty, or invented empirical modifiers.
- [x] Run Step 7 tests plus existing vapour-compression and HPR regressions to
  GREEN.

### Step 9: RED TESPy Settings, Resource, and Adapter Tests

- [x] Pin exact finite compressor isentropic-efficiency characteristic points,
  strict JSON keys, canonical bytes, identifier, and expected SHA-256 digest in
  resource tests before adding the resource.
- [x] Pin the exact versioned convergence-settings mapping and absence of caller
  tuning for schema `1.0`.
- [x] Add real TESPy preparation/offdesign/restoration/cleanup smokes for heat
  pump and refrigeration, pure fluid, registered blend, explicit binary, and
  explicit ternary mixture cases.
- [x] Add missing dependency, wrapper/state limitation, non-convergence,
  restoration failure, and no-CoolProp-fallback tests.
- [x] Require a genuine unsupported engine combination to fail with typed
  evidence; do not narrow the approved fluid categories to make a smoke pass.
- [x] Run the focused resource/TESPy module and record RED against absent
  settings, resource, and adapter.

**Traceability**: BR-U2-015 through BR-U2-017, BR-U2-031 through BR-U2-040,
BR-U2-067 through BR-U2-073; NFR-U2-003 through NFR-U2-005, NFR-U2-010,
NFR-U2-019 through NFR-U2-024, and NFR-U2-026.

### Step 10: Implement the Optional TESPy Leaf and Owned Resource

- [x] Create the package JSON characteristic and strict immutable loader using
  `importlib.resources`, exact-key validation, finite/ordered point checks, and
  canonical-byte SHA-256 provenance.
- [x] Create the fixed `openpinch-tespy-hpr-convergence-v1` settings value using
  only approved public TESPy solve arguments.
- [x] Implement the single-stage refrigerant-only TESPy network in the concrete
  leaf with lazy dependency import, compressor-only power, and no secondary
  loops or auxiliaries.
- [x] Solve one target-derived design point, save it in a session-private
  temporary directory, restore it before every offdesign point, and clean it on
  all exits.
- [x] Translate pure/blend/explicit molar inputs without an allowlist, inspect
  convergence/finite values explicitly, and return only normalized results.
- [x] Run Step 9 resource and real-engine tests to GREEN. Treat an unresolved
  approved mixture/API incompatibility as blocking rather than silently
  falling back or changing the contract.

### Step 11: Enforce Architecture, Packaging, and Blocking CI Profiles

- [x] Extend architecture tests to permit TESPy only in the concrete Unit 2
  leaf and to forbid Unit 3/application, OpenUtility, Pyomo, and HiGHS imports
  from the Unit 2 package.
- [x] Extend resource and release-artifact tests for the characteristic JSON,
  its exact digest, source/wheel inclusion, and installed access.
- [x] Register the real-TESPy marker, add `tespy` to optional-install matrices,
  and add separate blocking base/no-TESPy and real-TESPy profile commands to
  pull-request, develop, and publish workflows as applicable.
- [x] Extend optional and artifact installed-package smokes for cold default
  imports, explicit TESPy imports, and packaged resource access.
- [x] Run the architecture, packaging, workflow, and smoke-contract tests to
  GREEN.

### Step 12: Validate Properties, Performance, Coverage, and Regressions

- [x] Run all Unit 2 examples and properties with Hypothesis seed `20260715`,
  retaining shrinking and exact replay output.
- [x] Run the deterministic 10,000-point fake generation under 5.0 seconds and
  256 MiB additional traced memory, plus increasing-grid linear call-count
  checks.
- [x] Prove at least 95 percent statement and branch coverage for new Unit 2
  production modules without excluding error paths.
- [x] Run the complete existing thermodynamic regression suite under CoolProp
  8 and the blocking focused real-TESPy profile without elapsed-time assertions.
- [x] Run Ruff lint/format, package/resource checks, forbidden-import checks,
  and `git diff --check` over the complete Unit 2 patch.

### Step 13: Build, Install, and Summarize Unit 2

- [x] Build and validate the source and wheel distributions using existing
  repository tooling.
- [x] Inspect both archives for the exact compressor resource and verify the
  default wheel remains usable without TESPy while the TESPy extra passes its
  explicit smoke.
- [x] Create
  `aidlc-docs/construction/hpr-point-simulators-map-generation/code/code-summary.md`
  with created/modified files, implemented rules, dependency versions, physical
  assumptions, test evidence, PBT compliance, known engine limitations, and
  deferred Unit 3 work.
- [x] Verify no duplicate brownfield files, no unrelated user changes modified,
  no public targeting integration added early, and every Step 1 through Step 13
  checkbox is complete.

## Requirement and Property Coverage

- Steps 3 through 6 implement the engine-neutral context, traversal, lifecycle,
  diagnostics, physical validation, provenance, and all-or-nothing generation
  boundary.
- Steps 7 and 8 preserve CoolProp as the default and use the existing cycle as
  the independent oracle.
- Steps 9 and 10 implement TESPy as the explicitly selected optional simulator,
  including real pure/blend/binary/ternary evidence and deterministic owned
  model assumptions.
- Steps 1, 2, and 11 enforce dependency and import isolation in metadata,
  distribution smokes, and CI.
- Steps 3, 5, 7, 9, and 12 implement PBT-02, PBT-03, PBT-05 through PBT-10.
  PBT-04 is N/A because no repeated mutating operation claims idempotency.
- Security and Resiliency extensions are disabled. Ordinary sanitized errors,
  atomic cleanup, and dependency isolation remain approved Unit 2 requirements.

## Completion and Approval Tracking

- [x] Code Generation Part 1 read the approved Functional Design, all 28 NFRs,
  NFR Design, Unit 1 interfaces, unit dependency/story maps, and brownfield code
  structure.
- [x] Exact production, test, resource, configuration, workflow, and summary
  paths are listed.
- [x] The thirteen-step RED-GREEN-REFACTOR sequence covers all 73 business
  rules, all 28 NFRs, assigned FRs/acceptance criteria, and applicable PBT
  rules.
- [x] Database, repository, frontend, network API, infrastructure, and deployment
  artifact generation are explicitly N/A.
- [x] The plan was validated as Markdown with no Mermaid, ASCII diagram,
  embedded JSON/YAML, or executable code block.
- [x] Obtain explicit approval of this complete plan before Step 1.
- [x] Complete every Step 1 through Step 13 checkbox during generation.
- [x] Obtain explicit approval of generated Unit 2 code before Unit 3 begins.

## Content Validation

This plan contains no Mermaid, ASCII diagram, embedded JSON/YAML document, or
executable code block. Markdown headings, lists, paths, inline identifiers, and
all plan/substep checkboxes were checked for parser-safe syntax before creation.
