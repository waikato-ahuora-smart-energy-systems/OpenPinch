# Unit 3 HPR Target API Integration and Publication Code Generation Plan

This plan is the single source of truth for Unit 3 Code Generation. It applies
test-first, regression-preserving changes to the existing brownfield package.
No production implementation step may begin until the complete plan is
explicitly approved.

## Unit Context

- **Unit**: Current HPR Target API Integration and Publication.
- **Primary requirements**: FR-5, FR-8, FR-9, and FR-10; final public and
  packaging verification for FR-1 through FR-10 and acceptance criteria 1
  through 13.
- **User-selected semantic**: explicit `simulation_backend="tespy"` replaces
  thermodynamic cycle evaluations inside ordinary supported HPR targeting and
  determines returned target values. CoolProp remains the omitted-selector
  default and exact compatibility oracle.
- **Topology**: initial TESPy targeting and map publication accept one
  evaporator, one condenser, one refrigerant loop, and scalar period execution.
- **Working fluids**: pure fluids, registered blends, and explicit
  N-component molar mixtures are eligible according to installed property
  capability; REFPROP remains rejected.
- **Execution**: sequential and call-isolated now. No scheduler, shared cache,
  global lock, or thread-safety promise is added. Immutable boundaries remain
  suitable for future separately approved process parallelism.
- **Cache**: one exact-key call-local 512-entry LRU for TESPy candidates; no
  approximate or cross-call reuse.
- **External boundary**: OpenUtility consumes only Unit 1 plain schema/fixture
  data. OpenPinch imports neither OpenUtility, Pyomo, nor HiGHS.
- **Infrastructure, database, repository, frontend, and deployment work**: N/A.

## Dependencies and Stable Interfaces

### Completed upstream units

- Unit 1 supplies `HprPerformanceMapRequest`, `HprPerformanceMap`, strict schema
  `1.0`, and canonical heat-pump/refrigeration fixtures.
- Unit 2 supplies `HprTargetMapBasis`, fluid normalization, CoolProp/TESPy point
  simulators, atomic map generation, diagnostics, characteristics, optional
  extras, and blocking base/TESPy source/artifact profiles.

### Existing brownfield owners to modify

- `OpenPinch/application/_problem/accessors/target.py`
- `OpenPinch/application/workspace.py` only if explicit wrapper behavior needs
  a non-generic implementation; existing `**kwargs` forwarding is preferred.
- `OpenPinch/contracts/hpr.py`
- `OpenPinch/contracts/__init__.py` only for specialist-contract ownership if
  consistent with existing policy; no package-root export.
- `OpenPinch/domain/targets.py`
- `OpenPinch/application/_problem/periods/aggregation.py` only if the additive
  backend field requires explicit scalar aggregation behavior.
- `OpenPinch/analysis/heat_pumps/common/preprocessing.py`
- `OpenPinch/analysis/heat_pumps/optimisation_adapter.py`
- `OpenPinch/analysis/heat_pumps/targeting/cascade_vapour_compression.py`
- `OpenPinch/analysis/heat_pumps/service.py`
- `OpenPinch/analysis/heat_pumps/performance_maps/__init__.py`
- `OpenPinch/analysis/heat_pumps/performance_maps/adapters/tespy.py`
- relevant existing tests, docs, workflows, packaging metadata checks, and
  build scripts.

### Planned focused new internal modules

- `OpenPinch/analysis/heat_pumps/performance_maps/targeting_models.py` for frozen
  candidate requests, thermal profile points, normalized results, evaluator
  metadata, and stable failure values.
- `OpenPinch/analysis/heat_pumps/performance_maps/targeting.py` for selector
  normalization, compatibility preflight, evaluator protocol/factory boundary,
  exact LRU coordination, independent validation, winning-record construction,
  and pure target-to-basis extraction. Split this file only if focused review
  proves an existing owner or a smaller cohesive module boundary is clearer.

TESPy imports remain confined to the existing concrete
`performance_maps/adapters/tespy.py` leaf. No duplicate or suffixed brownfield
file is permitted.

## Expected Interfaces

- The two current vapour-compression target methods add keyword-only
  `simulation_backend: str = "coolprop"`.
- `_TargetAccessor.hpr_performance_map(*, target, request)` returns one detached
  `HprPerformanceMap`.
- `HeatPumpTargetInputs` and `HeatPumpTargetOutputs` carry normalized backend
  intent; output optionally carries one strict winning simulation record.
- `HeatPumpTargetBase` adds `hpr_simulation_backend="coolprop"`.
- The engine-neutral targeting evaluator accepts one frozen candidate request
  and returns one frozen normalized success or candidate-local failure.
- The target-to-basis builder returns the existing Unit 2
  `HprTargetMapBasis`; the existing generator remains the sole map-generation
  service.

## Story and Requirement Traceability

User Stories were intentionally skipped. Requirement ownership is:

| Concern | Primary planned steps |
|---|---|
| FR-5 current-method backend selection | Steps 1 through 4, 8, 11 |
| FR-6 TESPy physical generation integration | Steps 5 through 8, 12, 13 |
| FR-7 failure policy | Steps 5 through 8, 12 |
| FR-8 target/map bridge | Steps 9 and 10 |
| FR-9 downstream independence/publication | Steps 10, 14, 15 |
| FR-10 optional isolation | Steps 3, 6, 15, 17 |
| Acceptance criteria 1 through 13 | Steps 10 through 17 |
| NFR-U3-001 through NFR-U3-033 | Steps 1 through 18 |
| PBT-01 through PBT-10 | Steps 2, 5, 7, 9, 11 through 13, 16 |

## RED-GREEN-REFACTOR Generation Sequence

### Step 1: Freeze Baselines and Public RED Contracts

- [x] Record the focused pre-change target, HPR, architecture, API inventory,
  docs, packaging, and optional TESPy baseline counts.
- [x] Add failing public signature tests for the two backend selectors and
  explicit `hpr_performance_map` method.
- [x] Add omitted-versus-explicit CoolProp oracle examples that pin target type,
  success, numerical fields, streams, and thermodynamic call counts.
- [x] Add failing invalid-selector, no-hidden-map-work, no-root-export, and
  no-CLI tests.
- [x] Run the focused RED selection and record only expected missing-surface
  failures.
- [x] Mark Step 1 complete immediately after evidence is recorded.

### Step 2: Define Targeting Records and Property Strategies

- [x] Add failing contract tests for a frozen, extra-forbid,
  JSON-compatible `HprTargetSimulationRecord` with all nominal single-stage
  fields and structured assumptions.
- [x] Extend reusable HPR strategies for valid/invalid records, target values,
  backend strings, profiles, pure fluids, registered blends, and explicit
  N-component molar mixtures.
- [x] Add JSON round-trip, extra-field rejection, finite/range, composition, and
  immutable-detachment properties.
- [x] Implement the smallest additive record and normalized backend fields in
  `contracts/hpr.py` needed to make the record tests green.
- [x] Preserve existing manually constructed output fixtures through explicit
  CoolProp defaults without weakening new-record validation.
- [x] Run focused example/PBT tests with seed `20260715`; refactor while green.
- [x] Mark Step 2 complete immediately.

### Step 3: Plumb and Isolate Backend Intent

- [x] Add failing accessor, preprocessing, replay, cold-import, and forbidden
  dependency tests for normalized backend intent.
- [x] Implement one closed selector normalizer and pass the value independently
  of HPR cycle and black-box optimizer configuration.
- [x] Carry backend through `HeatPumpTargetInputs`, normalized output, service
  summary, domain target, target-run replay, selected-period calls, and existing
  generic workspace forwarding.
- [x] Prove invalid values fail before `_hpr`, optimizer, CoolProp, or TESPy.
- [x] Prove omitted and explicit CoolProp converge to the unchanged existing
  objective path with no TESPy import and no extra thermodynamic call.
- [x] Run focused tests, Ruff, formatting, and patch hygiene.
- [x] Mark Step 3 complete immediately.

### Step 4: Implement TESPy Compatibility Preflight

- [x] Add failing examples for one-stage acceptance and multi-loop, parallel,
  integrated-expander, analytic-cycle, shared-vector multiperiod, REFPROP,
  missing dependency, unsupported state, and malformed mixture rejection.
- [x] Add generated invariant properties over effective configuration and fluid
  categories.
- [x] Implement a pure preflight returning a frozen prepared single-stage
  targeting specification without constructing a TESPy network.
- [x] Reuse Unit 2 working-fluid resolution, composition preservation, and
  dew/bubble semantics; add no refrigerant allowlist.
- [x] Ensure selected scalar periods and independent all-period replays remain
  eligible while shared-vector TESPy optimization fails early.
- [x] Run focused tests and mark Step 4 complete immediately.

### Step 5: Define Engine-Neutral Target Evaluator RED Contracts

- [x] Add failing examples for candidate request/result/profile validation,
  success physics, corrupt results, local/fatal failures, and sanitized details.
- [x] Add generated properties for finite signs, useful duty, COP, energy
  closure, profile duty/order, mixture identity, native float preservation, and
  exact structural metadata.
- [x] Add a deterministic fake evaluator supporting ordered success,
  candidate-local failure, fatal failure, restoration failure, and cleanup
  failure.
- [x] Add a stateful Hypothesis model for created, ready, evaluating, fatal,
  closing, closed, and cleanup-failed states.
- [x] Run RED and record only the expected absent-model/coordinator failures.
- [x] Mark Step 5 complete immediately.

### Step 6: Implement Evaluator Models, Validation, and Lifecycle

- [x] Create frozen slotted candidate request, profile point, normalized result,
  metadata, and failure values in the planned engine-neutral module.
- [x] Implement independent quantity, COP, energy, profile, backend/model, and
  structured-detail validation using separately named tolerances.
- [x] Implement the explicit evaluator protocol/factory/session lifecycle and
  local/fatal failure partition without retry or fallback.
- [x] Guarantee exactly-once cleanup and no evaluation after fatal/closed state.
- [x] Run the Step 5 examples/state model and close all gaps without importing
  TESPy above its concrete leaf.
- [x] Run Ruff, formatting, coverage checkpoint, and patch hygiene.
- [x] Mark Step 6 complete immediately.

### Step 7: Implement Exact Bounded Candidate Caching

- [x] Add RED examples/properties for exact hits, physical-field key coverage,
  local-failure caching, fatal non-caching, LRU movement, 513-key eviction,
  evicted duplicate re-evaluation, per-call emptiness, and order independence.
- [x] Implement one call-local 512-entry `OrderedDict` LRU with callback, hit,
  miss, solve, insertion, and eviction counters.
- [x] Prohibit float quantization, tolerance buckets, object identity, temporary
  paths, diagnostic ordinals, persistent state, and cross-call reuse in keys.
- [x] Add generated call-count and lifecycle properties with shrinking and seed
  replay.
- [x] Run focused examples/PBT and mark Step 7 complete immediately.

### Step 8: Add TESPy Targeting Design Evaluation

- [x] Add failing real and fake integration tests for one genuine nominal TESPy
  design evaluation in heat-pump and refrigeration mode.
- [x] Cover pure R134a, registered R407C, explicit binary, and explicit ternary
  molar mixture cases without a component-count policy.
- [x] Extend only the existing concrete TESPy leaf with targeting candidate
  design-solve behavior, reusing the Unit 2 topology, characteristic, settings,
  fluid translation, result extraction, restoration, and cleanup helpers.
- [x] Ensure every cache miss is a fresh candidate design state, never an
  offdesign solve relative to a prior optimizer candidate.
- [x] Extract detached source/sink temperature-enthalpy profiles, nonnegative
  duties, compressor-only power, engine version, and stable model assumptions.
- [x] Prove candidate order independence, local failure recovery, fatal
  restoration/cleanup behavior, no CoolProp-cycle fallback, and no TESPy object
  leakage.
- [x] Run the marked TESPy slice, Ruff, formatting, and patch hygiene.
- [x] Mark Step 8 complete immediately.

### Step 9: Integrate TESPy into Single-Stage HPR Optimization

- [x] Add failing objective/service tests that inject the fake evaluator and
  prove TESPy values determine work, COP, streams, accounting, objective,
  candidate ranking, and final target.
- [x] Add all-local-failure, fatal-session, optimizer-error, and cleanup-error
  examples.
- [x] Modify the single-stage cascade vapour-compression objective path to
  receive one call-owned evaluator coordinator while leaving multi-stage and
  all CoolProp paths unchanged.
- [x] Translate normalized thermal profiles into ordinary HPR streams and route
  them through the existing `evaluate_vapour_hpr_result` logic.
- [x] Scope evaluator open/cache/close around the complete optimization so all
  callbacks share only the approved call-local exact cache.
- [x] Prove the returned TESPy target numbers and objective arise from TESPy,
  with no CoolProp target-cycle call after selection.
- [x] Run targeted and existing vapour-compression regressions; mark Step 9
  complete immediately.

### Step 10: Build the Winning Target Record and Map Basis

- [x] Add failing CoolProp/TESPy winning-record tests and target-to-basis
  compatibility examples/properties.
- [x] Build the CoolProp record from the existing winning cycle without another
  solve and the TESPy record from the detached winning evaluator result.
- [x] Store matching backend provenance on normalized output and domain target;
  attach no TESPy model object.
- [x] Implement pure record-to-`HprTargetMapBasis` conversion with deep-detached
  provenance and typed codes for failed, wrong-type, missing, aggregate,
  multi-port, nonscalar, and inconsistent targets.
- [x] Prove repeated extraction equality/idempotence, target non-mutation,
  mixture preservation, and no read of current mutable problem configuration or
  private engine model.
- [x] Run focused examples/PBT and mark Step 10 complete immediately.

### Step 11: Expose the Explicit Map Accessor

- [x] Add failing public examples for supported CoolProp and TESPy targets,
  request validation, reference-capacity override, and complete map return.
- [x] Add failure examples for incompatible targets and a spy proving no Unit 2
  simulator is created before compatibility succeeds.
- [x] Implement `_TargetAccessor.hpr_performance_map(*, target, request)` as
  validation, pure basis construction, and exactly one existing Unit 2
  generation call.
- [x] Prove backend is target-owned and cannot be overridden; prove target,
  request, problem, configuration, caches, and workspace state do not change.
- [x] Do not add automatic all-period/batch map mirrors or package-root exports.
- [x] Round-trip generated public maps and compare semantics with both golden
  fixtures.
- [x] Run focused application/contract tests and mark Step 11 complete
  immediately.

### Step 12: Complete Replay, Period, Batch, and Failure Coverage

- [x] Add selected-period and independent all-period TESPy examples with
  canonical order and one record/session per scalar target.
- [x] Add workspace batch propagation and isolated failure examples using the
  existing generic `**kwargs` wrappers; modify wrapper code only if tests prove
  explicit behavior is required.
- [x] Add shared-vector multiperiod TESPy rejection and unchanged CoolProp
  multiperiod regression tests.
- [x] Generate backend/order/non-mutation/failure-isolation properties across
  scalar period, all-period, and case-batch domains.
- [x] Ensure any aggregate target passed to the map bridge fails before engine
  work and directs the caller to choose one scalar target.
- [x] Run application, workspace, and multiperiod HPR slices; mark Step 12
  complete immediately.

### Step 13: Close PBT, Cache, Memory, and Performance Gates

- [x] Implement every applicable PBT-01 through PBT-10 item from Functional and
  NFR Design with reusable strategies and complementary explicit examples.
- [x] Run all Unit 3 properties with seed `20260715` and shrinking enabled.
- [x] Prove linear callback handling, TESPy solves no greater than resident
  unique requests and never greater than callbacks, and zero added CoolProp
  solves.
- [x] Prove less than 64 MiB additional traced Python memory for 512 maximum-size
  fake cached results.
- [x] Prove ten fake calls and at least three guarded real lifecycle calls retain
  no sessions, temporary directories, cache entries, or engine objects.
- [x] Run one marked real public TESPy target plus minimal map workflow through a
  deterministic bounded candidate search within 300 seconds and record
  target/map/total trends.
- [x] Mark Step 13 complete immediately.

### Step 14: Publish Public Documentation and Examples

- [x] Update `docs/guides/heat-pump-workflows.rst` with omitted/explicit
  CoolProp, explicit TESPy targeting, map follow-up, and plain JSON examples.
- [x] Update `docs/fundamentals/heat-pump-and-refrigeration-methods.rst` with
  target-candidate design versus winning-design map offdesign semantics,
  mixtures, dew/bubble anchors, power boundary, and failures.
- [x] Update `docs/reference/api-heat-pump.rst`, capability/support pages, and
  release notes with exact signatures, optional extra, limits, and ownership.
- [x] Document scalar selected-period and independent all-period support,
  shared-vector/multi-port rejection, process guidance for future parallelism,
  fixed capacity, adjacent interpolation, no fallback/partial map, and
  OpenUtility's plain-data-only role.
- [x] Add or extend executable documentation consistency tests; do not add a
  notebook, CLI, or OpenUtility dependency unless an approved requirement
  explicitly changes.
- [x] Build Sphinx with warnings as errors and mark Step 14 complete immediately.

### Step 15: Extend Architecture, Packaging, and CI Gates

- [x] Extend dependency tests so TESPy remains confined to the existing concrete
  leaf and OpenUtility/Pyomo/HiGHS remain forbidden.
- [x] Extend cold-import tests for root, contracts, domain targets, accessors,
  target evaluator interfaces, record/basis builders, and default targeting with
  TESPy blocked.
- [x] Extend API/root/CLI inventory tests for exactly the approved additive
  surface.
- [x] Extend packaging tests for target-record availability, optional extras,
  schema/fixture/characteristic resource parity, and source/wheel public smokes.
- [x] Extend existing pull-request/release workflows so base and TESPy source and
  artifact profiles execute the new focused suites and 300-second smoke.
- [x] Parse workflows and run focused architecture/packaging tests; mark Step 15
  complete immediately.

### Step 16: Run Focused Quality and Regression Gates

- [x] Run the complete Unit 3 example/PBT selection with the fixed seed.
- [x] Measure at least 95 percent combined statement and branch coverage over new
  Unit 3 modules and materially changed targeting integration paths.
- [x] Run all existing heat-pump, refrigeration, fluid, stream, targeting,
  multiperiod, application, workspace, contract, and Unit 1/2 map regressions.
- [x] Run focused and repository Ruff lint/format checks as proportionate to the
  touched patch; distinguish unrelated pre-existing formatting debt.
- [x] Run documentation, architecture, packaging, resource, workflow, syntax,
  and patch-hygiene gates.
- [x] Fix only Unit 3 regressions and mark Step 16 complete immediately.

### Step 17: Build and Verify Source/Wheel Artifacts

- [x] Build fresh sdist and wheel artifacts using repository tooling in a unique
  temporary output directory.
- [x] Inspect archives for exact schema, fixtures, characteristic resource,
  modules, extras metadata, and absence of duplicate/unplanned files.
- [x] Install the wheel with core dependencies into an isolated environment;
  prove TESPy absent, root/default targeting cold, and fixture/plain-data map
  consumption works.
- [x] Install the same wheel with its `tespy` extra into another isolated
  environment; execute explicit TESPy target, winning record, basis, and minimal
  map smoke within the approved profile.
- [x] Verify source/wheel public signatures and behavior match checkout behavior.
- [x] Record artifact paths, versions, resource digests, and smoke evidence;
  mark Step 17 complete immediately.

### Step 18: Summarize and Present Generated Code

- [x] Create
  `aidlc-docs/construction/hpr-target-api-integration-publication/code/code-summary.md`
  with modified/created files, design decisions, TDD evidence, PBT compliance,
  coverage, performance, optional profiles, docs, regressions, distributions,
  limitations, and OpenUtility boundary.
- [x] Validate the summary and all Unit 3 artifacts against content rules.
- [x] Confirm every Step 1 through Step 18 checkbox and associated requirement
  trace is complete, with no duplicate brownfield files or enabled-extension
  finding.
- [x] Update `aidlc-state.md` and append final execution evidence to `audit.md`.
- [x] Present the standardized generated-code review and wait for explicit
  approval before integrated Build and Test.
- [x] Mark Step 18 complete immediately after the completion prompt is logged.

## Quality Gate Summary

- Default CoolProp omitted/explicit behavior is a blocking oracle.
- TESPy affects ordinary target thermodynamics only in the supported
  single-stage scalar boundary.
- Fake evaluators own broad deterministic and property coverage; real TESPy
  owns small blocking integration and public smokes.
- The 512-entry exact cache, 64-MiB traced Python limit, call-count bounds,
  cleanup, and 300-second profile are blocking.
- New/changed Unit 3 code requires at least 95 percent statement and branch
  coverage.
- Both base and TESPy checkout/artifact profiles, warning-strict docs, Ruff,
  architecture, packages, builds, and installed-wheel behavior must pass.

## Plan Approval

- [x] Create and validate this complete plan before implementation.
- [x] Obtain explicit approval of the complete Unit 3 plan before Step 1.
- [x] Obtain explicit approval of generated Unit 3 code before integrated Build
  and Test begins.

## Content Validation

This plan contains no Mermaid diagram, ASCII diagram, embedded JSON/YAML, or
unbalanced executable code fence. Markdown headings, numbered steps, tables,
paths, inline code, blank lines, checkbox syntax, requirement identifiers, and
special characters were checked before creation.
