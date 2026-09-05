# AI-DLC State Tracking

## Project Information
- **Project Type**: Brownfield
- **Start Date**: 2026-07-12T21:17:32Z
- **Current Stage**: Workflow Complete (RTD and HPR Notebook Follow-up)

## Workspace State
- **Existing Code**: Yes
- **Programming Languages**: Python, reStructuredText, Markdown, JSON, YAML, TOML
- **Build System**: Hatchling with uv dependency and lockfile management
- **Project Structure**: Python library with application, domain, contracts,
  analysis, optimisation, adapters, presentation, CLI, tests, documentation,
  scripts, notebooks, examples, and packaged data
- **Reverse Engineering Needed**: No, refreshed on 2026-08-23
- **Reverse Engineering Artifacts**: Current under aidlc-docs/inception/reverse-engineering/
- **Workspace Root**: /Users/timothyw/Github_Local/OpenPinch

## Code Location Rules
- **Application Code**: Workspace root, never in aidlc-docs/
- **Documentation**: aidlc-docs/ only
- **Structure patterns**: See code-generation.md Critical Rules

## Extension Configuration
| Extension | Enabled | Decided At |
|---|---|---|
| Security Baseline | No | Utility Placement Requirements Analysis |
| Property-Based Testing | Yes | Utility Placement Requirements Analysis |
| Resiliency Baseline | No | Utility Placement Requirements Analysis |

## TESPy HPR Performance-Map Design Progress

- [x] FOLLOW-UP - Workspace Detection: existing brownfield Python library,
  current reverse-engineering artifacts, existing generated tutorial system,
  and completed HPR implementation detected. Requirements Analysis is active
  for the requested RTD update and comprehensive executable notebook.
- [x] FOLLOW-UP - Requirements Analysis: minimal requirements define notebook
  09 as the canonical comprehensive target-to-map tutorial, covering default
  CoolProp, explicit TESPy, mixtures, the winning record, multi-point map,
  plain JSON export, guarded failures, RTD integration, inventory alignment,
  and executable quality gates. Approved.
- [x] FOLLOW-UP - Requirements approved by the user's `Go` response.
- [x] FOLLOW-UP - Workflow Planning: one focused Code Generation unit followed
  by Build and Test; user stories, application/unit/functional/NFR/
  infrastructure design skipped because no product behavior or architecture
  changes. Awaiting execution-plan approval.
- [x] FOLLOW-UP - Workflow plan approved by the user's `go` response.
- [x] FOLLOW-UP - Code Generation: detailed six-step plan approved. Step 1 RED
  tutorial and RTD contracts are complete with four intended missing-content
  failures and a passing live-public-inventory contract. Step 2 expanded the
  canonical notebook 09 generator with the default CoolProp target, winning
  record, map/export sequence, explicit TESPy mixture screen, and comparison
  context. Step 3 regenerated only packaged notebook 09, proved a stable SHA-256
  across a second pass, compiled all five code cells, verified the full required
  source sequence, and retained the `slow-hpr` profile. Step 4 restored the map
  row to notebook-executable coverage and synchronized the HPR guide, notebook
  series, and 197/197 coverage page; all five focused contracts pass. Step 5
  verification passed: 56 tutorial/RTD/resource tests plus 3 expected profile
  skips, 302 focused HPR tests, real clean-directory notebook 09 execution in
  27.716 seconds, the isolated real TESPy public target/map oracle, warning-strict
  Sphinx, repository Ruff, formatting, compilation, idempotence, and patch
  hygiene. Step 6 produced the validated code summary and full requirement
  traceability. Code Generation was approved by the user's `Build & Test`
  response.
- [x] FOLLOW-UP - Build and Test: fresh sdist and wheel built with packaged
  notebook 09; 3060 of 3064 repository tests passed with 4 expected skips;
  clean notebook execution, real TESPy target-to-map smoke, warning-strict
  Sphinx, Ruff, formatting, compilation, idempotence, and patch hygiene passed.
  Approved by the user's `Go` response.
- [x] FOLLOW-UP - Operations: placeholder reviewed. No deployment, publication,
  monitoring, external service, or infrastructure action was requested or
  performed. The RTD and comprehensive HPR notebook follow-up is complete.

- [x] INCEPTION - Workspace Detection confirmed the existing brownfield Python
  library and current reverse-engineering artifacts.
- [x] INCEPTION - Reverse Engineering skipped because current repository-wide
  artifacts already cover HPR, optional TESPy, contracts, and dependency direction.
- [x] INCEPTION - Requirements Analysis documented in
  `aidlc-docs/inception/requirements/tespy-hpr-performance-map-requirements.md`;
  amended so existing CoolProp-backed HPR targeting methods are the tie-in point,
  CoolProp remains the default, and TESPy is explicitly selectable; approved by
  the user.
- [x] INCEPTION - Cross-project review of the OpenUtility Multi-Period HPR
  Optimization Boundary Plan documented in
  `aidlc-docs/inception/requirements/openutility-multiperiod-hpr-boundary-plan-review.md`;
  eight initial amendments were adopted by the revised plan. A follow-up review
  of the completed alpha implementation confirmed the package boundary and found
  remaining contract/formulation issues around adjacent-segment interpolation,
  capacity scaling, the separate electricity overlay, strict schema semantics,
  dimensional tolerances, refrigeration duty basis, and structured provenance.
  OpenUtility's correction pass resolved the contract/formulation findings and
  passed its release gate with 186 tests at 90.24 percent coverage. The separate
  HPR electricity overlay remains an explicitly documented alpha limitation.
  The final consumer field names are reflected in the approved OpenPinch design.
- [x] INCEPTION - User Stories skipped because the internal technical contract
  already has detailed acceptance criteria and no multi-persona workflow.
- [x] INCEPTION - Workflow Planning documented in
  `aidlc-docs/inception/plans/execution-plan.md`; focused Application Design is
  the only remaining authorized stage; approved by the user.
- [x] INCEPTION - Focused Application Design documented in
  `aidlc-docs/inception/application-design/`; exact contracts, public methods,
  services, dependencies, first-release single-port scope, OpenUtility
  preconditions, and optional TESPy isolation are defined and approved by the
  user.
- [x] INCEPTION - Units Generation Part 1 approved and Part 2 artifacts
  generated using the contract-first three-unit decomposition. Unit boundaries,
  dependency acyclicity, full requirement/acceptance traceability, optional
  TESPy isolation, and enabled Property-Based Testing ownership are validated.
  The user explicitly approved the generated units.
- [x] CONSTRUCTION - Unit 1 HPR Performance-Map Contract and Golden Fixtures:
  Functional Design, NFR Requirements, and NFR Design approved. Infrastructure
  Design is N/A. Code Generation Part 1 approved; all twelve Part 2 steps are
  complete and verified. Generated code approved by the user.
- [x] CONSTRUCTION - Unit 2 HPR Point Simulators and Map Generation: Functional
  Design, NFR Requirements, technology-stack decisions, and NFR Design are
  approved. Code Generation Part 1 and its thirteen-step
  RED-GREEN-REFACTOR plan are approved. Step 1 RED dependency/profile
  contracts are complete with four expected failures and a passing cold-import
  boundary. Step 2 upgraded the locked environment to CoolProp 8.0.0 and TESPy
  0.11.2; all dependency contracts and 532 existing thermodynamic/HPR tests
  pass. Step 3 internal-value, fluid, context, and point tests are complete with
  the expected absent-package RED collection result. Step 4 implemented the
  engine-neutral values, fluid resolution, immutable context, and lazy point
  iterator with 21 focused tests passing. Step 5 lifecycle/coordinator tests are
  complete with the expected absent-error/coordinator RED collection result;
  Step 6 engine-neutral coordinator implementation is complete with 36 focused
  example, property, and lifecycle tests passing. Step 7 CoolProp adapter oracle
  tests are complete with the expected absent-adapter RED collection result;
  Step 8 default adapter implementation is complete with 12 oracle tests and
  the complete 407-test heat-pump suite passing. Step 9 TESPy settings,
  resource, and adapter tests are complete with the expected absent-adapter RED
  collection result. Step 10 implemented the optional TESPy leaf, fixed
  convergence settings, strict characteristic resource, single-stage design
  and restored offdesign lifecycle, and typed engine failures; all 20 focused
  tests pass against TESPy 0.11.2. Step 11 enforces the optional-import firewall,
  exact source/wheel resource inclusion, core/TESPy install smokes, and blocking
  TESPy source and artifact jobs; all 186 packaging and architecture tests pass
  with three expected skips. Step 12 passes 125 seeded Unit 2 tests at 99 percent
  statement-and-branch coverage, the 10,000-point fake time/memory budget and
  linear call-count checks, 600 thermodynamic/stream/HPR regressions, and the
  41-test blocking TESPy profile. Step 13 built source and wheel distributions,
  verified byte-identical characteristic assets, passed isolated core/no-TESPy
  and explicit-TESPy wheel smokes, and created the Unit 2 code summary. All
  thirteen generation steps are complete; generated code was approved by the
  user on 2026-09-05.
  The
  approved 18 patterns and
  15
  logical components define the dependency firewall, lazy traversal, isolated
  simulator sessions, typed point-local/fatal failures, deterministic TESPy
  design restoration/settings/resources, exact provenance, blocking base/TESPy
  profiles, PBT lifecycle model, real-engine smokes, and performance gates. The
  design keeps
  CoolProp as the default simulator, defines TESPy as an explicitly selected
  optional refrigerant-only leaf, uses one global target-derived design point,
  reports compressor-only electricity with auxiliaries excluded, and preserves
  existing single-loop support for registered refrigerant blends and explicit
  mole-fraction mixtures of arbitrary component count with dew/bubble
  saturation anchors. There is no Unit 2 refrigerant allowlist; actual support
  is validated against the installed property backend and operating state.
  TESPy accepts the same working-fluid input categories and may not reject a
  fluid solely because it is a blend or N-component mixture. The approved NFR
  decisions require CoolProp 8 or newer, reject REFPROP, add a neutral TESPy
  extra with a tested minimum, and enforce separate blocking base/TESPy CI
  profiles.
- [ ] CONSTRUCTION - Unit 3 Current HPR Target API Integration and Publication:
  Functional Design, NFR Requirements, and technology-stack decisions are
  generated and approved. NFR Design is generated and approved. Infrastructure
  Design is N/A because the unit adds no service, deployment, persistence, or
  cloud component. Code Generation Part 1 produced an 18-step
  RED-GREEN-REFACTOR plan that is approved; Part 2 has begun with Step 1.
  Step 1 recorded a 164-test focused green baseline and the expected eight RED
  failures for the absent selectors, explicit map method, forwarding, and
  closed target manifest. Step 2 added the strict frozen winning-target
  simulation record, additive CoolProp defaults, reusable pure-fluid/blend/
  N-component-mixture and target/profile strategies, and deterministic seeded
  record properties; 107 focused contract and HPR regression tests pass. Step
  3 implemented one closed selector normalizer and carried runtime backend
  intent through the public vapour-compression methods, preprocessing, output,
  domain provenance, all-period forwarding, and recorded-period replay without
  storing it in numerical configuration. The 85-test affected API/HPR slice,
  34 cold-import/dependency tests, Ruff, formatting, and patch hygiene pass.
  Step 4 added a pure fail-fast TESPy compatibility gate and frozen prepared
  specification. It accepts the approved one-by-one scalar vapour-compression
  topology in heat-pump or refrigeration mode, reuses installed-property
  dew/bubble capability checks without a fluid allowlist or mixture-size cap,
  and rejects unsupported cycles, multi-loop/shared-vector forms, integrated
  expanders, REFPROP, unavailable dependencies, and invalid states before the
  optimizer. The complete 546-test HPR/import-firewall regression, Ruff,
  formatting, and patch hygiene pass. Step 5 added engine-neutral request,
  result, profile, metadata, validation, local/fatal failure, cleanup, fake
  evaluator, generated-physics, and stateful lifecycle RED contracts. The RED
  run stopped at the expected absent coordinator/model surface. Step 6
  implemented frozen slotted engine-neutral values, deep-frozen sanitized
  metadata, independent physical/profile validation, the evaluator protocol,
  and an exactly-once lifecycle coordinator with recoverable/fatal partitioning.
  Fifty-nine focused tests pass; the new modules have an 84 percent branch
  coverage checkpoint, and 34 import-boundary tests plus Ruff/format/hygiene
  pass. Step 7 added the exact unquantized call-local 512-entry LRU with
  immutable success/local-failure values; diagnostic identity is excluded from
  otherwise complete physical keys. Hit movement, 513th-key eviction,
  fatal non-caching, cache clearing, counters, call isolation, order
  independence, and seeded generated call counts pass in 72 focused tests;
  import boundaries, Ruff, formatting, and patch hygiene pass. Step 8 is
  active. Step 8 extended only the concrete optional TESPy leaf with a nominal
  targeting evaluator. Every cache miss builds fresh isolated topology and
  performs a design solve; it never uses offdesign state from another
  candidate. Detached compressor-only power, source/sink duties and profiles,
  metadata, mixtures, order independence, local recovery, fatal cleanup, and
  object isolation pass in 79 real TESPy/preflight/legacy-adapter tests plus 78
  engine-neutral/import-boundary tests. Ruff, formatting, and patch hygiene
  pass. Step 9 integrated a call-owned TESPy coordinator into only the approved
  scalar single-stage cascade path, translated detached profiles into ordinary
  condenser/evaporator utility streams, and routed TESPy duties and
  compressor-only power through the existing HPR accounting pipeline. Local,
  fatal, optimizer, cleanup, cache, ranking, no-CoolProp-fallback, and real
  TESPy accounting examples pass. The final focused regression is 195 tests,
  with a separate 24-test fake/real integration slice; Ruff, formatting, and
  patch hygiene pass. Step 10 built CoolProp records directly from already
  solved cycles and TESPy records from detached winning results/metadata,
  propagated scalar period identity without mutating the optimizer result, and
  added the pure typed target-to-map-basis compatibility bridge. Deterministic,
  deep-detachment, private-model isolation, mixture-preservation, and seeded
  generated-record properties pass in a 75-test focused slice with real TESPy,
  Ruff, formatting, and patch hygiene. Step 11 added the explicit keyword-only
  target accessor bridge. It validates one scalar target and the public request,
  constructs a detached basis, and calls the existing Unit 2 map generator once.
  Backend ownership, reference-capacity override, target/request/problem
  non-mutation, no automatic all-period mirror, no root export, public JSON
  round trips, and semantic alignment with both golden fixtures pass in 137
  focused application/contract tests; Ruff, formatting, and patch hygiene pass.
  Step 12 preserved backend and detached winning-record provenance in report
  outputs, proved canonical independent scalar-period order and isolation,
  verified generic workspace batch forwarding and per-case failure isolation,
  retained the explicit shared-vector TESPy rejection before optimizer setup,
  and kept CoolProp multiperiod behavior green. Generated backend/order/state
  properties and aggregate-map early rejection pass in a 71-test application,
  workspace, multiperiod, reporting, and accessor slice; Ruff, formatting, and
  patch hygiene pass. Step 13 completed all applicable PBT-01 through PBT-10
  gates with reusable seeded properties and explicit examples. The fixed-seed
  Unit 3 selection passes 150 tests with shrinking enabled. Structural tests
  prove linear callback handling, exact resident-key solve bounds, no added
  CoolProp evaluation on the TESPy path, 512-entry eviction, and less than 64
  MiB traced Python heap for 512 maximum-size fake records. Ten fake calls and
  three real TESPy calls release caches, sessions, temporary state, and engine
  objects. The marked public pulp-mill TESPy target plus two-point map smoke
  completes in 0.43 seconds against the supported local profile, below the
  300-second guard. Ruff and formatting pass. Step 14 published the CoolProp
  default and explicit TESPy target-to-map workflow, exact optional extra,
  mixture and dew/bubble rules, winning-design/offdesign lifecycle, scalar and
  independent-period limits, atomic fixed-capacity map semantics, adjacent
  interpolation ownership, process-parallel guidance, and the plain-data-only
  OpenUtility boundary across the guide, fundamentals, API, schema, capability,
  support, install/resource, README, and release-note surfaces. Twenty
  documentation consistency tests and the complete 55-source warnings-as-errors
  Sphinx build pass. No notebook, CLI, or downstream dependency was added. Step
  15 extended cold-import coverage through HPR contracts, target records,
  target/basis builders, accessors, and default CoolProp targeting with TESPy
  blocked; reinforced the TESPy leaf and downstream-optimizer dependency
  firewalls; and kept the root/API/CLI inventory closed. Optional and artifact
  smokes now verify the target record, evaluator boundary, default selector, and
  a real installed-package public TESPy target-to-map workflow. Develop, pull
  request, and publish jobs retain the complete TESPy suite and add a dedicated
  300-second public smoke guard. The workflows parse, the optional TESPy smoke
  and public helper pass, and 87 architecture, packaging, resource, and API tests
  pass. Step 16 ran the complete fixed-seed Unit 3 selection, the broad
  non-solver regression profile, and focused closure tests for the four Unit 3
  compatibility findings. The affected 871-test heat-pump, application,
  contract, architecture, packaging, and resource selection passes. Combined
  statement/branch coverage is 97 percent across the new Unit 3 modules and
  materially changed targeting paths, exceeding the 95-percent gate. Ruff,
  changed-surface formatting, compilation, workflow parsing, patch hygiene,
  optional TESPy smoke, and the warnings-as-errors 55-source Sphinx build pass;
  the repository-wide format check identifies 14 unrelated pre-existing files.
  Step 17 built fresh OpenPinch 0.6.4 source and wheel artifacts under the unique
  `/private/tmp/openpinch-step17.cUgtgY/artifacts` directory. Archive inspection
  found no duplicate members and proved byte-exact schema, fixture, and TESPy
  characteristic resources. The isolated 18-package core wheel profile excludes
  TESPy and passes in 3.11 seconds; the isolated 33-package TESPy 0.11.2 profile
  completes the public target, winning-record, basis, and two-point map workflow
  in 3.71 seconds. Checkout, core wheel, and TESPy wheel public signatures and
  resource digests match. Step 18 created and validated the complete
  generated-code summary, confirmed Steps 1 through 18 and FR-5 through
  FR-10/NFR-U3-001 through NFR-U3-033 traceability, found no duplicate
  brownfield files or enabled extension finding, and logged the standardized
  review prompt. Unit 3 Code Generation awaits explicit approval before
  integrated Build and Test. The user approved Unit 3 Code Generation with
  `go`. Integrated Build and Test completed with 3,059 passes, 4 expected
  optional-profile skips, and zero failures in the final 458.40-second
  solver-enabled run. The dedicated public TESPy target-to-map smoke passes in
  1.99 seconds under its 300-second guard. Ruff, changed-surface formatting,
  Python compilation, patch hygiene, warning-strict Sphinx, coverage,
  performance/lifecycle, source/wheel archives, and isolated core/TESPy wheel
  profiles pass. The tutorial inventory now accurately distinguishes 196
  notebook-demonstrated operations from the specialist target-owned map API,
  which is documented and executable through routine fake and guarded TESPy
  tests. The user approved Build and Test. The Operations phase is the current
  workflow placeholder and performs no deployment, publication, monitoring, or
  external mutation. The TESPy HPR performance-map workflow is complete.
  The user's answer B establishes full targeting semantics: explicit TESPy must
  replace the thermodynamic cycle evaluations inside ordinary single-stage HPR
  placement optimization and determine the returned target values. CoolProp
  remains the omitted-selector default with unchanged numerical behavior.
  TESPy targeting is initially restricted to one evaporator, one condenser, one
  refrigerant loop, and scalar period evaluation; pure fluids, registered
  blends, and explicit N-component molar mixtures remain eligible according to
  installed property support. Each optimizer candidate receives an independent
  TESPy design solve, while the winning detached simulation record supplies the
  later Unit 2 offdesign map basis. Shared-vector multi-period TESPy targeting,
  multi-port flattening, automatic grids, OpenUtility imports, Pyomo, and HiGHS
  remain excluded. PBT-01 is compliant with identified oracle, invariant,
  round-trip, idempotence, and easy-verification properties.
  The NFR design basis uses an exact-key call-local LRU cache capped at 512
  immutable entries, at most one TESPy design solve per resident unique
  candidate, less than 64 MiB traced Python cache overhead, isolated evaluator
  state per call, and one marked public TESPy target-and-map smoke within 300
  seconds on the supported CI profile. CoolProp performs no extra cycle solve,
  TESPy imports remain lazy, and base/TESPy source and artifact profiles remain
  blocking. PBT-09 is compliant through the existing Hypothesis/pytest stack.
  The NFR Design defines 19 patterns and 16 logical components. Current
  execution stays sequential and call-isolated with no scheduler, shared cache,
  global lock, or thread-safety promise. Frozen candidate inputs/results and
  call-owned factories remain process-ready for a future separately approved
  parallel optimizer. Recoverable candidate failures are not retried; fatal
  lifecycle/configuration failures abort. Every NFR-U3-001 through
  NFR-U3-033 and applicable PBT-01 through PBT-10 rule has an explicit pattern,
  component, and verification owner.

## Utility Placement Optimisation Progress

- [x] GITHUB CODEX HEAT-RECOVERY REVIEW CORRECTIONS - Preserved the requested
  recovery as the near-limit inverse feasibility target and resolved valid
  configured temperature aliases through Pint to delta-temperature output
  units. The focused gate passes 142 tests; Ruff, formatting, and patch hygiene
  pass; and the CI-equivalent non-solver suite passes 2,647 tests with 3
  expected skips and 4 solver deselections at 96 percent branch coverage.
  Plan:
  `aidlc-docs/construction/plans/github-codex-review-heat-recovery-corrections-plan.md`.

- [x] PULL-REQUEST AUTOMATIC VERSION-BUMP ORDERING - Restored automatic version
  advancement for same-repository pull requests targeting `main`, then validate
  the updated head version against the base without repeat bumps or fork writes.
  The focused gate passes 123 tests with 3 expected skips; the CI-equivalent
  non-solver suite passes 2,638 tests with 3 expected skips and 4 solver
  deselections at 96 percent branch coverage. Workflow YAML, Ruff, formatting,
  warning-strict RTD, the isolated pinned bump-tool smoke, version-file
  non-mutation, and patch hygiene pass.
  Plans:
  `aidlc-docs/inception/plans/pr-automatic-version-bump-ordering-execution-plan.md`
  and
  `aidlc-docs/construction/plans/pr-automatic-version-bump-ordering-code-generation-plan.md`.

- [ ] GITHUB ACTIONS COMPREHENSIVE HARDENING - Code, tests, documentation, and
  the post-change audit are complete. The focused gate passes 58 tests; YAML,
  Ruff, formatting, documentation, distributions, and patch hygiene pass; and
  the complete configured-solver suite passes 2,534 tests with 4 expected
  skips. Live Actions defaults are hardened, the PyPI environment remains
  protected, stable release tags are protected, and the stale Code Owners
  requirement is removed. The first successful pushed `pr-gate` run remains
  before that check can be required on `main`. Plan:
  `aidlc-docs/construction/plans/github-actions-comprehensive-hardening-code-generation-plan.md`.

- [x] GITHUB ACTIONS REVIEW PROVENANCE AND RETRY CORRECTIONS - Anchor the
  release manifest to the unprivileged build run and separate PyPI availability
  verification from the immutable upload. The focused gate passes 108 tests
  with 3 expected skips; all workflows parse; release checks, documentation,
  0.6.3 distributions, Ruff, formatting, and patch hygiene pass; and the
  complete configured-solver suite passes 2,522 tests with 4 expected skips.
  Plan:
  `aidlc-docs/construction/plans/github-actions-review-provenance-retry-corrections-plan.md`.

- [x] GITHUB ACTIONS TAG-CONTEXT PYPI PUBLICATION - Publish the GitHub Release
  after TestPyPI validation, dispatch the existing trusted workflow at the
  immutable version tag, validate and reuse the published artifacts, retain the
  protected `pypi` environment, and verify production publication. The focused
  packaging gate passes 106 tests with 3 expected skips; all workflows parse;
  release metadata, documentation, distributions, Ruff, formatting, and patch
  hygiene pass; and the complete configured-solver suite passes 2,520 tests
  with 4 expected skips. Plan:
  `aidlc-docs/construction/plans/github-actions-tag-context-pypi-publication-plan.md`.

- [x] RELEASE VERSION 0.6.2 - Advanced all canonical version records and
  satisfied the forward-version gate against main at 0.6.1. The focused gate
  passes 33 tests; the complete configured-solver suite passes 2,519 tests with
  4 expected skips; Ruff, lockfile consistency, fresh 0.6.2 archives, and patch
  hygiene pass. Plan:
  `aidlc-docs/construction/plans/release-version-0.6.2-plan.md`.

- [x] GITHUB ACTIONS COUENNE RUNNER CORRECTION - Pinned the release solver gate
  to Ubuntu 22.04 and added a fail-fast Couenne/IPOPT runtime probe. The focused
  gate passes 25 tests; all workflows parse; the exact probe passes; and the
  complete configured-solver suite passes 2,519 tests with 4 expected skips.
  Ruff, changed-file formatting, and patch hygiene pass. Plan:
  `aidlc-docs/construction/plans/github-actions-couenne-runner-correction-plan.md`.

- [x] GITHUB ACTIONS ARTIFACT SMOKE CORRECTION - Corrected checkout-local virtual
  environment classification, updated artifact actions to their supported Node
  runtime releases, and verified all affected workflow gates. The complete
  configured-solver suite passes 2,518 tests with 4 expected skips; the focused
  gate passes 27 tests; Ruff, YAML parsing, warning-strict RTD documentation,
  fresh distributions, installed-wheel smoke, and patch hygiene pass. Plan:
  `aidlc-docs/construction/plans/github-actions-artifact-smoke-correction-plan.md`.

- [x] GITHUB ACTIONS RELEASE AND CODEX REVIEW CORRECTIONS - Hardened PR CI,
  automated trusted tag/GitHub-release/TestPyPI/PyPI delivery, and resolved the
  three current GitHub Codex review findings. The complete configured-solver
  suite passes 2,515 tests with 4 expected skips; Ruff, YAML parsing, warning-
  strict RTD documentation, OpenPinch 0.6.1 distributions, and patch hygiene
  pass. Plan:
  `aidlc-docs/construction/plans/github-actions-release-and-codex-review-corrections-plan.md`.

- [x] REPOSITORY CRITICAL CORRECTIONS - Correct transactional problem loading,
  utility-profile validation, segmented utility assignment, period-cap
  identity, multiplier persistence, and optimizer constraint feasibility. The
  complete configured-solver suite passes 2,497 tests with 4 expected skips;
  Ruff, Sphinx, package build, and patch hygiene pass.
  Plan: `aidlc-docs/construction/plans/repository-critical-corrections-plan.md`.

- [x] GITHUB CODEX REVIEW CORRECTIONS - Correct period-aware maximum-duty
  replay serialization and make shared utility targeting replace every duty,
  including zeros. The complete no-deselection suite passes 2,480 tests with 4
  expected skips. Plan:
  `aidlc-docs/construction/plans/utility-placement-codex-review-corrections-plan.md`.

- [x] DEFAULT-UTILITY PENALTY COEFFICIENT 1000 - Increase the private
  Utility Placement squared fallback coefficient from 100 to 1000 while
  retaining physical entropy, feasibility, units, and period weighting. The
  complete no-deselection suite passes 2,471 tests with 4 expected skips. Plan:
  `aidlc-docs/construction/plans/utility-placement-default-penalty-1000-plan.md`.

- [x] DEFAULT-UTILITY PENALTY TENFOLD AMENDMENT - Increase the private
  utility-placement squared fallback coefficient from 10 to 100 while retaining
  physical entropy, feasibility, units, and weighted-period semantics. The
  focused gate passes 289 tests and the complete no-deselection suite passes
  2,471 tests with 4 expected skips. Plan:
  `aidlc-docs/construction/plans/utility-placement-default-penalty-tenfold-plan.md`.

- [x] FOUR-ISOTHERMAL NOTEBOOK CONTRACT - Align notebook 19, its canonical
  generator, fixed packaging contract, requirements, and RTD with four
  isothermal and zero sensible levels for both Process and Site examples. The
  complete no-deselection suite passes 2,471 tests with 4 expected skips.
  Plan: `aidlc-docs/construction/plans/utility-placement-four-isothermal-notebook-contract-plan.md`.

- [x] TOTAL SITE CACHED PROCESS-PROFILE AND SUGCC TARGETING - Complete each
  immediate process profile once per period, target candidate endpoints against
  detached copies, accumulate duties, and construct only one candidate SUGCC.
  Five structured candidates reproduce full hierarchy replay and ten
  representative Total Site candidates are 11.0959 times faster. The focused
  gate passes 372 tests and the complete applicable gate passes 2,467 tests
  with 4 expected skips and 4 user-notebook deselections. Plan:
  `aidlc-docs/construction/plans/utility-placement-cached-sugcc-code-generation-plan.md`.

- [x] TOTAL SITE UTILITY-SET PERFORMANCE INVESTIGATION - Profile exact replay
  across distinct utility sets, isolate candidate-specific work, prototype
  semantics-preserving shortcuts, assess process-based candidate parallelism,
  and rank acceleration opportunities without changing application code.
  Follow-up validation confirmed that completed process net-load profiles can
  be cached once per process and period, candidate utility temperatures can be
  inserted by interpolation, and site net duties can then be obtained from the
  aggregate SUGCC plus same-level cancellation. Ten structured candidate sets
  reproduced every named exact-replay duty exactly while reducing comparison
  time by 2.97 times before removing candidate problem-tree construction.

- [x] TOTAL SITE UNIFORM-TEMPERATURE-TOLERANCE CORRECTION - Canonicalise
  tolerance-equivalent source rows consistently while aligning Total Site
  process and utility problem tables, eliminate the 70/71-row broadcast
  failure, and verify the four-isothermal Total Site placement workflow.
- **Uniform-tolerance plan**:
  `aidlc-docs/construction/plans/utility-placement-total-site-uniform-tolerance-code-generation-plan.md`.

- [x] PREPARED TARGET REPLAY PERFORMANCE CORRECTION - Cache process-only direct
  load profiles per zone and period, copy and augment them with candidate
  utility intervals, retain candidate-specific utility and Total Site
  targeting, and prove equivalence to fresh `PinchProblem` replay under TDD.
  Process candidate replay is 8.84 times faster and Total Site replay is 1.65
  times faster at the measured median. The final focused gate passes 328 tests;
  the complete applicable gate passes 2,460 tests with 4 expected skips and 3
  deliberate user-notebook deselections. Ruff, Sphinx, and patch hygiene pass.
- **Prepared replay plan**:
  `aidlc-docs/construction/plans/utility-placement-prepared-target-replay-code-generation-plan.md`.

- [x] SUPPLY-ORDER INTERLEAVING CORRECTION - Order physical utility use by
  optimized supply temperature, permit isothermal and sensible generated
  levels to interleave, retain same-kind ordinal separation, and prove the
  notebook-derived entropy improvement without rewriting the user's notebook.

- [x] CMA-ES DEFAULT AMENDMENT - Changed the utility-placement black-box default
  from dual annealing to CMA-ES while retaining per-call backend selection and
  preserving the user's locally modified notebook 19. Focused and broad gates
  pass, including an executed temporary canonical notebook under CMA-ES.

- [x] UNCAPPED NOTEBOOK AMENDMENT - Removed `maximum_duties` from notebook 19's
  Process example while retaining bounded search options and the optional API.
  All four TDD steps are complete; the uncapped Process result uses no fallback,
  exact retarget replay holds, and all focused and complete gates pass.

- [x] PROFILE NON-CROSSING CORRECTION - Shared ordinary targeting retains the
  tightest interior breakpoint limit, so sensible Utility GCC profiles cannot
  cross the residual Process GCC. All six TDD steps, executable notebook/RTD,
  and the 2,450-pass solver-enabled suite are complete.

- [x] EXACT-TARGET REPLAY CORRECTION - Candidate temperatures are evaluated
  through detached ordinary Process, Site, Community, or Region targeting;
  target-owned duties, totals, fallbacks, and balanced-composite data feed the
  optimizer and reproduce on the returned case. Notebook 19 is executable and
  assertion-free. Ruff, Sphinx, packaging, and 2,448 tests pass with 4 expected
  skips.

- [x] PROFILE-AWARE DISPATCH CORRECTION - Replace temperature-only greedy duty
  selection with period-specific profile-aware duty decisions, align canonical
  ranking with the backend objective, use an entropy-unit scale, correct replay
  accounting, and refresh notebook 19 and RTD under the approved eight-step TDD
  plan. All eight steps are complete. The focused gate passes 220 tests, the
  refreshed notebook/RTD gate passes 41 tests with 3 guarded skips, and the
  complete solver-enabled suite passes 2,445 tests with 4 expected skips.

- [x] REQUIREMENTS AMENDMENT - Define per-utility maximum-duty inputs,
  multiperiod semantics, omitted-limit behavior, and infeasibility handling.
  Name-mapped inputs, period-resolved bounds, residual default fallback, and
  squared `g_penalty()` are approved and documented. Functional design and the
  TDD Code Generation plan are complete under standing authorization.
- [x] MAXIMUM-DUTY CONSTRUCTION - Execute the approved ten-step TDD plan,
  notebook/RTD integration, full Build and Test, review, and commit. Code
  Generation Steps 1 through 10 are complete. The final solver-enabled tree
  passed 2,438 tests with 4 expected skips.
- [x] REQUIREMENTS CORRECTION - Monetary utility placement and its paused
  boiler/turbine amendment are deferred in full. The current feature is
  thermodynamic-only and retains balanced-composite entropy generation,
  all-period weighted summation, and generated-default-utility exclusion.
- [x] CONSTRUCTION - Thermodynamic-only removal, non-cosmetic scope audit, and
  case-based API usability amendment completed; 2,389 tests passed with 4
  expected skips and solver markers enabled.
- [x] API USABILITY AMENDMENT - The concise call uses `isothermal` and
  `sensible`, returns a detached normal optimized case, retains evidence on
  that case, and registers it explicitly through `workspace.add(...)`.
- [x] HIERARCHY API AMENDMENT - Make Process, Site, Community, and Region
  utility ownership explicit and remove the ambiguous public target selector;
  infer typed levels from existing utilities when counts are omitted. TDD,
  notebook/RTD, packaging, installed-wheel smoke, and full Build and Test are
  complete under continuing authorization; 2,404 tests passed with 4 expected
  skips.
- [x] PROFILE-ENVELOPE CORRECTION - The feasible temperature envelope and
  deterministic starts follow residual Process and Site profile support so the
  entropy optimizer can reduce the visible background/utility gap. Notebook 19,
  RTD, properties, package artifacts, installed-wheel execution, and the full
  solver-enabled suite pass; 2,408 tests passed with 4 expected skips.
- [x] COUPLED GENERATED PAIRS - Matching generated hot/cold isothermal and
  sensible entries share one temperature interval with exact reversed
  endpoints and independent duties. Requirements, TDD, notebook/RTD,
  distribution, installed-wheel execution, and complete Build and Test are
  complete; 2,410 tests passed with 4 expected skips.

- [x] INCEPTION - Workspace Detection completed for the existing brownfield Python library.
- [x] INCEPTION - Reverse Engineering refreshed and approved by the user.
- [x] INCEPTION - Comprehensive Requirements Analysis updated and approved by the user.
- [x] INCEPTION - User Stories assessment, approved plan, four personas, and 12 stories generated and approved.
- [x] INCEPTION - Workflow Planning generated and approved by the user.
- [x] INCEPTION - Application Design generated, validated, and approved by the user.
- [x] INCEPTION - Units Generation Part 1 decomposition plan answered and approved by the user.
- [x] INCEPTION - Units Generation Part 2 artifacts generated, validated, and approved by the user.
- [x] CONSTRUCTION - Unit 1 Placement Contracts and Pure Model Functional Design generated, PBT-01 compliant, and approved by the user.
- [x] CONSTRUCTION - Unit 1 NFR Requirements generated, validated, PBT-09 compliant, and approved by the user.
- [x] CONSTRUCTION - Unit 1 NFR Design generated, validated, and approved by the user.
- [x] CONSTRUCTION - Unit 1 Infrastructure Design skipped; no infrastructure boundary exists.
- [x] CONSTRUCTION - Unit 1 Code Generation Part 1 plan generated and approved by the user.
- [x] SCOPE AMENDMENT - CLI integration excluded; Unit 3 delivers exactly one
  executable generated notebook demonstrating the thermodynamic workflow,
  named-case utility replacement, and standard plots, with manifest,
  execution, and package-data verification.
- [x] CONSTRUCTION - Unit 1 Code Generation Part 2 implementation, focused verification, regression, packaging, summary, and generated-code approval complete.
- [x] CONSTRUCTION - Unit 2 Placement Evaluation and Optimisation Service
  Functional Design generated, validated, PBT-01 compliant, and approved.
- [x] CONSTRUCTION - Unit 2 NFR Requirements generated, validated, PBT-09
  compliant, and approved under continuing authorization.
- [x] CONSTRUCTION - Unit 2 NFR Design generated, validated, and approved under
  continuing authorization; Infrastructure Design remains skipped.
- [x] CONSTRUCTION - Unit 2 Code Generation Part 1 plan generated, PBT-01
  through PBT-10 compliant, and approved under continuing authorization.
- [x] CONSTRUCTION - Unit 2 Code Generation Part 2 implementation, properties,
  real optimizer regression, performance, coverage, compatibility,
  distribution, summary, and generated-code approval complete.
- [x] CONSTRUCTION - Unit 3 Functional Design, NFR Requirements, and NFR
  Design generated, validated, and approved under continuing authorization;
  Infrastructure Design skipped because no infrastructure boundary exists.
- [x] CONSTRUCTION - Unit 3 Code Generation Part 1 plan generated and approved
  under continuing authorization.
- [x] CONSTRUCTION - Unit 3 Code Generation Part 2 implementation, public
  integration, properties, notebook, docs, regression, distribution, summary,
  and generated-code approval complete.
- **Unit 3 Code Generation plan**:
  `aidlc-docs/construction/plans/utility-placement-public-workflow-presentation-integration-code-generation-plan.md`.
- **Unit 3 Code Generation summary**:
  `aidlc-docs/construction/utility-placement-public-workflow-presentation-integration/code/code-summary.md`.
- **Executable notebook**:
  `OpenPinch/data/notebooks/19_utility_placement_optimisation.ipynb`.
- **Unit 2 Functional Design plan**:
  `aidlc-docs/construction/plans/utility-placement-evaluation-optimisation-service-functional-design-plan.md`.
- **Unit 2 Functional Design artifacts**:
  `aidlc-docs/construction/utility-placement-evaluation-optimisation-service/functional-design/`.
- **Unit 2 NFR Requirements plan**:
  `aidlc-docs/construction/plans/utility-placement-evaluation-optimisation-service-nfr-requirements-plan.md`.
- **Unit 2 NFR Requirements artifacts**:
  `aidlc-docs/construction/utility-placement-evaluation-optimisation-service/nfr-requirements/`.
- **Unit 2 NFR Design plan**:
  `aidlc-docs/construction/plans/utility-placement-evaluation-optimisation-service-nfr-design-plan.md`.
- **Unit 2 NFR Design artifacts**:
  `aidlc-docs/construction/utility-placement-evaluation-optimisation-service/nfr-design/`.
- **Unit 2 Code Generation plan**:
  `aidlc-docs/construction/plans/utility-placement-evaluation-optimisation-service-code-generation-plan.md`.
- **Unit 1 Code Generation summary**: `aidlc-docs/construction/utility-placement-contracts-pure-model/code/code-summary.md`.
- **Unit 1 Code Generation plan**: `aidlc-docs/construction/plans/utility-placement-contracts-pure-model-code-generation-plan.md`.
- **Unit 1 NFR Design plan**: `aidlc-docs/construction/plans/utility-placement-contracts-pure-model-nfr-design-plan.md`.
- **Unit 1 NFR Design artifacts**: `aidlc-docs/construction/utility-placement-contracts-pure-model/nfr-design/`.
- **Unit 1 NFR Requirements plan**: `aidlc-docs/construction/plans/utility-placement-contracts-pure-model-nfr-requirements-plan.md`.
- **Unit 1 NFR Requirements artifacts**: `aidlc-docs/construction/utility-placement-contracts-pure-model/nfr-requirements/`.
- **Unit 1 Functional Design plan**: `aidlc-docs/construction/plans/utility-placement-contracts-pure-model-functional-design-plan.md`.
- **Unit 1 Functional Design artifacts**: `aidlc-docs/construction/utility-placement-contracts-pure-model/functional-design/`.
- [x] CONSTRUCTION - All unit Functional Design stages completed with blocking PBT-01.
- [x] CONSTRUCTION - All applicable unit NFR Requirements and NFR Design stages completed.
- [x] CONSTRUCTION - Infrastructure Design skipped; no infrastructure change.
- [x] CONSTRUCTION - TDD Code Generation complete and approved for Units 1, 2, and 3.
- [x] CONSTRUCTION - Integrated Build and Test complete; summary at
  `aidlc-docs/construction/build-and-test/build-and-test-summary.md`.
- [x] OPERATIONS - N/A for Utility Placement; the workflow stage is a
  placeholder and no deployment or publication was requested.
- **Current scope**: Utility placement accepts at least two isothermal levels
  and optional sensible levels per side and minimizes physical entropy
  generation from balanced composite curves. Monetary placement is deferred.
- **Testing approach**: Test-driven development is mandatory for construction.
- **Example delivery**: Unit 3 owns
  `OpenPinch/data/notebooks/19_utility_placement_optimisation.ipynb`; no
  utility-placement CLI is planned.
- **Question artifact**: `aidlc-docs/inception/requirements/utility-placement-requirement-verification-questions.md`.
- **Clarification artifact**: `aidlc-docs/inception/requirements/utility-placement-requirements-clarification-questions.md`.
- **Requirements artifact**: `aidlc-docs/inception/requirements/utility-placement-requirements.md`.
- **Requirements approval**: `aidlc-docs/inception/requirements/utility-placement-requirements-approval.md`.
- **User Stories assessment**: `aidlc-docs/inception/plans/utility-placement-user-stories-assessment.md`.
- **Story generation plan**: `aidlc-docs/inception/plans/story-generation-plan.md`.
- **Story plan approval**: `aidlc-docs/inception/plans/utility-placement-story-plan-approval.md`.
- **Personas**: `aidlc-docs/inception/user-stories/utility-placement/personas.md`.
- **Stories**: `aidlc-docs/inception/user-stories/utility-placement/stories.md`.
- **Generated stories approval**: `aidlc-docs/inception/user-stories/utility-placement/approval.md`.
- **Execution plan**: `aidlc-docs/inception/plans/utility-placement-execution-plan.md`.
- **Workflow plan approval**: `aidlc-docs/inception/plans/utility-placement-workflow-plan-approval.md`.
- **Application Design plan**: `aidlc-docs/inception/plans/application-design-plan.md` (Utility Placement section).
- **Application Design artifacts**: Utility Placement sections in `aidlc-docs/inception/application-design/components.md`, `component-methods.md`, `services.md`, `component-dependency.md`, and `application-design.md`.
- **Application Design approval**: `aidlc-docs/inception/application-design/utility-placement-application-design-approval.md`.
- **Unit plan**: `aidlc-docs/inception/plans/unit-of-work-plan.md` (Utility Placement section).
- **Unit artifacts**: Utility Placement sections in `aidlc-docs/inception/application-design/unit-of-work.md`, `unit-of-work-dependency.md`, and `unit-of-work-story-map.md`.
- **Units approval**: `aidlc-docs/inception/application-design/utility-placement-units-approval.md`.
- **Extensions**: Full Property-Based Testing enforcement enabled; Security and Resiliency disabled.
- **Corrective closure**: A final thermodynamic sanity audit replaced empty
  problem-table-derived entropy evidence with period-resolved physical process
  stream/segment slices at real temperatures and calibrated Total Site rounded
  profiles to exact target residuals. The refreshed focused gate passed 278
  tests with 3 skips at 97 percent application/presentation branch coverage;
  the complete non-solver gate passed 2,456 tests with 3 skips and 4 solver
  deselections, with the sandbox-only Chrome export passing under local-browser
  permission.
- [x] SCOPE AMENDMENT - The executable notebook requires two isothermal and two
  sensible utility levels per side for the thermodynamic objective and standard
  GCC or Total Site Profile output with the optimized utilities.
- [x] CONSTRUCTION - Mixed-level graph amendment requirements and TDD plan
  approved under the user's continuing authorization.
- [x] CONSTRUCTION - Mixed-level graph amendment Code Generation complete.
- [x] CONSTRUCTION - Mixed-level graph amendment Build and Test refresh complete.
- **Mixed-level graph closure**: The canonical notebook executes two
  isothermal plus two sensible levels per hot/cold side and displays the
  optimized direct utility profile on a standard GCC. The public
  detached plot selects a TSP for Total Site results. The refreshed focused
  gate passed 397 tests with 3 skips at 97 percent utility-placement
  application/presentation branch coverage; the complete non-solver gate
  passed 2,458 tests with 3 skips and 4 solver deselections, with the unchanged
  sandbox-only Chrome export passing under local-browser permission.
- [x] REQUIREMENTS CORRECTION - `problem.plot.utility_placement(...)` is
  explicitly excluded. The single notebook shall replace utilities in a new
  named case and display that case with the existing standard GCC and Total
  Site Profile plot methods.
- [x] CONSTRUCTION - Named-case replacement correction complete through Unit 3
  Code Generation and integrated Build and Test under the user's continuing
  authorization.
- **Named-case correction closure**: The dedicated placement plot accessor and
  graph adapter were removed. Notebook 19 replaces eight optimized utilities
  in a new case, runs ordinary direct and Total Site targets, and displays the
  standard GCC and TSP while leaving the baseline unchanged. The focused gate
  passed 169 tests at 97 percent application branch coverage; the complete
  non-solver gate passed 2,459 tests with 3 skips and 4 solver deselections;
  Ruff, 54-page warnings-as-errors documentation, exact 194/194 inventory,
  fresh archives, and wheel-only notebook execution pass.
- [x] REQUIREMENTS CORRECTION - Thermodynamic placement minimizes the entropy-
  weighted area between the residual process profile and candidate utility
  profile; whole-process entropy is not the placement objective.
- [x] CONSTRUCTION - Profile-gap thermodynamic correction complete through Unit
  2/3 Code Generation and refreshed integrated Build and Test under the user's
  continuing authorization.
- **Profile-gap correction closure**: The objective now integrates absolute
  residual-process/utility horizontal separation over reciprocal absolute
  temperature. The corrected notebook result uses active hot utility
  178.01 to 148.006 degC and active cold utility 2.99 to 32.994 degC at
  0.8764152060442507 kW/K. The focused ownership gate passed 179 tests and the
  complete focused/notebook/documentation gate passed 216 tests with 3 guarded
  skips; the kernel has 100 percent branch coverage; the fixed-seed non-solver
  gate passed 2,469 tests with 3 guarded skips and 4 solver deselections. Ruff,
  54-source warnings-as-errors documentation, fresh archives, and final
  installed-wheel notebook execution pass.
- [x] REQUIREMENTS CORRECTION - Physical thermodynamic cost uses logarithmic
  entropy generation from candidate-local balanced composite curves; generated
  default `HU`/`CU` utilities are excluded through a deterministic infeasible
  penalty.
- [x] CONSTRUCTION - Balanced-composite entropy and default-utility penalty
  correction complete through refreshed integrated Build and Test under
  continuing authorization.
- **Balanced-composite correction closure**: Thermodynamic placement now builds
  candidate-local real-temperature balanced composite curves and evaluates
  physical `CP * ln(T_out / T_in)` sensible entropy plus signed `Q/T`
  isothermal limits. Positive generated `HU`/`CU` duty is infeasible. The two
  corrected kernels have 100 percent branch coverage; the full non-solver gate
  passed 2,480 tests with 3 guarded skips and 4 solver deselections. Ruff,
  patch hygiene, warning-clean Sphinx, fresh source/wheel archives, and the
  installed-wheel notebook pass. The notebook thermodynamic objective is
  1.0940185892615926 kW/K and its entropy decomposition closes exactly.
- **Thermodynamic-only closure**: The monetary objective selector, price and
  cogeneration template fields, electricity/turbine inputs, monetary result
  types, placement-specific economics/cogeneration modules, reporting columns,
  documentation path, and second notebook workflow were removed. General
  cogeneration analysis remains unchanged. The final review removed cosmetic
  rewrites of notebooks 1-18, made the canonical generator preserve equivalent
  checked-in notebooks, and corrected stale RTD scope and coverage text. The
  complete post-review suite passed 2,385 tests with 4 expected skips and
  solver markers enabled; Ruff, patch hygiene, warning-clean documentation,
  fresh archives, and installed-wheel notebook execution pass.
- **API usability closure**: The target call now returns a normal unsolved
  `PinchProblem` containing the best utility set; the source remains unchanged.
  `workspace.add(case, name=..., activate=False)` registers it and preserves
  placement evidence. The verbose count keywords, detailed-result return,
  placement-specific presentation methods/module, notebook result traversal,
  and manual utility-dictionary construction were removed. The focused gate
  passed 297 tests with 3 expected skips; the full solver-enabled suite passed
  2,389 tests with 4 expected skips. Ruff, patch hygiene, warning-clean RTD,
  fresh archives, and isolated installed-wheel notebook execution pass.
- **Balanced-composite duty-conservation closure**: The imbalance was caused
  by thermodynamic reconstruction, not by an unbalanced targeting allocation.
  A process breakpoint approximately `0.00000795 K` inside a near-isothermal
  utility span was removed by the problem-table temperature tolerance, losing
  about `2.625 kW` because that utility had a large heat-capacity flow. Utility
  entropy profiles now use exact allocated duty and analytical temperature
  fractions, so every subdivided grid conserves endpoint duty. The exact local
  Process and Site cells complete in 355.17 seconds with objectives
  `0.03524514018747493 kW/K` and `0.3452285101943513 kW/K`, no invalid-composite
  diagnostic, and an unchanged baseline. The broad solver-enabled gate passes
  2,456 tests with 4 expected skips and 3 local-notebook metadata/generator
  deselections. The user's notebook and `.gitignore` remain unmodified.

## Utility Placement Canonical Fallback Penalty Progress

- [x] INCEPTION - Reused the current brownfield workspace and reverse-engineering artifacts.
- [x] INCEPTION - Minimal Requirements Analysis completed from the explicit correction request.
- [x] INCEPTION - User Stories, Application Design, Units Generation, and NFR stages skipped.
- [x] INCEPTION - Bounded workflow and Code Generation plan approved by the user's direct instruction.
- [x] CONSTRUCTION - RED example, integration, and canonical-oracle property coverage complete with eight expected failures.
- [x] CONSTRUCTION - GREEN canonical squared-penalty delegation and oracle property complete.
- [x] CONSTRUCTION - Focused Build and Test complete.
- [x] OPERATIONS - N/A; no deployment change requested.
- **Requirements**: `aidlc-docs/inception/requirements/utility-placement-canonical-fallback-penalty-requirements.md`.
- **Plan**: `aidlc-docs/construction/plans/utility-placement-canonical-fallback-penalty-code-generation-plan.md`.
- **Implementation summary**: `aidlc-docs/construction/utility-placement-canonical-fallback-penalty/code/implementation-summary.md`.
- **Build and Test summary**: `aidlc-docs/construction/utility-placement-canonical-fallback-penalty/build-and-test/build-and-test-summary.md`.
- **Outcome**: normalized hot/cold fallback residuals now delegate to canonical `g_ineq_penalty` with `PenaltyForm.SQUARE` and default `rho=10`. Twenty-seven focused checks and the 283 completed broad checks pass; the user-modified Notebook 19 remains untouched.
- **Extensions**: Full Property-Based Testing enabled; Security and Resiliency disabled.

## Indirect Profile Precision Progress

- [x] INCEPTION - Workspace Detection reused the current brownfield assessment.
- [x] INCEPTION - Reverse Engineering reused current targeting artifacts.
- [x] INCEPTION - Minimal Requirements Analysis completed from the clear defect
  report and approved by the user's direct correction request.
- [x] INCEPTION - User Stories, Application Design, and Units Generation
  skipped for this bounded numerical bug fix.
- [x] INCEPTION - Workflow Planning completed and approved by the correction
  request.
- [x] CONSTRUCTION - Functional, NFR, and Infrastructure Design skipped.
- [x] CONSTRUCTION - Code Generation Part 1 plan created and approved by the
  correction request.
- [x] CONSTRUCTION - Code Generation Part 2 complete.
- [x] CONSTRUCTION - Build and Test complete.
- [x] OPERATIONS - N/A; no deployment change requested.
- **Requirements**:
  `aidlc-docs/inception/requirements/indirect-profile-precision-requirements.md`.
- **Code Generation plan**:
  `aidlc-docs/construction/plans/indirect-profile-precision-code-generation-plan.md`.
- **Implementation summary**:
  `aidlc-docs/construction/indirect-profile-precision/code/code-generation-summary.md`.
- **Build and Test summary**:
  `aidlc-docs/construction/indirect-profile-precision/build-and-test/build-and-test-summary.md`.
- **Current stage**: Complete. Direct target cascades remain full precision,
  graph payloads remain four-decimal, and indirect reconstruction preserves
  fine temperature intervals and duties. All focused, complete-suite,
  notebook, packaging, static, distribution, and wheel gates pass.
- **Extensions**: Security and Resiliency remain disabled; partial PBT applies
  to numerical reconstruction invariants.

## Indirect Target Terminology Progress

- [x] INCEPTION - Workspace Detection reused the current brownfield assessment.
- [x] INCEPTION - Reverse Engineering reused current target/report artifacts.
- [x] INCEPTION - Standard Requirements Analysis completed and approved by the
  supplied implementation plan.
- [x] INCEPTION - User Stories skipped for this technical contract refactor.
- [x] INCEPTION - Workflow Planning completed and approved.
- [x] INCEPTION - Application Design and Units Generation skipped; existing
  boundaries and one cohesive implementation unit are sufficient.
- [x] CONSTRUCTION - Functional, NFR, and Infrastructure Design skipped under
  the approved workflow.
- [x] CONSTRUCTION - Code Generation Part 1 plan created and approved by the
  explicit implementation request.
- [x] CONSTRUCTION - Code Generation Part 2 complete.
- [x] CONSTRUCTION - Build and Test complete.
- [x] OPERATIONS - N/A; no deployment work requested.
- **Requirements**:
  `aidlc-docs/inception/requirements/indirect-target-terminology-requirements.md`.
- **Workflow plan**:
  `aidlc-docs/inception/plans/indirect-target-terminology-execution-plan.md`.
- **Code Generation plan**:
  `aidlc-docs/construction/plans/indirect-target-terminology-code-generation-plan.md`.
- **Implementation summary**:
  `aidlc-docs/construction/indirect-target-terminology/code/code-generation-summary.md`.
- **Build and Test summary**:
  `aidlc-docs/construction/indirect-target-terminology/build-and-test/build-and-test-summary.md`.
- **Current stage**: Complete. The generic indirect model and explicit public
  metadata contract are implemented. Notebook 2 retains the corrected duties
  and LPS ledge. The complete fixed-seed non-solver suite passed 2,195 tests
  with 3 optional-profile skips and 4 solver deselections; packaging, Ruff,
  Sphinx, distribution, isolated-wheel, stale-symbol, and patch gates pass.
- **Extensions**: Security and Resiliency disabled; partial PBT applies to
  serialization and multi-period metadata alignment.

## Total Site Profiles Diagnostic

- [x] Reproduced Notebook 2 with `pulp_mill.json` and captured the Total Site
  process-profile and utility-profile duties.
- [x] Traced direct GCC segment conversion, child-first multi-scale targeting,
  net-stream import, Total Process utility aggregation, and Total Site graph
  serialization.
- [x] Confirmed the cause with a controlled replay that restored immediate
  Process Zone direct-GCC profiles before the Site Total Site solve.
- **Finding**: Process Zone indirect targeting overwrites each zone's
  direct-GCC-derived `net_hot_streams` and `net_cold_streams` with Unit Operation
  child profiles. Site targeting then combines those child-level profiles with
  Process Zone direct utility targets, mixing two hierarchy levels.
- **Observed residual**: Both duty identities were overstated by 59,168.043 kW.
- **Controlled replay residual**: 0.103 kW for Hot Utility versus Cold CC and
  0.0623 kW for Cold Utility versus Hot CC after restoring immediate direct
  profiles; the remaining difference is graph-table rounding/reconstruction.
- **Status**: Root cause diagnosed. No production-code change was authorized or
  made in this investigation.

## Total Site Profile Hierarchy Fix Progress

- [x] INCEPTION - Workspace Detection reused the current brownfield assessment.
- [x] INCEPTION - Reverse Engineering reused current targeting artifacts.
- [x] INCEPTION - Minimal Requirements Analysis completed and approved by the
  user's decision-complete implementation plan.
- [x] INCEPTION - User Stories skipped for this bounded internal defect fix.
- [x] INCEPTION - Workflow Planning completed and approved by the user's
  implementation request.
- [x] INCEPTION - Application Design and Units Generation skipped; one existing
  targeting component is corrected without a new public contract.
- [x] CONSTRUCTION - Functional, NFR, and Infrastructure Design skipped under
  the approved workflow.
- [x] CONSTRUCTION - Code Generation Part 1 plan created and approved by the
  user's explicit request to implement the supplied plan.
- [x] CONSTRUCTION - Code Generation Part 2 complete.
- [x] CONSTRUCTION - Build and Test complete.
- [x] OPERATIONS - N/A; no deployment or operational change was requested.
- [x] CONSTRUCTION - Explicit second per-zone net-profile pair added and
  verified at user request.
- [x] CONSTRUCTION - Notebook 2 rounded SUGCC LPS ledge restored and verified.
- **Requirements**:
  `aidlc-docs/inception/requirements/total-site-profile-hierarchy-fix-requirements.md`.
- **Workflow plan**:
  `aidlc-docs/inception/plans/total-site-profile-hierarchy-fix-execution-plan.md`.
- **Code Generation plan**:
  `aidlc-docs/construction/plans/total-site-profile-hierarchy-fix-code-generation-plan.md`.
- **Implementation summary**:
  `aidlc-docs/construction/total-site-profile-hierarchy-fix/code/code-generation-summary.md`.
- **Build and Test summary**:
  `aidlc-docs/construction/total-site-profile-hierarchy-fix/build-and-test/build-and-test-summary.md`.
- **Current stage**: Complete. Consecutive identical graph coordinates are
  removed before collinearity cleanup, preserving the HPS-to-LPS vertical
  connection and approximately 138.5 degC LPS ledge. The expanded focused suite
  passed 145 tests; Notebook 2 passed; the complete fixed-seed non-solver suite
  passed 2,191 tests with 3 optional-profile skips and 4 solver deselections;
  84 packaging tests, repository Ruff, fresh distributions, direct built-wheel
  corner smoke, and patch hygiene passed.
- **Extension decisions**: Security and Resiliency disabled; partial PBT applies
  to the deterministic reconstruction invariant.

## Notebook Improvement Progress

- [x] INCEPTION - Workspace Detection completed by reusing the current
  brownfield assessment.
- [x] INCEPTION - Reverse Engineering reused because the current architecture,
  component inventory, and technology stack cover the notebook subsystem.
- [x] INCEPTION - Initial Requirements Analysis questions answered and
  validated.
- [x] INCEPTION - Presentation clarification answers received and validated.
- [x] INCEPTION - Standard Requirements Analysis artifact generated.
- [x] INCEPTION - Requirements approved by the user.
- [x] INCEPTION - User Stories assessment completed; the stage is required for
  the customer-facing learning workflow.
- [x] INCEPTION - Story-generation plan and methodology questions created.
- [x] INCEPTION - Story-generation answers validated; 18 notebook-level stories
  will be grouped by execution profile.
- [x] INCEPTION - Story-generation plan approved by the user's response "Go".
- [x] INCEPTION - Two personas and 18 profile-grouped notebook stories generated
  with Given/When/Then criteria, INVEST verification, and full traceability.
- [x] INCEPTION - Generated User Stories approved by the user's instruction to
  continue through completion.
- [x] INCEPTION - Workflow Planning completed and approved under completion
  authorization.
- [x] INCEPTION - Application Design skipped; no new components or services.
- [x] INCEPTION - Units Generation skipped; one cohesive generator-owned unit.
- [x] CONSTRUCTION - Functional Design skipped; no domain model or complex logic.
- [x] CONSTRUCTION - NFR Requirements and NFR Design skipped; approved NFRs use
  the existing stack and patterns.
- [x] CONSTRUCTION - Infrastructure Design skipped; no infrastructure change.
- [x] CONSTRUCTION - Code Generation Part 1 plan created and approved under the
  user's completion authorization.
- [x] CONSTRUCTION - Code Generation Part 2 implementation and focused
  verification complete; approved under completion authorization.
- [x] CONSTRUCTION - Build and Test complete and approved under completion
  authorization.
- [x] OPERATIONS - N/A; no deployment or production runtime change requested.
- **Request clarity**: Vague; the target notebooks, improvement goal, source of
  truth, handling of generated outputs, and verification profile require user
  decisions.
- **Requirements depth**: Standard.
- **Question artifact**:
  `aidlc-docs/inception/requirements/notebook-improvement-requirement-verification-questions.md`.
- **Clarification artifact**:
  `aidlc-docs/inception/requirements/notebook-improvement-requirements-clarification-questions.md`.
- **Requirements artifact**:
  `aidlc-docs/inception/requirements/notebook-improvement-requirements.md`.
- **Next stage after User Stories approval**: Workflow Planning.
- **User Stories assessment**:
  `aidlc-docs/inception/plans/notebook-improvement-user-stories-assessment.md`.
- **Story-generation plan**:
  `aidlc-docs/inception/plans/notebook-improvement-story-generation-plan.md`.
- **Story-plan approval artifact**:
  `aidlc-docs/inception/plans/notebook-improvement-story-plan-approval-questions.md`.
- **Personas artifact**:
  `aidlc-docs/inception/user-stories/notebook-improvement/personas.md`.
- **Stories artifact**:
  `aidlc-docs/inception/user-stories/notebook-improvement/stories.md`.
- **Workflow plan**:
  `aidlc-docs/inception/plans/notebook-improvement-execution-plan.md`.
- **Code Generation plan**:
  `aidlc-docs/construction/plans/notebook-presentation-code-generation-plan.md`.
- **Code Generation summary**:
  `aidlc-docs/construction/notebook-presentation/code/code-generation-summary.md`.
- **Build and Test summary**:
  `aidlc-docs/construction/notebook-presentation/build-and-test/build-and-test-summary.md`.
- **Notebook Improvement status**: Complete. All 18 generator-owned tutorials
  now present a subject-specific inline result before interpretation while
  remaining source-only and unattended. All profile, complete-suite, Ruff,
  Sphinx, distribution, installed-wheel, and patch gates pass.
- **Approval mode**: The user explicitly authorized continuation through
  completion for all remaining approved-scope review gates.
- **Existing notebook state**: The previously observed generated notebook edits
  and `OpenPinch/data/notebooks/openpinch-workspace.json` are gone. Only AI-DLC
  documentation changes for this workflow remain in the working tree.
- **Extension decisions for this workflow**: Security disabled; Resiliency
  disabled; Property-Based Testing partially enabled for pure functions and
  serialization round trips.

## Repository Issue Remediation Progress

- [x] INCEPTION - Workspace Detection reused the current brownfield assessment.
- [x] INCEPTION - Reverse Engineering reused current artifacts; stale current-API
  statements are included in remediation scope.
- [x] INCEPTION - Standard Requirements Analysis completed from six reproduced
  findings and the clean-break contract.
- [x] INCEPTION - User Stories skipped because the work is bounded remediation
  for the existing process-engineer persona.
- [x] INCEPTION - Workflow Planning completed.
- [x] INCEPTION - Workflow plan approved by the user.
- [x] INCEPTION - Minimal Application Design approved by the user.
- [x] INCEPTION - Units Generation Part 1 decomposition plan completed for
  application/filesystem, exact OpenHENS loading, and current documentation;
  generation approved by the user.
- [x] INCEPTION - Units Generation Part 2 artifacts generated and validated;
  final units approved by the user.
- [x] CONSTRUCTION - Unit 1 Functional Design approved by the user.
- [x] CONSTRUCTION - Unit 1 Code Generation Part 1 plan generated; approval is
  approved by the user.
- [x] CONSTRUCTION - Unit 1 Code Generation Part 2 implementation and focused
  verification complete; generated code approved by the user.
  - [x] Step 1 focused baseline: 128 passed.
  - [x] Step 2 problem-state regressions: 5 expected pre-fix failures confirmed.
  - [x] Step 3 identity/containment regressions: 51 expected failures, 7 controls passed.
  - [x] Step 4 workbook-allocation regressions: 4 expected pre-fix failures confirmed.
  - [x] Step 5 shared case-identifier validator: 30 focused tests passed.
  - [x] Step 6 runtime validation/export containment: 58 focused tests passed.
  - [x] Step 7 detached input/multiplier guard: 11 focused tests passed.
  - [x] Step 8 workbook allocation/cleanup: 14 reporting tests passed.
  - [x] Step 9 integrated verification: 203 tests and Ruff/patch gates passed.
  - [x] Step 10 structural review: planned owners only; 34 contract tests passed.
  - [x] Step 11 implementation summary and handoff complete.
- [x] CONSTRUCTION - Unit 2 Functional Design completed under the user's
  authorization to continue through completion.
- [x] CONSTRUCTION - Unit 2 NFR Requirements, NFR Design, and Infrastructure
  Design skipped as approved: no new NFR, stack, or infrastructure decision.
- [x] CONSTRUCTION - Unit 2 Code Generation Part 1 plan completed and approved
  under the user's authorization to continue through completion.
- [x] CONSTRUCTION - Unit 2 Code Generation Part 2 implementation and focused
  verification complete; approved under the user's completion authorization.
  - [x] Focused baseline: 3 passed.
  - [x] Regression-first checkpoint: 5 expected pre-fix failures confirmed.
  - [x] Exact-checkout prerequisite suite: 8 passed.
  - [x] Integrated Unit 1/Unit 2 architecture selection: 123 passed.
  - [x] Ruff lint/format and patch-hygiene checks passed.
- [x] CONSTRUCTION - Unit 3 Functional Design skipped as documentation-only;
  no new data model or business logic is introduced.
- [x] CONSTRUCTION - Unit 3 NFR Requirements, NFR Design, and Infrastructure
  Design skipped under the approved workflow.
- [x] CONSTRUCTION - Unit 3 Code Generation Part 1 plan completed and approved
  under the user's authorization to continue through completion.
- [x] CONSTRUCTION - Unit 3 Code Generation Part 2 documentation and drift
  guards complete; approved under the user's completion authorization.
  - [x] Existing focused baseline: 27 passed.
  - [x] New scoped guard failed before refresh and passed after refresh.
  - [x] Five Mermaid diagrams validated with text alternatives.
  - [x] Documentation, architecture, and entrypoint selection: 70 passed.
  - [x] Ruff lint/format, stale-symbol scan, and patch hygiene passed.
- [x] CONSTRUCTION - Build and Test complete.
  - [x] Complete post-correction non-solver suite: 2,181 passed, 3 skipped,
    4 deselected.
  - [x] Focused non-external OpenHENS/HEN profile: 458 passed, 4 deselected.
  - [x] Repository Ruff lint and 460-file format check passed.
  - [x] Clean 53-source warning-as-error Sphinx build passed.
  - [x] OpenPinch 0.5.2 wheel/source build and installed-wheel smoke passed.
  - [x] Patch hygiene and scoped current-contract scan passed.
- [x] OPERATIONS - N/A; no deployment work requested.
- **Requirements**:
  `aidlc-docs/inception/requirements/repository-issue-remediation-requirements.md`.
- **Workflow plan**:
  `aidlc-docs/inception/plans/repository-issue-remediation-execution-plan.md`.
- **Extensions**: Security and Resiliency remain disabled. Partial PBT applies to
  generated case-name/path containment coverage with the repository seed and
  shrinking retained.

## Historical Stage Progress (Prior Workflows)
- [x] INCEPTION - Workspace Detection
- [x] INCEPTION - Reverse Engineering
- [x] INCEPTION - Requirements Analysis
- [x] INCEPTION - User Stories assessment (skipped: internal technical refactor)
- [x] INCEPTION - Workflow Planning
- [x] INCEPTION - Application Design assessment and design
- [x] INCEPTION - Units Generation assessment and generation
- [x] CONSTRUCTION - Per-unit stages
- [x] CONSTRUCTION - Build and Test
- [x] CONSTRUCTION - Post-Implementation Quality Audit
- [x] CONSTRUCTION - Revalidation after quality corrections
- [x] OPERATIONS - Placeholder (no deployment work requested)

## Reverse Engineering Status
- [x] Reverse Engineering - Completed on 2026-07-12T21:26:45Z
- **Artifacts Location**: aidlc-docs/inception/reverse-engineering/
- **Approval Status**: Approved by the user's explicit request to implement the reviewed comprehensive plan
- **Next Stage After Approval**: Completed; workflow advanced through approved inception artifacts

## Execution Plan Summary
- **Stages to execute**: Application Design, Units Generation, Functional Design, NFR Requirements, NFR Design, Code Generation, Build and Test
- **Stages skipped**: User Stories (internal refactor), Infrastructure Design (no infrastructure change), Operations (placeholder)
- **Units**: Domain and Input; Targeting and Integration; Heat Exchanger Network

## Current Status

- **Lifecycle Phase**: COMPLETE
- **Current workflow**: Repository Issue Remediation
- **Current unit**: Repository Issue Remediation integration
- **Current stage**: Operations N/A
- **Status**: Repository Issue Remediation is complete. Application
  state/filesystem contracts, exact OpenHENS checkout loading, and current
  documentation/drift guards are implemented. The root exposes exactly
  `PinchProblem` and `PinchWorkspace`; concrete application, domain, contracts,
  analysis, optimisation, adapters, and presentation owners remain intact. All
  focused, complete non-solver, Ruff, clean Sphinx, distribution, installed-
  wheel, and patch gates pass.
- **Post-gate correction**: generic mapping-shaped workspace bundles now receive
  the same schema-version and case-key validation as concrete dictionaries. The
  regression, corrected wheel, and installed-artifact guard pass.
- **Compatibility policy**: immediate clean break; no aliases, migration paths,
  deprecated forwarding, or legacy workflow selectors.
- **Extensions**: Security Baseline and Resiliency Baseline are disabled for
  this workflow. Partial Property-Based Testing is enabled for generated case
  identifier and containment invariants with seed `20260715` and shrinking.

## Historical Package Architecture Modernization Status

The following sections record previously completed workflows. Their claims and
test counts are historical evidence, not the active package contract. Current
contract statements are maintained in the preceding status block and in the
refreshed reverse-engineering artifacts.

## Experimental Discopt Removal Progress

- [x] INCEPTION - Workspace Detection resumed from the completed benchmark.
- [x] INCEPTION - Reverse Engineering skipped because current repository
  artifacts already cover the affected HEN solver subsystem.
- [x] INCEPTION - Minimal Requirements Analysis completed.
- [x] INCEPTION - Requirements approved by the user.
- [x] INCEPTION - Workflow Planning completed.
- [x] INCEPTION - Workflow plan approved by the user.
- [x] CONSTRUCTION - Code Generation Part 1 plan completed.
- [x] CONSTRUCTION - Code Generation plan approved by the user.
- [x] CONSTRUCTION - Code Generation Part 2 implementation completed.
- [x] CONSTRUCTION - Generated code approved by the user.
- [x] CONSTRUCTION - Build and Test completed.
- [x] CONSTRUCTION - Build and Test approval superseded by requested Tier 0/1
  regression extension.
- [x] CONSTRUCTION - HEN Tier 0/1 exact regression complete.
- [x] CONSTRUCTION - Build and Test.
- [ ] OPERATIONS - Placeholder; no deployment work requested.
- **Requirements artifact**: `aidlc-docs/inception/requirements/discopt-removal-requirements.md`.
- **Workflow artifact**: `aidlc-docs/inception/plans/discopt-removal-execution-plan.md`.
- **Code Generation plan**: `aidlc-docs/construction/plans/discopt-removal-code-generation-plan.md`.
- **Code Generation summary**: `aidlc-docs/construction/discopt-removal/code/code-generation-summary.md`.
- **Build and Test summary**: `aidlc-docs/construction/discopt-removal/build-and-test/build-and-test-summary.md`.
- **Regression plan**: `aidlc-docs/construction/discopt-removal/build-and-test/hens-tier-0-1-regression-plan.md`.
- **Current work**: Tier 0/1 regression complete. All 14 case/tier pairs match
  pre-segmentation revision `973d2322` exactly under the deterministic result
  contract; focused multiperiod and segmented-stream tests also pass.

## Follow-up Plans
- **Package architecture modernization**: A detailed dependency-ordered Code
  Generation checklist is ready for explicit review at
  `aidlc-docs/construction/plans/package-architecture-modernization-code-generation-plan.md`.
  It keeps `OpenPinch/main.py` as the sole current external contract, creates a
  reusable package-level optimisation capability, moves code into domain,
  contracts, application, analysis, adapters, and presentation owners, and
  retires the old top-level namespaces without facades.
- **Remove compatibility facades**: The user requested a package-wide clean
  break from compatibility-only synthesis import and pickle paths. Requirements
  are approved and Code Generation is active. Intentional root/lib/schema API
  barrels remain; synthesis exports route directly to concrete owners.
- **Package-wide owner-oriented reorganization**: The user approved four
  dependency-ordered units covering completed class extractions, schemas and
  lazy barrels, service-owned helpers, and HEN solver decomposition. Unit 1 is
  active. The detailed checklist is
  `aidlc-docs/construction/plans/package-wide-owner-reorganization-code-generation-plan.md`.
- **Private helper reorganization and parent-owned runtime records**: The user
  approved an intentional breaking cleanup of `OpenPinch.classes`. Inception
  planning is complete and Code Generation is active. The detailed checklist is
  `aidlc-docs/construction/plans/classes-private-helper-reorganization-code-generation-plan.md`.
- **Pre-release corrective review findings**: The user approved a four-PR
  breaking-change implementation plan closing fifteen validated review findings.
  Code Generation Part 1 is complete; PR 1 Domain and Input Correctness is the
  active unit. The execution checklist is
  `aidlc-docs/construction/plans/pre-release-corrective-code-generation-plan.md`.

## Package Usability Refactor Progress

- [x] INCEPTION - Workspace Detection and continuity
- [x] INCEPTION - Requirements Analysis
- [x] INCEPTION - User Stories
- [x] INCEPTION - Workflow Planning
- [x] INCEPTION - Application Design
- [x] INCEPTION - Units Generation
- [x] CONSTRUCTION - Unit 1 Contract and Correctness Foundation
- [x] CONSTRUCTION - Unit 2 PinchProblem Targeting Workflow
- [x] CONSTRUCTION - Unit 3 Component, Design, Plot, and Workspace Workflows
- [x] CONSTRUCTION - Unit 4 Tutorial Templates
- [x] CONSTRUCTION - Unit 5 RTD Coverage and Documentation
- [x] CONSTRUCTION - Build and Test
- [x] OPERATIONS - N/A; no deployment work requested
- **Approval mode**: The user explicitly approved every remaining standardized
  workflow gate through task completion.
- **Compatibility policy**: Clean break; no aliases for retired public workflow
  names or string selectors.
- **Extension configuration**: Security disabled; Resiliency disabled; partial
  Property-Based Testing enabled and blocking where applicable.
- **Unit 1 evidence**: 16 focused tests passed with seed `20260715`; Ruff lint,
  Ruff format, and patch-hygiene checks passed.
- **Completion revalidation**: The post-completion audit expanded the exact
  tutorial denominator from 129 to 186 live operations, including constructors,
  returned Process MVR behavior, and ordered batch target/design/report/export
  surfaces. All 18 notebooks now have process-engineer study questions,
  interpretation, and adaptation guidance. All four notebook execution
  profiles pass. The complete non-solver suite passes 2,084 tests with 3
  opt-in-profile skips and 4 external-solver deselections; Ruff, architecture,
  offline warning-as-error Sphinx, and stale-symbol checks pass.

## Compatibility Shim Canonicalization Progress

- [x] INCEPTION - Workspace Detection and continuity
- [x] INCEPTION - Reverse Engineering reused with focused live audit
- [x] INCEPTION - Requirements Analysis approved by implementation request
- [x] INCEPTION - User Stories reused from package usability refactor
- [x] INCEPTION - Workflow Planning
- [x] INCEPTION - Application Design reused
- [x] INCEPTION - Units Generation skipped; one coupled clean-break unit
- [x] CONSTRUCTION - Functional Design
- [x] CONSTRUCTION - NFR Requirements and Design reused
- [x] CONSTRUCTION - Infrastructure Design skipped
- [x] CONSTRUCTION - Code Generation
- [x] CONSTRUCTION - Build and Test
- [x] OPERATIONS - N/A; no deployment work requested
- **Approval mode**: The user's explicit implementation request approves the
  decision-complete plan, and the earlier blanket approval remains active through
  completion.
- **Compatibility policy**: Immediate clean break with compact wire keys retained but
  no runtime aliases, forwarding facades, transition pages, or migration behavior.
- **Extension configuration**: Security disabled; Resiliency disabled; partial
  Property-Based Testing enabled for round trips, mutation invariants, generators,
  shrinking, and fixed-seed pytest integration.
- **Current stage**: Complete. All ten code-generation steps and every build,
  test, documentation, tutorial, distribution, and isolated-install gate pass.
- **Verification evidence**: 2,089 complete-suite tests passed; the fixed-seed
  non-solver gate passed 2,086 tests with 3 optional skips and 4 solver
  deselections; slow-HPR and HEN solver tutorial profiles passed; Ruff,
  warning-free Sphinx, source/wheel build, isolated wheel smoke, stale-symbol
  checks, and patch hygiene passed.
- **Requirements**:
  `aidlc-docs/inception/requirements/compatibility-shim-canonicalization-requirements.md`.
- **Workflow plan**:
  `aidlc-docs/inception/plans/compatibility-shim-canonicalization-execution-plan.md`.
- **Functional design**:
  `aidlc-docs/construction/compatibility-shim-canonicalization/functional-design/`.
- **Code Generation plan**:
  `aidlc-docs/construction/plans/compatibility-shim-canonicalization-code-generation-plan.md`.
- **Code Generation summary**:
  `aidlc-docs/construction/compatibility-shim-canonicalization/code/code-generation-summary.md`.
- **Build and Test summary**:
  `aidlc-docs/construction/compatibility-shim-canonicalization/build-and-test/build-and-test-summary.md`.

## Residual Compatibility Shim Cleanup Progress

- [x] INCEPTION - Workspace Detection and continuity reused
- [x] INCEPTION - Reverse Engineering reused with focused repository scan
- [x] INCEPTION - Requirements clarification complete
- [x] INCEPTION - Requirements Analysis artifact complete
- [x] INCEPTION - Requirements approval
- [x] INCEPTION - User Stories assessment (skipped: internal refactor)
- [x] INCEPTION - Workflow Planning artifact complete
- [x] INCEPTION - Workflow plan approval
- [x] INCEPTION - Application Design assessment (skipped: existing boundaries)
- [x] INCEPTION - Units Generation assessment (skipped: one coupled unit)
- [x] CONSTRUCTION - Functional Design assessment (skipped: requirements are sufficient)
- [x] CONSTRUCTION - NFR Requirements and Design assessment (skipped: no new NFRs)
- [x] CONSTRUCTION - Infrastructure Design assessment (skipped: no infrastructure)
- [x] CONSTRUCTION - Code Generation Part 1 planning
- [x] CONSTRUCTION - Code Generation Part 2 implementation
- [x] CONSTRUCTION - Build and Test
- [x] OPERATIONS - N/A; no deployment work requested
- **Compatibility policy**: Repository-wide clean break. Remove genuine aliases,
  dependency-version retries, upstream monkeypatches, and transition pages while
  retaining canonical engineering normalization, algorithmic resilience, wire
  contracts, optional-dependency guards, and solver-shape invariants.
- **Requirements**:
  `aidlc-docs/inception/requirements/residual-compatibility-shim-cleanup-requirements.md`.
- **Approval file**:
  `aidlc-docs/inception/requirements/residual-compatibility-shim-cleanup-approval.md`.
- **Workflow plan**:
  `aidlc-docs/inception/plans/residual-compatibility-shim-cleanup-execution-plan.md`.
- **Workflow approval file**:
  `aidlc-docs/inception/plans/residual-compatibility-shim-cleanup-plan-approval.md`.
- **Code Generation plan**:
  `aidlc-docs/construction/plans/residual-compatibility-shim-cleanup-code-generation-plan.md`.
- **Approval mode**: The user explicitly approved the workflow through task completion;
  the approval covers the detailed dependency-ordered Code Generation plan and all
  remaining standardized gates.
- **Extension configuration**: Security and Resiliency disabled. Partial PBT applies
  to penalty and unit-group invariants with Hypothesis seed `20260715`.
- **Current stage**: Complete. Enum-only penalty selection, canonical unit-group
  terminology, current Pyomo availability, an unmodified OpenHENS prerequisite,
  removal of the final transition page, and static retirement guards are in place.
- **Verification evidence**: The affected suite passed 275 tests; the complete
  fixed-seed non-solver suite passed 2,108 tests with 3 intentional opt-in skips
  and 4 solver deselections; the real HEN solver profile passed 3 tests with 1
  intentional nine-stream skip. Ruff checked all 460 Python files, Sphinx built
  53 sources warning-free, both distributions built, the isolated wheel smoke
  passed, and stale-symbol and patch-hygiene checks are clean.
- **Solver regression correction**: Live four-stream runs retained the exact
  checked-in objective and design while producing 99 and 97 conditionally
  generated ESM branches. The regression now enforces a 95-to-100 branch bound,
  matching the existing bounded live-solver policy without weakening design,
  topology, or cost assertions.
- **Code Generation summary**:
  `aidlc-docs/construction/residual-compatibility-shim-cleanup/code/implementation-summary.md`.
- **Build and Test summary**:
  `aidlc-docs/construction/residual-compatibility-shim-cleanup/build-and-test/build-and-test-summary.md`.

## Private Helper Reorganization Progress

- [x] INCEPTION - Workspace Detection
- [x] INCEPTION - Reverse Engineering (reused existing artifacts; focused
  affected-module analysis complete)
- [x] INCEPTION - Requirements Analysis
- [x] INCEPTION - User Stories assessment (skipped: internal refactor)
- [x] INCEPTION - Workflow Planning
- [x] INCEPTION - Application Design assessment (skipped: existing component
  boundaries and approved ownership design)
- [x] INCEPTION - Units Generation assessment (skipped: one coupled unit)
- [x] CONSTRUCTION - Code Generation
- [x] CONSTRUCTION - Build and Test
- [x] OPERATIONS - N/A
- **Current stage**: Complete. The owner-oriented hierarchy, parent-owned
  runtime records, integration updates, documentation, generated properties,
  and complete verification matrix are green.
- **Compatibility policy**: Intentional clean break; no public aliases,
  compatibility imports, pickle migration, or version bump.
- **Extension configuration**: Security disabled; Resiliency disabled;
  Property-Based Testing Partial enabled and blocking where applicable.
- **Segment batch update and pricing**: Approved Code Generation plan created at
  `aidlc-docs/construction/plans/segment-batch-update-and-pricing-code-generation-plan.md`.
  Implementation and Build and Test are complete. Segment prices remain
  independent, utility parent price is duty-weighted, current HEN utility
  selection is preserved, all 1,978 non-solver tests pass at 98% coverage, and
  the solver-marked, Ruff, documentation, packaging, and patch-hygiene gates pass.
- **Segmented parent dt_cont transaction**: Focused Code Generation plan created
  at `aidlc-docs/construction/plans/segmented-stream-dt-cont-transaction-code-generation-plan.md`.
  The plan propagates full and indexed parent assignments through detached child
  candidates with atomic replacement. Code Generation is complete and awaiting
  user review. All 1,960 CI-selected non-solver tests pass, four solver tests are
  deselected, total coverage is 99%, and focused Ruff/patch checks pass.
- **Read the Docs configuration verification**: Workspace detection is complete.
  The tracked root `.readthedocs.yaml` matches the current repository and the
  official Read the Docs v2 schema. Hosted-build troubleshooting was selected;
  the stable-version resolution is awaiting an answer in
  `aidlc-docs/inception/requirements/readthedocs-stable-resolution-questions.md`.
- **Couenne vs APOPT vs Discopt v0.6.0 HEN benchmark**: Feasibility review and
  implementation plan created at
  `aidlc-docs/construction/plans/couenne-apopt-discopt-hen-benchmark-plan.md`;
  approved with a Python 3.14 source-build amendment and now executing Step 7.
- **Staged change quality audit**: Complete at
  `aidlc-docs/construction/plans/staged-change-quality-audit-plan.md`.
- **Stream model refactor**: Requirements and the detailed implementation plan
  are complete and awaiting final review at
  `aidlc-docs/inception/plans/stream-model-refactor-plan.md`.
- **Private helper extractions for input preparation and exchanger area slices**:
  Code generation complete and awaiting review at
  `aidlc-docs/construction/plans/input-preparation-segment-helper-code-generation-plan.md`.
- **Heat exchanger area-slice model refinement**: Plan created at
  `aidlc-docs/inception/plans/heat-exchanger-area-slice-refinement-plan.md` and
  completed, including documentation updates. The completed code-generation
  checklist is at
  `aidlc-docs/construction/plans/hen-area-slice-code-generation-plan.md`.

## Package-Wide Owner Reorganization Progress

- [x] INCEPTION - Workspace Detection
- [x] INCEPTION - Reverse Engineering (reused current artifacts and focused
  package scan)
- [x] INCEPTION - Requirements Analysis
- [x] INCEPTION - User Stories assessment (skipped: internal refactor)
- [x] INCEPTION - Workflow Planning
- [x] INCEPTION - Application Design (approved owner/composition design)
- [x] INCEPTION - Units Generation (four approved units)
- [x] CONSTRUCTION - Unit 1 Complete Existing Class Extractions
- [x] CONSTRUCTION - Unit 2 Schemas and Package Barrels
- [x] CONSTRUCTION - Unit 3 Service-Owned Helpers and Runtime Records
- [x] CONSTRUCTION - Unit 4 HEN Equation and Solver Internals
- [x] CONSTRUCTION - Build and Test
- [x] OPERATIONS - N/A
- **Compatibility policy**: Preserve documented APIs and public schemas; remove
  only the runtime and solver-state aliases named in the approved plan.
- **Extension configuration**: Security disabled; Resiliency disabled;
  Property-Based Testing Partial enabled and blocking where applicable.
- **Unit 1 evidence**: Semantic logic, interval insertion, segment transactions,
  Value coercion/units, and workspace views now have concrete owner modules.
  Focused seeded tests passed (130 tests) and focused Ruff checks passed.
- **Unit 2 evidence**: Synthesis models now have concrete common/topology/method/
  task/result owners with compatibility facades and no reverse barrel imports.
  The classes, lib, and schemas barrels are typed and lazy. Schema, pickle,
  cold-import, public API, structural, and Ruff checks passed (108 tests).
- **Final evidence**: 2,018 non-solver tests passed at 98% coverage; solver tests
  passed 3 with 1 skip. Ruff lint/format, warning-free Sphinx, ten notebook
  parses, isolated wheel/sdist builds, stale-path checks, and patch hygiene pass.

## Remove Compatibility Facades Progress

- [x] INCEPTION - Workspace Detection and continuity
- [x] INCEPTION - Focused Requirements Analysis
- [x] INCEPTION - User Stories assessment (skipped: import cleanup)
- [x] INCEPTION - Workflow Planning
- [x] CONSTRUCTION - Code Generation
- [x] CONSTRUCTION - Build and Test
- [x] OPERATIONS - N/A
- **Outcome**: Synthesis compatibility modules and package re-exports are
  removed. Public lib/schema barrels map directly to concrete schema owners.
- **Verification**: 2,019 non-solver tests at 98% coverage; 3 solver tests
  passed with 1 skip; Ruff, Sphinx, notebooks, distributions, stale-path, built
  artifact, and patch checks passed.
- **Extensions**: Security and Resiliency disabled (N/A); partial PBT is N/A
  because this cleanup changes import ownership without algorithmic behavior.

## Package Architecture Modernization Progress

- [x] INCEPTION - Approved architecture direction captured through iterative
  package scan and design review.
- [x] CONSTRUCTION - Code Generation Part 1 detailed checklist created.
- [x] CONSTRUCTION - Code Generation Part 1 explicit approval.
- [x] CONSTRUCTION - Code Generation Part 2 implementation.
- [x] CONSTRUCTION - Build and Test.
- [x] CONSTRUCTION - Package-wide compatibility shim audit.
- **Compatibility audit outcome**: Import facades, dynamic barrels, reverse
  re-exports, module injections, and Pydantic field aliases are absent. Four
  residual behavioural shim groups and several duplicate naming aliases remain
  for generated-code review.
- [x] OPERATIONS - N/A; no deployment work requested.
- **External contract**: Only
  `OpenPinch.main.pinch_analysis_service` is compatibility protected.
- **Plan**:
  `aidlc-docs/construction/plans/package-architecture-modernization-code-generation-plan.md`.
- **Current gate**: Implementation and Build and Test are complete; generated
  code is awaiting explicit user review.
- **Post-review correction**: The HEN `results` package is no longer hidden by
  the repository-wide `results/` ignore rule. All four source modules are
  visible to version-control discovery; direct dependent imports, a fresh
  `-E -W` Sphinx build, 209 focused tests, artifact content checks, Ruff, and
  patch hygiene pass. A clean Git-index snapshot reproduced five Ruff `I001`
  failures when the result owner was absent and passed unchanged when the owner
  and ignore exception were restored. New gates reject Git-ignored Python
  source and require HEN result assembly in wheel/sdist artifacts.
- **Step 6 evidence**: HPR optimisation semantics now cross one explicit
  adapter into the reusable package-level optimiser; direct MVR, Process MVR,
  and multiperiod state have concrete owner packages with no legacy facades.
  Exact single/multiperiod fixtures pass alongside 344 HPR/MVR/cycle/contract
  tests, 55 optimisation tests, 59 protected main-contract tests, 14 notebook
  tests, architecture/API gates, and repository-wide Ruff lint/format checks.
- **Requirements**:
  `aidlc-docs/inception/requirements/package-architecture-modernization-requirements.md`.
- **Application design**:
  `aidlc-docs/inception/application-design/package-architecture-modernization-design.md`.
- **Implementation summary**:
  `aidlc-docs/construction/package-architecture-modernization/code/implementation-summary.md`.
- **Build and Test summary**:
  `aidlc-docs/construction/package-architecture-modernization/build-and-test/build-and-test-summary.md`.
- **Final evidence**: 2,039 non-solver tests passed; 3 solver tests passed and
  1 explicitly skipped; combined coverage 96.73%, statement coverage 97.95%,
  branch coverage 92.79%; 59 clean-wheel external-contract tests passed under
  warnings-as-errors; wheel and sdist each contain 326 intended entries.
- **Scores**: Overall quality 9.3/10; Test Gates 9.6/10.
- **Step 7 evidence**: HEN base, StageWise, and pinch-decomposition models are
  concrete coordinators over explicit owner-state composition helpers. Solver
  extraction is split into recovery, utility, period-state, segment-area, and
  metadata owners; controllability models and analysis have separate contract
  and HEN owners; the former HEN `common` package and model barrel are retired.
  The final matrix passed 419 non-solver HEN tests, 3 solver tests with 1
  intentional skip, 2 construction-order regressions, 77 architecture/API/main
  tests, and repository-wide Ruff, format, stale-path, and patch checks.
- **Step 8 evidence**: Production, tests, scripts, notebooks, and source
  documentation import concrete owners. The root package and owner package
  initializers are import-free markers; `classes`, `lib`, `services`, `utils`,
  and `streamlit_webviewer` are physically removed with no aliases, dynamic
  export barrels, or pickle accommodations. The protected `main.py` signature
  and implementation remain unchanged apart from concrete owner imports and a
  corrected module description. The final gate passed 124 focused contract,
  architecture, documentation, notebook, and cold-import tests; full-suite
  collection; Ruff lint; formatting of all 443 Python files; retired-path
  searches; and patch hygiene.
- **Step 9 evidence**: Tests now follow e2e, application, domain, analysis,
  optimisation, adapters, presentation, contracts, architecture, and packaging
  owner layers. Stable test-support paths replaced depth-sensitive fixture
  lookups; private export/helper-only assertions were removed; HPR optimisation
  accepts explicit candidate-search and optimiser seams. AST gates enforce
  allowed directions, exact boundary exceptions, concrete module imports, and
  owner-layer test placement. Fresh-process imports cover every package layer,
  and the installed-wheel smoke protects only `OpenPinch.main`. The seeded
  non-solver matrix passed 2,033 tests with four solver tests deselected;
  combined statement/branch coverage is 97%, the focused main/architecture/PBT
  gate passed 125 tests, and Ruff lint, formatting of 457 files, and patch
  hygiene passed. No shrunk Hypothesis defect occurred.
- **Step 10 interim evidence**: Architecture, support-policy, API, guide,
  example, notebook, and release documentation now identifies
  `OpenPinch.main.pinch_analysis_service` as the sole protected Python import.
  Every advanced notebook carries an unsupported-internal notice, and notebook
  09 contrasts the protected main call with internal owner workflows. Package
  discovery includes all seven owners. A warning-free Sphinx build passes. A
  final PEP 517 isolated 0.5.0 wheel and sdist each contain 322 files with all
  intended owners and no retired package or forwarding-module path. Explicit
  Pydantic v2 report serializers replaced the deprecated encoder configuration.
  A fresh Python 3.14.2 environment installed the wheel and current declared
  dependencies, imported OpenPinch only from site-packages, passed the artifact
  smoke, and passed all 59 external-contract tests with warnings treated as
  errors. The complete final gate remains active.
- **Extensions**: Security and Resiliency disabled (N/A); partial PBT requires
  PBT-02, PBT-03, PBT-07, PBT-08, and PBT-09.

## Pre-Release Corrective Review Progress

- [x] CONSTRUCTION - Code Generation Part 1 plan approved by the user's explicit
  implementation request.
- [x] CONSTRUCTION - PR 1 Domain and Input Correctness.
- [x] CONSTRUCTION - PR 2 Period-Native PDM and Utility Constraints.
- [x] CONSTRUCTION - PR 3 Period-Native HEN Results.
- [x] CONSTRUCTION - PR 4 Summary Isolation and HPR Economics.
- [x] CONSTRUCTION - Final Build and Test.
- **Current stage**: GitHub publication handoff. All four dependency-ordered PR units
  are independently green with implementation, documentation, packaging,
  solver, coverage, and Build and Test evidence complete. PR 3 records bounded
  tier-1 solver timeouts separately from its green correctness matrix.
- **Publication blocker**: The required `gh` CLI is not installed and GitHub DNS
  is unavailable in the restricted environment. Local `develop` is four commits
  ahead of `origin/develop`, so the correct remote base cannot be inferred safely
  and no branch or PR has been published.
- **Compatibility policy**: Pre-release clean breaks; no compatibility shims.
- **Plan**: `aidlc-docs/construction/plans/pre-release-corrective-code-generation-plan.md`.

## Read the Docs Configuration Follow-up Progress
- [x] INCEPTION - Workspace Detection
- [ ] INCEPTION - Requirements Analysis (version-policy answer pending)
- [ ] INCEPTION - Workflow Planning
- [ ] CONSTRUCTION - Code Generation
- [ ] CONSTRUCTION - Build and Test

## Workspace Detection Summary
- Existing code was detected: 322 tracked Python files, plus tests, documentation, examples, scripts, notebooks, and project resources.
- The project is packaged as `OpenPinch` version 0.4.5 and targets Python 3.14.2 or newer.
- Core runtime dependencies include NumPy, pandas, Pint, CoolProp, Pydantic, and SciPy.
- Optional dependency groups cover dashboards, heat-pump cycles, notebooks, and heat-exchanger-network synthesis.
- No previous AI-DLC state or reverse-engineering documentation existed at workflow start.
- Reverse Engineering is complete; the next stage is Requirements Analysis after explicit approval.

## GitHub CI Regression Follow-up Progress
- [x] INCEPTION - Workspace Detection
- [x] INCEPTION - Requirements Analysis
- [x] INCEPTION - User Stories assessment (skipped: isolated internal bug fix)
- [x] INCEPTION - Workflow Planning
- [x] CONSTRUCTION - Code Generation
- [x] CONSTRUCTION - Build and Test
- [ ] OPERATIONS - Placeholder (no deployment work requested)
- **Requirements artifact**: Approved minimal requirements at
  `aidlc-docs/inception/requirements/github-ci-heat-pump-zero-duty-requirements.md`.
- **Workflow artifact**: Focused execution plan ready for review at
  `aidlc-docs/inception/plans/github-ci-heat-pump-zero-duty-execution-plan.md`.
- **Code Generation plan**: Ready for review at
  `aidlc-docs/construction/plans/heat-pump-zero-duty-ci-code-generation-plan.md`.
- **Code Generation summary**: Generated-code details are available at
  `aidlc-docs/construction/github-ci-heat-pump-zero-duty/code/code-generation-summary.md`.
- **Build and Test summary**: Verified results are recorded at
  `aidlc-docs/construction/build-and-test/build-and-test-summary.md`.
- **Current gate**: Build and Test is complete; explicit approval is required
  before closing at the Operations placeholder.

## Residual Compatibility Shim Removal Progress

- [x] CONSTRUCTION - Code Generation Part 1 plan approved by the user's explicit
  implementation request.
- [x] CONSTRUCTION - Baseline findings, affected owners, and protected main
  contract verified.
- [x] CONSTRUCTION - Obsolete HPR helper retries removed and verified.
- [x] CONSTRUCTION - HPR typed records made attribute-only and verified.
- [x] CONSTRUCTION - Optimiser identifiers restricted to canonical values.
- [x] CONSTRUCTION - Legacy `StreamCollection` pickle repair removed.
- [x] CONSTRUCTION - Documentation and release notes updated.
- [x] CONSTRUCTION - Focused and complete Build and Test gates passed.
- [x] CONSTRUCTION - Code Generation summary and review handoff complete.
- **Current stage**: Build and Test complete; generated code awaits explicit
  review. Operations is N/A because no deployment work was requested. The focused gate
  passed 277 tests; the complete non-solver gate passed 2,063 tests with four
  solver-tagged tests deselected. Ruff, warning-free Sphinx, stale-surface, and
  patch-hygiene gates pass.
- **Protected external contract**: `OpenPinch.main.pinch_analysis_service` and its
  canonical request/response behaviour remain unchanged.
- **Compatibility policy**: Intentional clean break for unsupported internals;
  no deprecation period, aliases, migration loaders, or warnings.
- **Plan**:
  `aidlc-docs/construction/plans/residual-compatibility-shims-code-generation-plan.md`.
- **Implementation summary**:
  `aidlc-docs/construction/residual-compatibility-shim-removal/code/implementation-summary.md`.
- **Build and Test summary**:
  `aidlc-docs/construction/build-and-test/build-and-test-summary.md`.
- **Extensions**: Security and Resiliency disabled (N/A); partial PBT is N/A
  because no numerical algorithm changes.

## Post-Implementation Import and Type Audit

- [x] Resolve every internal import target statically, including imports under
  `TYPE_CHECKING`.
- [x] Cold-import all 301 discoverable package modules.
- [x] Run repository-wide Pylint error analysis and classify dynamic-model false
  positives separately from reproducible defects.
- [x] Reproduce candidate runtime failures directly.
- **Runtime import status**: all 301 package modules import successfully in the
  locked development environment.
- **Open findings**: two unresolved type-only module imports, one redundant
  self-import/type redefinition, one wrong period keyword causing a runtime
  `TypeError`, and one uninitialized heat-transfer result for unsupported row
  counts.
- **Type-gate limitation**: no mypy, Pyright, basedpyright, or `ty` executable or
  configuration is present; the audit used AST resolution, runtime imports, and
  Pylint error analysis.
- **Current stage**: Generated-code review; findings reported for explicit fix
  approval.

## Post-Implementation Import and Type Fix Progress

- [x] Code Generation plan approved by the user's explicit fix request.
- [x] Five findings reproduced and classified.
- [x] Regression tests added.
- [x] Type-only imports and Zone self-import corrected.
- [x] Total-site keyword and crossflow validation corrected.
- [x] Focused and complete quality gates passed: 96 focused tests and 2,067
  non-solver tests pass; four solver-tagged tests are deselected.
- [x] Evidence and review handoff complete.
- **Current stage**: Code Generation and Build and Test complete; generated-code
  review requested. All five findings are resolved.
- **Plan**:
  `aidlc-docs/construction/plans/post-implementation-import-type-fixes-code-generation-plan.md`.

## Serialized HEN Target Input Progress

- [x] INCEPTION - Workspace Detection and focused current-package scan.
- [x] INCEPTION - Requirements Analysis approved by explicit implementation request.
- [x] INCEPTION - User Stories and Workflow Planning.
- [x] CONSTRUCTION - Functional Design.
- [x] CONSTRUCTION - Code Generation Part 1 plan approved by explicit request.
- [x] CONSTRUCTION - Code Generation Part 2 implementation.
- [x] CONSTRUCTION - Build and Test.
- [x] OPERATIONS - N/A; no deployment work requested.
- **Current stage**: Code Generation and Build and Test complete; generated-code
  review requested.
- **Plan**:
  `aidlc-docs/construction/plans/serialized-hen-target-input-code-generation-plan.md`.
- **Extensions**: Security and Resiliency disabled; partial PBT requires exact
  JSON round-trip, domain-specific generators, shrinking, and reproducibility.
- **Verification**: 2,091 non-solver tests passed with four solver-marked tests
  deselected; Ruff lint/format, warning-as-error Sphinx, architecture,
  stale-symbol, and patch-hygiene gates passed.

## Serialized HEN JSON-Safety Fix Progress

- [x] Review finding reproduced in runtime, canonical-input, and workspace paths.
- [x] Regression tests added and confirmed failing before the fix.
- [x] `StreamID` made string-backed without compatibility behavior.
- [x] Runtime, canonical-input, and workspace regressions pass after the fix.
- [x] Focused and complete quality gates.
- [x] Build and Test evidence and review handoff.
- **Plan**:
  `aidlc-docs/construction/plans/serialized-hen-json-safety-fix-plan.md`.
- **Extensions**: Security and Resiliency disabled (N/A); partial PBT remains
  enabled for the serialized-network round-trip properties.
- **Verification**: 574 focused tests and 2,093 complete non-solver tests pass;
  four solver-marked tests are deselected. Ruff, Sphinx, stale-symbol, and
  patch-hygiene gates pass.
- **Current stage**: Code Generation and Build and Test complete; generated-code
  review requested. Operations is N/A.

## Root Workflow Exports Progress

- [x] INCEPTION - Minimal requirements and user story recorded from the explicit
  import-contract request.
- [x] CONSTRUCTION - Exact root export and cold-import regressions added.
- [x] CONSTRUCTION - `PinchProblem` and `PinchWorkspace` exported from
  `OpenPinch` with concrete owner identity preserved.
- [x] CONSTRUCTION - Curated documentation and all packaged notebooks migrated
  to package-root workflow imports.
- [x] CONSTRUCTION - Complete affected Build and Test gates and evidence; the
  pre-existing executed-output state of notebook 01 remains explicitly
  isolated from the source import change.
- [x] OPERATIONS - N/A; no deployment work requested.
- **Current stage**: Code Generation and Build and Test complete; generated code
  ready for review.

## Package Usability Refactor Planning Progress

- [x] INCEPTION - Workspace Detection resumed from the current brownfield
  package and ten-notebook execution review.
- [x] INCEPTION - Comprehensive Requirements Analysis.
- [x] INCEPTION - User Stories assessment, personas, and acceptance stories.
- [x] INCEPTION - Workflow Planning and five-unit decomposition.
- [x] INCEPTION - Plan approval.
- [x] INCEPTION - Application Design for canonical target, workspace, and HEN
  application-view contracts.
- [x] INCEPTION - Units Generation approval.
- [x] CONSTRUCTION - Units 1-4 Functional Design and Code Generation.
- [x] CONSTRUCTION - Unit 5 RTD coverage and executable quality gates.
- [x] CONSTRUCTION - Build and Test.
- [x] OPERATIONS - N/A; no deployment work requested.
- **Current stage**: Package Usability Refactor complete; generated code,
  tutorials, RTD coverage, and build-and-test evidence are ready for review.
  The user explicitly approved every remaining AI-DLC gate through task
  completion. Completion prompts and approval decisions will still be logged,
  but they no longer pause execution. The five units are
  dependency ordered as contract/correctness, problem targeting/configuration,
  components/design/workspace/presentation, tutorials, and documentation/
  executable quality gates. The package-usability plan is
  represented by a two-class facade, application-owned workflow accessors,
  mirrored workspace batch operations, an effective-argument resolver, strict
  execution-versus-observation state rules, and a tutorial/RTD manifest
  boundary. Targeting selection belongs to explicit
  `PinchProblem.target.*` methods. The plan uses `all_heat_integration()` for
  bulk direct-plus-Total-Site traversal, retains focused direct and indirect
  heat-integration methods, and removes all `TARGETING_*_ENABLED` selectors.
  The interaction contract now covers every public `problem.*` surface with
  `named kwargs > options > stored config > defaults` precedence, explicit state
  invalidation, and no hidden execution from read/report/plot/export methods.
  Core workflow selectors remain explicit and `HENS_METHOD_SEQUENCE` is removed
  rather than used as configuration fallback. A complete argument review now
  removes OpenPinch-owned closed string answers from normal workflows:
  specialized callables select `carnot_heat_pump()`,
  `carnot_refrigeration()`, vapour-compression, Brayton, MVR,
  cogeneration, HEN, and multiperiod algorithms; booleans express only genuine
  placement/topology decisions; and named load values replace load-mode
  strings. Workspace case batches and plot exports likewise use mirrored
  accessors or method references instead of workflow or graph-type strings.
  The tutorial plan now expands to eighteen focused notebooks for one
  process-engineer persona, with explicit
  multiperiod heat-integration, heat-pump, cogeneration, and HEN-synthesis
  paths, plus multi-segment stream modelling and complete public-method
  coverage. A live-to-canonical map now accounts for every `PinchProblem`,
  `PinchWorkspace`, target, component, design, selected-network, and plot
  operation; retiring symbols remain tracked until removal tests pass. The same
  canonical CSV manifest will render a Read the Docs coverage page linked from
  the tutorial index, notebook series, both workflow API pages, and capability
  matrix.
- **Requirements**:
  `aidlc-docs/inception/requirements/package-usability-refactor-requirements.md`.
- **User stories**:
  `aidlc-docs/inception/user-stories/package-usability-refactor-stories.md`.
- **Execution plan**:
  `aidlc-docs/inception/plans/package-usability-refactor-execution-plan.md`.
- **Feature-to-tutorial map**:
  `aidlc-docs/inception/requirements/pinchproblem-workspace-tutorial-coverage-map.md`.
- **Workflow argument map**:
  `aidlc-docs/inception/requirements/workflow-argument-simplification-map.md`.
- **Compatibility policy**: Clean break for stateful workflow methods; retire
  `pinch_analysis_service` from the supported package experience and preserve
  the exact two-class root surface.
- **Extensions**: Security and Resiliency disabled. Partial PBT applies during
  construction to pure aggregation and normalization policies.
- **Plan**:
  `aidlc-docs/construction/plans/root-workflow-exports-code-generation-plan.md`.
- **Compatibility policy**: No legacy alias or compatibility layer was added;
  the requested package-root imports are the canonical workflow surface.
- **Extensions**: Security and Resiliency disabled (N/A); partial PBT N/A
  because this change affects import ownership rather than numerical logic.
- **Verification**: 2,092 non-solver tests passed with four solver tests and the
  pre-existing notebook-output cleanliness assertion deselected. The isolated
  cleanliness assertion still fails because notebook 01 already contains
  execution counts and outputs; those local results were preserved. Root
  identity/cold imports, curated docs/notebooks, Ruff, warning-as-error Sphinx,
  notebook JSON parsing, stale-contract search, and patch hygiene pass.

## GitHub CI HEN Solver-Isolation Progress

- [x] INCEPTION - Workspace Detection resumed from the current brownfield
  repository.
- [x] INCEPTION - Reverse Engineering reused current repository artifacts.
- [x] INCEPTION - Minimal Requirements Analysis completed.
- [x] INCEPTION - Requirements approval.
- [x] INCEPTION - User Stories assessment (skipped: isolated internal test fix).
- [x] INCEPTION - Workflow Planning artifact completed.
- [x] INCEPTION - Workflow plan approval.
- [x] CONSTRUCTION - Code Generation Part 1 planning.
- [x] CONSTRUCTION - Code Generation plan approval.
- [x] CONSTRUCTION - Code Generation Part 2 implementation.
- [x] CONSTRUCTION - Generated-code approval.
- [x] CONSTRUCTION - Build and Test.
- [x] CONSTRUCTION - Build and Test approval.
- [x] OPERATIONS - N/A; no deployment work requested.
- **Diagnosed cause**: One unmarked owner-boundary test invokes live HEN
  synthesis. Local IDAES binaries mask the dependency, while GitHub's non-solver
  job has neither Couenne nor IPOPT.
- **Recommended repair**: Use the existing fake-executor monkeypatch helper in
  the affected test; do not install solvers or change production behavior.
- **Requirements**:
  `aidlc-docs/inception/requirements/github-ci-hen-solver-isolation-requirements.md`.
- **User Stories assessment**:
  `aidlc-docs/inception/plans/github-ci-hen-solver-isolation-user-stories-assessment.md`.
- **Workflow plan**:
  `aidlc-docs/inception/plans/github-ci-hen-solver-isolation-execution-plan.md`.
- **Code Generation plan**:
  `aidlc-docs/construction/plans/github-ci-hen-solver-isolation-code-generation-plan.md`.
- **Code Generation summary**:
  `aidlc-docs/construction/github-ci-hen-solver-isolation/code/code-generation-summary.md`.
- **Build and Test summary**:
  `aidlc-docs/construction/github-ci-hen-solver-isolation/build-and-test/build-and-test-summary.md`.
- **Current stage**: Complete. Build and Test results are approved; Operations
  is N/A. The exact
  regression passed 1 test, the containing CI-selected module passed 22 tests,
  and Ruff plus patch-hygiene checks passed.
- **Extensions**: Security and Resiliency disabled. Partial PBT is N/A for the
  isolated example repair except existing fixed-seed and framework compliance.

## Heat-Recovery `dt_min` Progress

- [x] DT_MIN TERMINOLOGY CORRECTION - Replaced the new inverse service's
  `approach_temperature` naming family with `dt_min` across its public API,
  contracts, implementation, tests, notebooks, RTD, and workflow artifacts.
  Pre-existing exchanger-network approach-temperature contracts remain out of
  scope. The focused behavioral suite passed 47 tests; notebook and RTD tests
  passed 52 tests with 3 expected skips; warning-strict Sphinx passed; the full
  configured suite passed 2,584 tests with 4 skips; the exact non-solver
  coverage selection passed 2,581 tests with 3 skips and 4 deselections at 96
  percent branch coverage. Distributions built and the isolated Python 3.14
  installed-wheel smoke passed. Ruff, formatting, and patch hygiene passed.
  Plan:
  `aidlc-docs/construction/plans/heat-recovery-dt-min-terminology-code-generation-plan.md`.

- [x] THRESHOLD-LIMIT CORRECTION - Return the greatest positive global `dt_min`
  that retains maximum recovery for threshold problems, then use that boundary
  directly in notebook 02. The Bleaching regression returns approximately
  58.34505 delta_degC at 14,121.972 kW. The inverse feature suite passed 46
  tests; notebook and RTD checks passed 47 tests with 3 expected skips; the
  configured full suite passed 2,583 tests with 4 expected skips; the exact
  non-solver coverage job passed 2,580 tests with 3 expected skips and 4
  deselections at 96 percent branch coverage. Ruff, formatting, and patch
  hygiene passed. Plan:
  `aidlc-docs/construction/plans/heat-recovery-threshold-limit-correction-plan.md`.

- [x] SUPERSEDED NOTEBOOK DT_MIN CORRECTION - A provisional 12,000 kW
  notebook-only workaround was completed locally but superseded before commit
  by the threshold-limit requirement correction. Plan:
  `aidlc-docs/construction/plans/heat-recovery-notebook-dt-min-correction-plan.md`.

- [x] DOCUMENTATION AMENDMENT - Expanded the generated selected-period notebook
  example and comprehensively integrated the inverse `dt_min` workflow throughout
  Read the Docs. The focused gate passes 46 tests with 3 expected skips,
  including notebook execution and warning-strict Sphinx; Ruff and patch
  hygiene also pass. Plan:
  `aidlc-docs/construction/plans/heat-recovery-dt-min-documentation-amendment-code-generation-plan.md`.

- [x] INCEPTION - Workspace Detection resumed from the current brownfield
  repository.
- [x] INCEPTION - Reverse Engineering reused current repository artifacts.
- [x] INCEPTION - Requirements Analysis completed from the approved feature
  specification.
- [x] INCEPTION - Requirements approval supplied by the implementation request.
- [x] INCEPTION - Minimal user-story assessment completed through the explicit
  public API workflows and acceptance scenarios in the approved plan.
- [x] INCEPTION - Workflow Planning completed and approved.
- [x] INCEPTION - Application Design completed and approved.
- [x] CONSTRUCTION - Functional Design completed and approved.
- [x] CONSTRUCTION - NFR Requirements and NFR Design skipped; the established
  numerical and property-testing stack is reused.
- [x] CONSTRUCTION - Infrastructure Design skipped; no infrastructure changes.
- [x] CONSTRUCTION - Code Generation Part 1 planned and approved.
- [x] CONSTRUCTION - Code Generation Part 2 implementation.
- [x] CONSTRUCTION - Build and Test.
- [x] OPERATIONS - N/A; no deployment work requested.
- **Requirements**:
  `aidlc-docs/inception/requirements/heat-recovery-dt-min-requirements.md`.
- **Application design**:
  `aidlc-docs/inception/application-design/heat-recovery-dt-min-application-design.md`.
- **Functional design**:
  `aidlc-docs/construction/heat-recovery-dt-min/functional-design/functional-design.md`.
- **Workflow plan**:
  `aidlc-docs/inception/plans/heat-recovery-dt-min-execution-plan.md`.
- **Code Generation plan**:
  `aidlc-docs/construction/plans/heat-recovery-dt-min-code-generation-plan.md`.
- **Extensions**: Property-Based Testing enabled. Security and Resiliency are
  disabled. PBT-01 through PBT-05 and PBT-07 through PBT-10 apply; PBT-06 is
  N/A because the solver has no persistent mutable state.
- **Code Generation summary**:
  `aidlc-docs/construction/heat-recovery-dt-min/code/code-generation-summary.md`.
- **Build and Test summary**:
  `aidlc-docs/construction/heat-recovery-dt-min/build-and-test/build-and-test-summary.md`.
- **Current stage**: Complete. The service, contracts, orchestration,
  properties, documentation, generated tutorials, full coverage suite,
  distributions, and installed-wheel smoke are verified.

## Heat-Recovery `dt_min` Audit Resolutions

- [x] REQUIREMENTS ANALYSIS - Approved strict scalar inputs, physical
  `1e-6 delta_degC` boundary accuracy, local address-based zone resolution,
  exact-zero status semantics, strict dimensional and relational result
  contracts, canonical-period mapping precedence, and expanded adversarial/PBT
  coverage.
- [x] WORKFLOW PLANNING - Focused one-unit correction workflow approved.
- [x] FUNCTIONAL DESIGN - Strict predicates, precise inverse evaluation,
  bracket verification, local zone resolution, contract invariants, and PBT
  properties approved.
- [x] CODE GENERATION - Plan approved; Red Steps 1 through 4 produced 44
  expected failures and 46 passing existing cases. Green Steps 5 through 9 are
  complete: the six corrections, expanded properties, permanent shrunk
  examples, RTD synchronization, and generated-notebook drift validation pass.
  Step 10 verification and summaries are complete; the user approved the
  generated correction with option B.
- [x] BUILD AND TEST - Verification evidence is complete and approved: 2,641 configured
  tests pass with 4 expected skips; the CI-equivalent selection passes 2,638
  tests with 3 expected skips and 4 solver deselections at 96 percent branch
  coverage; RTD, notebooks, distributions, installed-wheel smoke, Ruff, and
  patch hygiene pass.
- [x] OPERATIONS - N/A; this library-level numerical, contract, test, notebook,
  and documentation correction introduces no deployment or monitoring change.
- **Current stage**: Complete. All six audit findings are resolved and the
  generated correction plus Build and Test evidence are approved.
