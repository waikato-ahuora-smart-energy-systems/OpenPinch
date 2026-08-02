# Units of Work

## Domain and Input

Adds schemas, `StreamSegment`, ordered aggregate behavior, mutation, profile construction, persistence, and numeric projections.

## Targeting and Integration

Adapts problem tables, counting, area targets, HPR, Brayton, and MVR generation to consume or create segmented parents.

## Heat Exchanger Network

Adds segment tensors, cumulative heat-coordinate equations, pinch splitting, segment-summed area, extraction, verification, and presentation support.

## Package Usability Refactor Units

### Unit 1: Contract and Correctness Foundation

**Purpose**: Freeze the intended public contract and correct defects that would
otherwise force tutorials to use private helpers or workarounds.

**Responsibilities**:

- capture regressions for the seven confirmed notebook failures and the real
  shared-HPR weighted-summary defect;
- define aggregation field policies and optional-value behavior;
- freeze the live and target public inventories, root exports, state model,
  argument precedence, and golden process-engineer examples;
- establish failing contract tests before public API migration.

**Exit evidence**: regression tests reproduce each defect; golden examples
compile against the target surface; the inventory and state contracts are
machine readable.

### Unit 2: PinchProblem Interaction, Targeting, and Configuration

**Purpose**: Make `PinchProblem` the sole explicit owner of analysis selection
and establish one predictable argument and state model.

**Responsibilities**:

- add the effective-argument resolver, omitted sentinel, provenance, and
  read-only effective configuration;
- implement descriptive heat-integration, model-specific HPR, cogeneration,
  exergy, energy-transfer, area/cost, and mirrored all-period methods;
- preserve efficient all-zone traversal while removing callable target and
  selector-driven configuration;
- separate mutation/execution from summary, report, plot, comparison, export,
  and dashboard observation;
- simplify lifecycle and serialization and remove obsolete public aliases.

**Exit evidence**: root-only golden examples pass; no analysis-selection config
keys or OpenPinch-owned closed workflow strings remain; numerical parity and
state tests pass.

### Unit 3: Components, Design, Workspace, and Presentation

**Purpose**: Extend the same interaction rules to process modification, HEN
design, named studies, case batches, and publication surfaces.

**Responsibilities**:

- expose `components.add_process_mvr()` with predictable invalidation;
- implement named single-period, enhanced, advanced, and multiperiod HEN
  methods plus application-owned ranked/network/grid views;
- make `scenario()` unsolved and `cases()` a typed ordered batch surface;
- retire variant and workflow-string APIs while retaining case persistence and
  active forwarding;
- replace aggregation and plot-type strings with binary flags and method
  references; guarantee no hidden execution.

**Exit evidence**: component, HEN, workspace, summary, report, plot, and export
golden examples use only root workflow imports and execute exactly the named
operation.

### Unit 4: Capability-Complete Tutorial Suite

**Purpose**: Provide eighteen executable process-engineering study templates
covering every supported core and advanced workflow.

**Responsibilities**:

- rewrite ten stale notebooks and add eight focused notebooks;
- teach first solve, Total Site, multi-segment streams, scenarios, persistence,
  and multiperiod heat integration as reusable core templates;
- teach area/cost/exergy, Carnot, vapour-compression, Brayton, MVR,
  cogeneration, energy transfer, and HEN studies as focused advanced templates;
- declare dependency profile, runtime, deterministic settings, and engineering
  interpretation; remove private imports, hidden reruns, and committed outputs;
- maintain an exact operation and semantic-mode coverage manifest.

**Exit evidence**: all notebooks compile and execute honestly under their
declared profiles; AST and manifest checks report 100 percent canonical
operation and semantic-mode coverage.

### Unit 5: Documentation and Executable Quality Gates

**Purpose**: Publish and enforce the same public experience across RTD, tests,
CI, and distributions.

**Responsibilities**:

- make stateful problem/workspace workflows the supported guides and retire
  `pinch_analysis_service` from the process-engineer experience;
- publish complete problem/workspace interaction matrices and state/config
  rules;
- render the RTD tutorial coverage page from the same CSV used by CI;
- add inventory, AST, clean-kernel, optional-profile, Ruff, Sphinx, stale-symbol,
  package-content, and distribution gates;
- preserve honest base, slow HPR, solver-backed HEN, and guarded-interactive
  results.

**Exit evidence**: docs and CI derive from one canonical manifest; warning-free
Sphinx and clean distributions contain all verified tutorials; all acceptance
gates pass.

## Repository Issue Remediation Units

These are logical construction units inside one Python distribution. They do
not create services, packages, deployment boundaries, or new root exports.

### Unit 1: Application State and Filesystem Contracts

**Purpose**: close the workspace path-escape, mutable-input, workbook-collision,
and unloaded-problem error findings within existing application and reporting
owners.

**Inputs**:

- approved FR-1, FR-2, FR-4, and FR-5;
- existing workspace bundle schema and `PinchWorkspace` case repository;
- existing `PinchProblem` authoritative input and prepared root state;
- existing Excel reporting writer.

**Responsibilities**:

- define and reuse one strict portable case-identifier validator;
- enforce identifier validation in runtime and persisted bundle entry points;
- prove batch export paths remain beneath their selected root;
- return detached `problem_data` snapshots;
- acquire the prepared root before multiplier mutation;
- reserve workbook paths exclusively and clean failed reservations;
- add focused, generated, and concurrency regression coverage.

**Outputs**:

- validated case identifiers with unchanged valid display values;
- contained per-case export directories;
- isolated problem-input observations;
- unique successful workbook paths;
- actionable unloaded-problem errors.

**Exclusions**:

- no stream-field mutation API;
- no case-name aliases or silent sanitization;
- no report sheet/column changes;
- no numerical or solver behavior changes.

**Exit evidence**: all Unit 1 reproductions fail before implementation and pass
afterward; generated path tests retain the repository seed and shrinking; valid
workspace bundles and workbook contents remain stable.

### Unit 2: Exact OpenHENS Checkout Loading

**Purpose**: make comparison execution independent of ambient Python module
state and prove that capabilities originate from the requested checkout.

**Inputs**:

- approved FR-3;
- requested `openhens_root`;
- current required capability list;
- current comparison case runner.

**Responsibilities**:

- isolate existing `openhens` and `openhens.*` module cache entries;
- prioritize the resolved checkout for the scoped import lifetime;
- validate required callables and every available module source origin;
- inject the verified `OpenHENS` factory into source execution;
- restore `sys.path` and `sys.modules` on success and failure;
- preserve failure-before-output behavior.

**Outputs**:

- one verified OpenHENS factory bound to the requested checkout;
- actionable errors for missing, foreign, or unsupported modules;
- restored interpreter import state.

**Exclusions**:

- no upstream OpenHENS mutation or patching;
- no fallback to an installed/cached checkout;
- no solver model or ranking changes.

**Exit evidence**: foreign cached modules are rejected, exact-root modules are
accepted, execution consumes the verified factory, and interpreter state is
identical before and after every tested exit path.

### Unit 3: Current Documentation and Drift Guards

**Purpose**: align current AI-DLC/reverse-engineering documentation with the
canonical root API and prevent current-contract regression.

**Inputs**:

- approved FR-6;
- canonical `OpenPinch.__all__`;
- final Unit 1 and Unit 2 contracts;
- current-state and reverse-engineering documentation.

**Responsibilities**:

- correct active state and current reverse-engineering API descriptions;
- preserve audit and explicitly historical construction records;
- add scoped stale-symbol assertions for retired current-contract claims;
- verify Sphinx from a clean build tree;
- record final implementation and build/test evidence.

**Outputs**:

- current documentation that describes `PinchProblem` and `PinchWorkspace` as
  the only root exports;
- drift tests that exclude historical evidence from false positives;
- warning-free documentation and complete AI-DLC records.

**Exclusions**:

- no historical audit rewriting;
- no reintroduction of `OpenPinch.main` or `pinch_analysis_service`;
- no tutorial or public API expansion.

**Exit evidence**: scoped scans find no retired active contract, Sphinx passes
with warnings as errors, and root/wheel smoke confirms only canonical imports.
