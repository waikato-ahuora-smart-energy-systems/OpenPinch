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

## Utility Placement Optimisation Units

These are logical units inside one OpenPinch package and distribution. They are
not independently deployed services and do not create team or runtime silos.
Construction follows Unit 1 -> Unit 2 -> Unit 3, with red-green-refactor
evidence recorded for every production slice.

### Unit 1: Placement Contracts and Pure Model

**Purpose**: establish the stable specialist vocabulary and deterministic pure
model consumed by every later numerical and public workflow.

**Responsibilities**:

- define objective, base-target, level-kind, and utility-side enums;
- define immutable/Pydantic template, request, options, tolerances, diagnostic,
  candidate, per-period, aggregate, termination, and result contracts;
- define typed placement exceptions and structured candidate-infeasibility data;
- validate separate isothermal and sensible counts and matching hot/cold
  template inventories;
- normalize supplied or generated templates, units, identities, direction,
  fixed near-isothermal span, prices, eligibility, and temperature/span bounds;
- derive the all-period feasible-bound intersection and deterministic valid
  starting points;
- encode and decode decision vectors with stable coordinate ordering;
- guarantee detached JSON serialization and equality within named tolerances;
- provide domain-valid Hypothesis strategies and pure-model reference helpers.

**Primary inputs**: approved placement public arguments, utility-template
metadata, units/configuration conventions, and detached period feasibility
limits supplied by Unit 2 through a stable context value object.

**Outputs**: `UtilityPlacementRequest`, `UtilityTemplateSet`,
`UtilityPlacementModel`, result/diagnostic contracts, exception taxonomy,
bounded vector schema, and reusable test strategies.

**Owned requirements and stories**: FR-002 through FR-006, FR-012 contract
shape, FR-014 contract shape, NFR-001, NFR-002 ordering, NFR-004, NFR-005 pure
boundaries, NFR-006 error vocabulary; UPO-01, UPO-08 contract behavior, UPO-09,
and UPO-12 foundations.

**Exclusions**:

- no process-profile targeting or duty allocation;
- no entropy, monetary, turbine, or aggregate objective calculation;
- no optimiser backend execution;
- no application accessor, cache, workspace batch, or presentation changes;
- no package-root export or new runtime dependency.

**Construction readiness**: comprehensive Functional Design, standard NFR
Requirements and Design, then approved TDD Code Generation. PBT-01 must assign
round-trip, ordering, bound, vector, serialization, and generator properties
before code planning.

**Exit evidence**: focused contract validation and serialization examples;
encode/decode, ordering, fixed-span, bounds, copying, and JSON properties;
fixed-seed reproducibility; Ruff and architecture/import checks for the unit.

### Unit 2: Placement Evaluation and Optimisation Service

**Purpose**: implement the detached numerical capability that evaluates and
optimises one utility placement across direct or Total Site periods.

**Responsibilities**:

- build immutable placement context from isolated direct or Total Site target
  snapshots and preserve canonical period order and weights;
- replay one decoded placement independently for every selected period;
- allocate all residual heating and cooling demand with existing targeting
  calculations and verify both conservation equalities within tolerance;
- integrate utility-side and process-side entropy in absolute temperature,
  including stable near-isothermal limits, and compute exergy destruction;
- compute thermal purchase cost, detached eligible-level cogeneration,
  electricity credit, and net monetary objective;
- form the explicit weighted sum of period objectives without surrogate profiles;
- build and run the existing bounded solver-neutral optimisation contract;
- reject or penalize infeasible candidates deterministically, ensure feasible
  candidates always rank first, and normalize bounded alternatives;
- translate context, targeting, thermodynamic, turbine, non-finite, and
  optimiser failures into Unit 1's typed diagnostics;
- compare bounded analytical cases with brute-force or structured-grid oracles.

**Primary inputs**: Unit 1 request/model/contracts; isolated domain target
inputs; existing direct and Total Site targeting services; existing
solver-neutral minimisation; existing steam-turbine calculation.

**Outputs**: a complete detached `UtilityPlacementResult`, including best and
alternative candidates, period and aggregate coverage/objective evidence,
termination metadata, and diagnostics.

**Owned requirements and stories**: FR-005 bounds consumption; FR-007 through
FR-012; FR-014 numerical result content; NFR-001 through NFR-003, NFR-005, and
NFR-006; UPO-02 numerical workflow, UPO-03 through UPO-08, UPO-11, and UPO-12.

**Exclusions**:

- no public problem/workspace method ownership;
- no canonical `Zone`, utility, target, result, or workspace mutation;
- no private optimisation-backend import or new optimiser implementation;
- no reimplementation of target-profile or steam-turbine physics;
- no implicit reporting or hidden reruns.

**Construction readiness**: comprehensive Functional Design must specify
equations, intervals, feasibility, penalties, aggregation, and oracles before
standard NFR stages and approved TDD Code Generation. PBT-01 must identify
coverage, thermodynamic, objective, feasibility, oracle, copying, and
reproducibility properties.

**Exit evidence**: hand-calculable hot/cold entropy and monetary/cogeneration
examples; direct and Total Site examples; per-period coverage and weighted-sum
properties; feasible-vs-penalty invariant; small-grid oracle comparison;
fixed-seed bounded solve and typed-exhaustion tests.

### Unit 3: Public Workflow and Presentation Integration

**Purpose**: expose the completed specialist service through stable OpenPinch
workflows and observation paths without changing existing study behavior.

**Responsibilities**:

- add explicit `problem.target.utility_placement(...)` keyword arguments and
  immediate request normalization;
- add the special shared-placement all-period method without using the generic
  independent-period loop;
- add ordered workspace case-batch and batch-all-period mirrors using existing
  `CaseBatchResult` failure isolation;
- resolve zones, scopes, canonical periods, weights, and isolated execution
  copies in the application layer and delegate once to Unit 2;
- retain only the detached result in a dedicated placement observation slot;
- expose already-computed placement metrics, summaries, comparisons, and
  reports without changing legacy `TargetOutput` schemas or triggering reruns;
- document specialist imports, both objectives, units, options, errors,
  direct/Total Site usage, multiperiod semantics, batches, and result
  interpretation;
- generate exactly one executable notebook,
  `19_utility_placement_optimisation.ipynb`, that uses public Python APIs to
  demonstrate the default thermodynamic and monetary/cogeneration workflows;
  register it in the tutorial manifest and notebook execution profile;
- verify root facade, optional dependencies, wheel/source distribution,
  backward compatibility, non-mutation, and repository-wide quality gates.

**Primary inputs**: Unit 1 public contracts and Unit 2 service; existing
application target and workspace accessors; presentation/reporting owners;
approved documentation and packaging conventions.

**Outputs**: public problem/all-period/batch workflows, explicit observation
and reporting behavior, the one required generated executable notebook, user
documentation, and end-to-end evidence.

**Owned requirements and stories**: FR-001, FR-007 public scope, FR-008 public
period selection, FR-013, FR-015, FR-016, FR-017, and final integration of
FR-002 and FR-014; NFR-002, NFR-004 through NFR-007; UPO-02, UPO-05 notebook
delivery, UPO-06, UPO-07, UPO-09, UPO-10, and UPO-12.

**Exclusions**:

- no utility-placement equations in application or presentation modules;
- no root-level export beyond `PinchProblem` and `PinchWorkspace`;
- no utility-placement CLI command or CLI integration;
- no additional utility-placement notebook beyond the single FR-017 artifact;
- no implicit application of the chosen placement to source utilities;
- no network, persistence, credential, deployment, or infrastructure service;
- no hidden target or optimisation execution during observation.

**Construction readiness**: comprehensive Functional Design defines accessor,
cache, batch, and reporting behavior; standard NFR stages capture compatibility
and observability; approved TDD Code Generation then integrates one public
surface at a time. PBT-01 must identify batch-order/state, copy invariance,
serialization-through-public-workflow, and repeatability properties.

**Exit evidence**: focused accessor/all-period/batch/report tests; success and
failure non-mutation properties; ordered batch isolation; installed-package
specialist import and root-facade tests; clean execution of the generated
thermodynamic and monetary/cogeneration notebook; tutorial-manifest and
wheel/source package-data checks; full fixed-seed non-solver, architecture,
Ruff, packaging, and distribution gates.

### Unit Boundary Validation

- Unit 1 owns stable values and pure transformations; Unit 2 consumes them and
  owns numerical orchestration; Unit 3 consumes both and owns user integration.
- No dependency points from Unit 1 to Unit 2/3 or from Unit 2 to Unit 3.
- Thermodynamic and monetary objectives share Unit 2 orchestration but retain
  independent pure evaluators.
- Every unit remains inside the existing package and preserves established
  component ownership.
- Infrastructure Design remains skipped for every unit.
