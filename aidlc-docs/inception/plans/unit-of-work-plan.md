# Unit of Work Plan

- [x] Define the domain-input unit.
- [x] Define the targeting-integration unit.
- [x] Define the HEN unit.
- [x] Document dependencies and implementation order.
- [x] Map requirements to units.
- [x] Validate every requirement is assigned.

The approved implementation plan fixes the grouping, dependency, deployment, and business-boundary decisions; no unanswered decomposition questions remain.

## Package Usability Refactor Unit Plan

### Decomposition Assessment

- **Story grouping**: five units follow the approved execution plan and group
  contract foundations, problem targeting, workspace/design, tutorials, and
  documentation/verification by change affinity.
- **Dependencies**: units execute in numerical order. Later units consume the
  public contracts and behavior established by earlier units.
- **Team alignment**: OpenPinch is one in-process Python package for one
  process-engineer persona; units are planning boundaries, not separate team or
  ownership silos.
- **Technical considerations**: all units ship in one wheel and source
  distribution. Optional HPR and HEN profiles remain test/runtime profiles,
  not deployable services.
- **Business domain**: the decomposition preserves one heat-integration domain
  and separates learning/publication work only because it consumes the public
  contract and has distinct executable quality gates.
- **Code organization**: brownfield; existing `application`, `analysis`,
  `domain`, `contracts`, `presentation`, `data/notebooks`, `docs`, and `tests`
  ownership is retained.

### Generation Steps

- [x] Generate the five unit definitions and responsibilities in the
  namespaced usability section of `unit-of-work.md`.
- [x] Generate the unit dependency matrix and implementation order in the
  namespaced usability section of `unit-of-work-dependency.md`.
- [x] Map US-1 through US-5 and US-8, plus tutorial/RTD acceptance, to units in
  the namespaced usability section of `unit-of-work-story-map.md`.
- [x] Validate that every approved requirement, story, tutorial owner, and
  verification gate is assigned to at least one unit.
- [x] Validate that dependencies are acyclic and no later unit defines a public
  contract consumed by an earlier unit.

### Question 1

How should Units Generation proceed with the five-unit decomposition already
defined by the approved execution plan?

A) Approve the unit plan and generate the five unit artifacts

B) Request changes to the unit boundaries (describe them after the `[Answer]:`
tag)

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A

## TESPy HPR Performance-Map Unit Plan

### Decomposition Assessment

- **Story grouping**: user stories were intentionally skipped because the
  approved technical requirements already contain acceptance criteria. The
  work divides cleanly into the transport contract, thermodynamic map
  generation, and current-target API integration.
- **Dependencies**: contract fields and fixtures must stabilize before either
  simulator emits maps; simulators and the generation service must stabilize
  before the public target accessor exposes them. Documentation and tests stay
  with their owning unit, followed by one integrated gate.
- **Team alignment**: OpenPinch is one in-process Python package. Units are
  sequential implementation, TDD, and review boundaries rather than separate
  teams or independently owned services.
- **Technical considerations**: all units ship in one distribution. CoolProp
  remains the default installed path; TESPy is isolated behind an optional
  extra and leaf import. OpenUtility, Pyomo, and HiGHS are not dependencies.
- **Business domain**: all three units serve one HPR targeting and physical-map
  generation capability. OpenUtility's investment and dispatch optimization
  remains outside the domain boundary.
- **Code organization**: this is brownfield work. Existing `contracts`, HPR
  target/service, optional-adapter, application accessor, documentation, test,
  and package-data owners remain authoritative; no new deployable service or
  top-level package is introduced.

### Proposed Units

1. **HPR Performance-Map Contract and Golden Fixtures**
   - Strict Pydantic request, units, point, map, JSON-value, and typed-error
     contracts.
   - Exact alpha schema `1.0` fields matching OpenUtility, including
     `energy_balance_tolerance` and `temperature_match_tolerance`.
   - Canonical JSON Schema plus heat-pump and refrigeration golden fixtures.
   - Deterministic serialization, invariant, property, cold-import, and package
     content tests without TESPy or OpenUtility.
2. **HPR Point Simulators and Map Generation**
   - Engine-neutral context, operating-point/result values, simulator protocol,
     deterministic grid driver, lifecycle, and all-or-nothing failure behavior.
   - CoolProp-backed simulator as the default path tied to current
     vapour-compression HPR target semantics.
   - Explicit TESPy simulator leaf, optional dependency extra, design/offdesign
     policy, convergence diagnostics, and fake-adapter/optional smoke tests.
3. **Current HPR Target API Integration and Publication**
   - `simulation_backend="coolprop"` on current vapour-compression heat-pump
     and refrigeration target methods, with explicit `"tespy"` selection and
     unchanged default call behavior.
   - Normalized backend replay intent on target results and explicit
     `problem.target.hpr_performance_map(target=..., request=...)` generation.
   - Compatibility rejection for unsupported and multi-port targets, public
     documentation/examples, architecture checks, cross-package golden-fixture
     verification instructions, and final distribution gates.

### Generation Steps

- [x] Reconcile the approved requirements and Application Design with the
  corrected OpenUtility alpha consumer contract.
- [x] Evaluate story grouping, dependencies, team alignment, technical and
  deployment concerns, business-domain boundaries, and brownfield code
  organization.
- [x] Define the proposed three-unit contract-first decomposition and its
  implementation order.
- [x] Collect and analyze the user's unit-plan answer; add follow-up questions
  if it is ambiguous.
- [x] Obtain explicit approval before generating unit artifacts.
- [x] Add the three scoped unit definitions, responsibilities, inputs, outputs,
  exclusions, and exit evidence to
  `aidlc-docs/inception/application-design/unit-of-work.md`.
- [x] Add the dependency matrix, critical path, coordination points, rollback
  boundaries, and test checkpoints to
  `aidlc-docs/inception/application-design/unit-of-work-dependency.md`.
- [x] Map every TESPy HPR requirement, acceptance criterion, and enabled
  property-testing obligation in
  `aidlc-docs/inception/application-design/unit-of-work-story-map.md`.
- [x] Validate acyclic dependencies, complete traceability, optional-dependency
  isolation, and absence of OpenUtility/Pyomo/HiGHS ownership leakage.
- [x] Obtain explicit approval of generated units before Construction.

### Unit Plan Approval Question

How should Units Generation proceed with this contract-first three-unit
decomposition?

A) Approve the proposed units and generate the three unit artifacts

B) Request changes to the unit boundaries and describe them after the
`[Answer]:` tag

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A (approved by explicit chat response `A`)

## Utility Placement Optimisation Unit Plan

### Decomposition Assessment

- **Story grouping**: The approved execution plan proposes three cohesive
  units: placement contracts and pure model; placement evaluation and
  optimisation; and public workflow integration. This groups high-affinity
  stories without splitting the two objective modes into duplicated pipelines.
- **Dependencies**: The default critical path is Unit 1 -> Unit 2 -> Unit 3.
  Public integration consumes stable contracts and the completed specialist
  service. Test fixtures and documentation can develop alongside later slices
  after their consumed interfaces stabilize.
- **Team alignment**: OpenPinch is one in-process Python package. In the absence
  of a separate team topology, units are implementation, TDD, and review
  boundaries rather than ownership silos.
- **Technical considerations**: All units ship in the same wheel and source
  distribution, reuse existing optional solver profiles, add no runtime
  dependency, and introduce no separately deployable process or service.
- **Business domain**: One utility-placement capability owns both objective
  modes. The split follows stable contracts, core numerical capability, and
  user-facing integration while preserving one domain vocabulary.
- **Code organization**: N/A as a question because this is brownfield. Existing
  `contracts`, `domain`, `analysis`, `optimisation`, `application`,
  `presentation`, `tests`, and documentation owners remain authoritative.

### Proposed Units

1. **Unit 1 - Placement Contracts and Pure Model**
   - Specialist enums, request/template/options/results, diagnostics, and error
     contracts.
   - Template normalization, count validation, bound derivation, deterministic
     starting points, and vector encoding/decoding.
   - Serialization, units, ordering, and pure-model invariants.
2. **Unit 2 - Placement Evaluation and Optimisation Service**
   - Detached direct and Total Site period contexts and duty allocation.
   - Per-period coverage, entropy/exergy, monetary/cogeneration evaluation,
     aggregation, feasibility, optimiser coordination, and alternatives.
   - Typed numerical failures, deterministic bounded solve, and analytical or
     structured-grid oracle tests.
3. **Unit 3 - Public Workflow and Presentation Integration**
   - Problem target, explicit all-period, and ordered workspace-batch surfaces.
   - Dedicated detached result observation, metrics, summaries, comparisons,
     reports, public docs, exactly one executable generated notebook covering
     thermodynamic and monetary/cogeneration workflows, compatibility, and
     end-to-end gates. CLI integration is excluded.

### Generation Steps

- [x] Reconcile the approved requirements, 12 stories, Application Design, and
  three-unit execution-plan hypothesis.
- [x] Collect and analyze every decomposition answer below; add follow-up
  questions for any ambiguity.
- [x] Obtain explicit approval of the completed unit plan before generation.
- [x] Generate the Utility Placement section of
  `aidlc-docs/inception/application-design/unit-of-work.md` with unit
  definitions, responsibilities, inputs, outputs, exclusions, and readiness.
- [x] Generate the Utility Placement section of
  `aidlc-docs/inception/application-design/unit-of-work-dependency.md` with the
  dependency matrix, critical path, coordination points, and test checkpoints.
- [x] Generate the Utility Placement section of
  `aidlc-docs/inception/application-design/unit-of-work-story-map.md` mapping
  every UPO story, FR, NFR, and enabled PBT obligation to one or more units.
- [x] Validate that dependencies are acyclic, existing package ownership is
  preserved, and no later unit defines a contract required by an earlier unit.
- [x] Validate that all UPO-01 through UPO-12, FR-001 through FR-017, NFR-001
  through NFR-007, TDD acceptance work, and PBT obligations are assigned.
- [x] Assign the approved no-CLI and single executable notebook scope amendment
  to Unit 3 with manifest, execution-profile, and package-data verification.
- [x] Obtain explicit approval of generated units before Construction.

### Question 1 - Story Grouping

Which story-grouping strategy should define the units?

A) Use the proposed three-unit split: contracts/pure model, evaluation/service,
then public integration

B) Use a different grouping and describe the desired unit boundaries after the
`[Answer]:` tag

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A (accepted by explicit chat response `Go`)

### Question 2 - Dependencies

How should inter-unit delivery be sequenced?

A) Use the dependency-ordered critical path Unit 1 -> Unit 2 -> Unit 3, with
only non-blocking fixtures and documentation prepared in parallel

B) Optimize for broader parallel delivery even if temporary interfaces or
integration rework are required

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A (accepted by explicit chat response `Go`)

### Question 3 - Team Alignment

What ownership model should the decomposition assume?

A) Treat units as sequential TDD and review boundaries for one package team

B) Align units to separate teams or maintainers and describe the ownership
split after the `[Answer]:` tag

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A (accepted by explicit chat response `Go`)

### Question 4 - Technical Deployment Boundary

What does "analysis service" mean for deployment and packaging?

A) Keep it as an in-process OpenPinch analysis service shipped in the existing
package, with no network or independent deployment boundary

B) Introduce a separately deployable or remotely callable service boundary

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A (accepted by explicit chat response `Go`)

### Question 5 - Business Domain Boundary

How should the two cost objectives affect unit boundaries?

A) Keep thermodynamic and monetary objectives in one evaluation/service unit
with separate pure evaluators and a shared placement pipeline

B) Split thermodynamic and monetary placement into separate implementation
units despite their shared templates, constraints, and orchestration

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A (accepted by explicit chat response `Go`)

## Repository Issue Remediation Unit Plan

### Decomposition Assessment

- **Story grouping**: no new user stories are required. Requirements are grouped
  by shared state and side-effect boundary: application/filesystem contracts,
  exact external-checkout loading, and current documentation/drift protection.
- **Dependencies**: Unit 1 and Unit 2 have no runtime dependency on each other.
  They execute sequentially for diagnostic clarity. Unit 3 consumes their final
  contracts and therefore runs last.
- **Team alignment**: OpenPinch is one in-process Python package for one
  process-engineer persona. Units are implementation and review boundaries, not
  team or deployment boundaries.
- **Technical considerations**: all units ship in the same wheel/source
  distribution. No new dependency, service, process, or configuration channel
  is introduced.
- **Business domain**: all units preserve the existing heat-integration domain;
  the split follows technical ownership because the findings are correctness
  defects rather than new business capabilities.
- **Code organization**: N/A as a decomposition question because this is a
  brownfield package. Existing `application`, `contracts`, `presentation`,
  `scripts`, `tests`, and `aidlc-docs` ownership is retained.

### Unit Definitions

1. **Application State and Filesystem Contracts**
   - Workspace case-identifier validation and bundle enforcement.
   - Batch export path containment.
   - Detached problem-input observation.
   - Prepared-root multiplier guard.
   - Exclusive workbook allocation and failure cleanup.
2. **Exact OpenHENS Checkout Loading**
   - Import-cache isolation and requested-root precedence.
   - Required-capability and module-origin validation.
   - Verified factory injection and interpreter-state restoration.
3. **Current Documentation and Drift Guards**
   - Current AI-DLC state and reverse-engineering API correction.
   - Scoped retired-contract assertions.
   - Warning-free documentation and package verification.

### Generation Steps

- [x] Generate `aidlc-docs/inception/application-design/unit-of-work.md` with
  scoped unit definitions, responsibilities, inputs, outputs, and exclusions.
- [x] Generate `aidlc-docs/inception/application-design/unit-of-work-dependency.md`
  with the dependency matrix, critical path, coordination points, and testing
  checkpoints.
- [x] Generate `aidlc-docs/inception/application-design/unit-of-work-story-map.md`
  mapping FR-1 through FR-6, NFRs, and acceptance criteria to units in place of
  skipped user stories.
- [x] Validate unit boundaries against the approved Application Design.
- [x] Verify every requirement and acceptance criterion is assigned.
- [x] Verify dependencies are acyclic and no unit introduces a new public root
  export or deployable service.
- [x] Update plan checkboxes immediately as each artifact is generated.

### Unit Plan Approval Question

How should Units Generation proceed with this three-unit decomposition?

A) Approve the unit plan and generate all three unit artifacts

B) Request changes to the unit boundaries and describe them after the
`[Answer]:` tag

X) Other (please describe after the `[Answer]:` tag below)

[Answer]: A
