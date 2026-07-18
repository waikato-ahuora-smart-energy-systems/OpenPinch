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
