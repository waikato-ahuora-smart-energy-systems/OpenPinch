# Application Design Plan

- [x] Define components and responsibilities.
- [x] Define public and internal methods.
- [x] Define service orchestration changes.
- [x] Define dependencies and data flow.
- [x] Validate design consistency against the approved plan.

Decision questions were answered by the approved implementation plan; no blank answers remain.

## Package Usability Refactor Application Design

- [x] Reconcile the approved usability requirements, stories, coverage map,
  workflow argument map, and live application architecture.
- [x] Preserve completed segmented-stream and architecture-modernization design
  records while defining a separate usability-refactor design.
- [x] Generate component definitions and high-level responsibilities in
  `components.md` and the namespaced consolidated design.
- [x] Generate public method signatures and return contracts in
  `component-methods.md` and the namespaced consolidated design.
- [x] Generate service definitions and orchestration patterns in `services.md`
  and the namespaced consolidated design.
- [x] Generate dependency relationships, communication patterns, a validated
  Mermaid flow, and its text alternative in `component-dependency.md` and the
  namespaced consolidated design.
- [x] Validate design completeness against all five usability user stories,
  eighteen tutorial owners, the RTD coverage contract, clean-break policy, and
  no-hidden-execution rules.

The approved workflow plan resolves component boundaries, public vocabulary,
service ownership, dependency direction, tutorial allocation, and compatibility
policy. No additional design question would change the application boundary;
there are no blank `[Answer]:` tags.

## Repository Issue Remediation Application Design

- [x] Reconcile the six reproduced findings with the approved requirements and
  execution plan.
- [x] Correct the workflow prerequisite by executing minimal Application Design
  before Units Generation.
- [x] Define the existing component owners for workspace identity, problem-state
  observation, workbook allocation, exact OpenHENS loading, and documentation
  drift protection in `components.md`.
- [x] Define internal method signatures and unchanged public return contracts in
  `component-methods.md`.
- [x] Define orchestration and failure boundaries in `services.md`.
- [x] Define dependency direction, a dependency matrix, a validated Mermaid
  flow, and its text alternative in `component-dependency.md`.
- [x] Consolidate the design decision in `application-design.md`.
- [x] Validate the design against the clean-break policy, cross-platform path
  behavior, exact module identity, atomic file allocation, and no-new-runtime-
  dependency constraint.

The approved remediation requirements determine all component boundaries and
interfaces. No unresolved choice would change the design, so no blank
`[Answer]:` tags are required.

## Utility Placement Optimisation Application Design

- [x] Reconcile the approved utility-placement requirements, personas, 12 user
  stories, reverse-engineered architecture, and approved execution plan.
- [x] Define components, responsibilities, ownership boundaries, and public or
  specialist interfaces in `components.md`.
- [x] Define public and internal method signatures, typed inputs and outputs,
  and high-level method purposes in `component-methods.md`.
- [x] Define application, analysis, targeting, optimisation, cogeneration,
  reporting, all-period, and workspace-batch orchestration in `services.md`.
- [x] Define dependency direction, communication patterns, a dependency matrix,
  and a validated data-flow diagram with a text alternative in
  `component-dependency.md`.
- [x] Consolidate the design and considered alternatives in
  `application-design.md`.
- [x] Validate completeness and consistency against FR-001 through FR-017,
  NFR-001 through NFR-007, UPO-01 through UPO-12, the TDD mandate, and enabled
  Property-Based Testing constraints.
- [x] Incorporate the approved scope amendment: no CLI integration and exactly
  one generated executable notebook covering thermodynamic and
  monetary/cogeneration workflows.
- [x] Obtain explicit user approval before Units Generation.

The approved artifacts settle the design categories that could otherwise need
questions: component organization follows existing layer ownership; the public
method uses explicit keyword inputs and normalizes an immutable request;
application accessors resolve scopes and periods while analysis services own
equations; existing targeting, optimisation, and cogeneration services are
composed through their public boundaries; and detached placement results are
returned and retained only in a dedicated observation cache for reporting.
The existing notebook generator, tutorial manifest, execution profiles, and
package-data gates own the single example artifact; no CLI owner is involved.
Consequently, no unresolved design question would change a component boundary
or interface and there are no blank `[Answer]:` tags.
