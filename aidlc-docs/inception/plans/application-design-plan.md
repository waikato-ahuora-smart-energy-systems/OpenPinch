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
