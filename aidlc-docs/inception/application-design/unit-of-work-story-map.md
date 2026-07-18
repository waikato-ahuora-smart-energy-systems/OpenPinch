# Requirement-to-Unit Map

| Requirement group | Unit |
|---|---|
| FR-01 through FR-10, FR-16, FR-17 | Domain and Input |
| FR-11, FR-12 | Targeting and Integration |
| FR-13 through FR-15 | Heat Exchanger Network |
| NFR-01 through NFR-04, NFR-07 | Domain and Input |
| NFR-02, NFR-06 | Targeting and Integration |
| NFR-05 through NFR-07 | Heat Exchanger Network |

## Package Usability Refactor Story and Requirement Map

| Story or requirement | Primary unit | Supporting units |
|---|---|---|
| US-1 First solve | 2 | 1, 4, 5 |
| US-2 Total Site workflow | 2 | 1, 4, 5 |
| US-3 Scenario comparison | 3 | 1, 4, 5 |
| US-4 Advanced thermal workflows | 2 | 1, 3, 4, 5 |
| US-5 HEN selection and visualization | 3 | 1, 4, 5 |
| US-8 Predictable interaction | 2 | 1, 3, 4, 5 |
| FR-1 Canonical package boundary | 2 | 1, 4, 5 |
| FR-2 Explicit targeting vocabulary | 2 | 1, 4, 5 |
| FR-3 Friendly workflow configuration | 2 | 1, 3, 4, 5 |
| FR-4 Intuitive workspace scenarios | 3 | 1, 4, 5 |
| FR-5 Public result operations | 2 | 1, 3, 4, 5 |
| FR-6 Public HEN design experience | 3 | 1, 4, 5 |
| FR-7 Tutorial redesign | 4 | 1, 2, 3, 5 |
| FR-8 Executable tutorial contracts | 5 | 1, 4 |
| FR-9 Complete PinchProblem interaction contract | 2 | 1, 3, 4, 5 |
| Acceptance 1 notebook execution | 4 | 5 |
| Acceptance 2 public import boundary | 4 | 5 |
| Acceptance 3 retired syntax and private-helper removal | 4 | 5 |
| Acceptance 4 weighted HPR aggregation | 1 | 2, 4, 5 |
| Acceptance 5 root-only quickstart | 4 | 2, 5 |
| Acceptance 6 lint/docs/architecture/non-solver gates | 5 | 1, 2, 3, 4 |
| Acceptance 7 selector removal | 2 | 1, 3, 5 |
| Acceptance 8 complete problem interaction classification | 1 | 2, 3, 5 |
| Acceptance 9 tutorial manifest completeness | 4 | 1, 5 |
| Acceptance 10 multiperiod study templates | 4 | 2, 3, 5 |
| Acceptance 11 signature and stale-symbol guards | 5 | 1, 2, 3 |
| Acceptance 11 multi-segment example | 4 | 2, 5 |
| Acceptance 12 live inventory and RTD coverage parity | 5 | 1, 4 |
| Acceptance 13 retiring workspace methods absent | 3 | 1, 5 |
| Acceptance 14 100 percent executable coverage | 4 | 5 |
| Acceptance 15 warning-free generated RTD coverage | 5 | 4 |

Every story, FR section, and numbered acceptance criterion is assigned. Unit 1
owns contract evidence, Units 2 and 3 own runtime behavior, Unit 4 owns teaching
coverage, and Unit 5 owns public documentation and enforcement. There are no
orphan stories and no story is assigned only to a downstream consumer without
an implementation owner.

## Repository Issue Remediation Requirement Map

User Stories are intentionally skipped for these bounded correctness fixes.
Functional requirements, NFRs, and acceptance criteria provide full assignment
traceability instead.

| Requirement | Primary unit | Supporting unit |
|---|---|---|
| FR-1 Workspace identifiers and export containment | 1 | 3 |
| FR-2 Detached problem-input observation | 1 | 3 |
| FR-3 Exact OpenHENS checkout identity | 2 | 3 |
| FR-4 Collision-free workbook allocation | 1 | 3 |
| FR-5 Consistent unloaded-problem error | 1 | 3 |
| FR-6 Current contract documentation | 3 | 1 and 2 |
| NFR Safety and filesystem containment | 1 | 3 |
| NFR Reliability and concurrent allocation | 1 | 3 |
| NFR State consistency | 1 | 3 |
| NFR Deterministic checkout identity | 2 | 3 |
| NFR Cross-platform portability | 1 | 3 |
| NFR Maintainability and shared validation | 1 | 3 |
| NFR No new runtime dependencies | 1 and 2 | 3 |
| NFR Bounded performance overhead | 1 | 3 |

| Acceptance criterion | Primary unit | Final verification |
|---|---|---|
| 1. Six reproductions have focused regressions | 1 and 2 | 3 |
| 2. Unsafe identifiers rejected and exports contained | 1 | 3 |
| 3. Snapshot mutation cannot affect internal state | 1 | 3 |
| 4. OpenHENS modules originate from requested checkout | 2 | 3 |
| 5. Repeated/concurrent exports never collide | 1 | 3 |
| 6. Empty multiplier update raises canonical error | 1 | 3 |
| 7. Current docs contain no retired API claims | 3 | 3 |
| 8. Full quality and distribution gates pass | 3 | 3 |

Every FR, NFR, and numbered acceptance criterion is assigned to an implementation
owner and a final verification owner. Unit 3 cannot redefine Unit 1 or Unit 2
behavior; it documents and enforces their completed contracts.
