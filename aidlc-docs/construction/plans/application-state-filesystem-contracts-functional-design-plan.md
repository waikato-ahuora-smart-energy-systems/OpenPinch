# Application State and Filesystem Contracts Functional Design Plan

## Context Resolution

The approved requirements, Application Design, and Unit 1 definition resolve all
functional choices. No unanswered `[Answer]:` tags are required.

- **Business logic modeling**: strict rejection at input boundaries, detached
  observation, guarded mutation, contained export, and exclusive allocation.
- **Domain model**: existing `PinchProblem`, `PinchWorkspace`, workspace bundle,
  `CaseBatchResult`, and reporting path types remain authoritative.
- **Business rules**: clean break, no aliases or silent case-name sanitization,
  no raw case identifiers used without validation, and no mutable input leaks.
- **Data flow**: inputs are validated before storage; observation is copied on
  return; exports reserve paths before writing.
- **Integration points**: local filesystem and pandas/openpyxl only; no network,
  database, or service integration.
- **Error handling**: actionable validation/runtime errors and guaranteed failed-
  reservation cleanup.
- **Business scenarios**: runtime/bundle creation, batch export, snapshot
  mutation, lazy root rebuilding, repeated export, concurrent export, and writer
  failure are covered.
- **Frontend components**: N/A; OpenPinch is an in-process Python package.

## Generation Checklist

- [x] Analyze Unit 1 responsibilities, FR assignments, NFRs, and exclusions.
- [x] Define the end-to-end business logic model.
- [x] Define exact case-identifier rules and boundary coverage.
- [x] Define batch path containment and symlink-aware resolution behavior.
- [x] Define detached input observation and mutation ownership.
- [x] Define prepared-root multiplier behavior and cache invalidation.
- [x] Define exclusive workbook allocation and failure cleanup.
- [x] Generate `business-logic-model.md`.
- [x] Generate `business-rules.md`.
- [x] Generate `domain-entities.md`.
- [x] Validate completeness against FR-1, FR-2, FR-4, FR-5 and assigned NFRs.
- [x] Confirm no frontend artifact is applicable.

