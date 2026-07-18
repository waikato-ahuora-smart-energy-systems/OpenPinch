# Current Documentation and Drift Guards Code Generation Plan

This document is the single source of truth for Unit 3 Code Generation. Steps
execute in numerical order, and checkboxes are marked in the same interaction
in which work completes.

## Unit Context

- **Unit**: Current Documentation and Drift Guards.
- **Project type**: Brownfield Python package; update current AI-DLC documents
  and existing repository contract tests.
- **Workspace root**: `/Users/timothyw/Github_Local/OpenPinch`.
- **Requirements**: FR-6, maintainability and no-new-dependency NFRs,
  acceptance criteria 7 and the Unit 3 portion of 8.
- **User stories**: N/A; bounded correctness requirements provide traceability.
- **Dependencies**: completed Unit 1 and Unit 2 contracts.
- **Public contract**: package root exports exactly `PinchProblem` and
  `PinchWorkspace`; concrete advanced types remain available from owner modules.
- **Historical boundary**: audit records and explicitly historical construction
  records remain intact; active state and current reverse-engineering material
  must describe only the canonical contract.
- **Database, frontend, infrastructure, and deployment entities**: none.

## Existing Files to Modify

- `aidlc-docs/aidlc-state.md`
- `aidlc-docs/inception/reverse-engineering/business-overview.md`
- `aidlc-docs/inception/reverse-engineering/architecture.md`
- `aidlc-docs/inception/reverse-engineering/code-structure.md`
- `aidlc-docs/inception/reverse-engineering/api-documentation.md`
- `aidlc-docs/inception/reverse-engineering/component-inventory.md`
- `aidlc-docs/inception/reverse-engineering/dependencies.md`
- `tests/packaging/test_docs_consistency.py`

## Generation Steps

### Step 1: Confirm current documentation baseline

- [x] Re-read the active state, reverse-engineering artifacts, root exports, and
  existing documentation-contract tests.
- [x] Run the focused documentation baseline.
- [x] Inventory stale current-contract names and malformed generated entries.

**Step 1 evidence**: the existing documentation/root-API selection passed 27
tests, demonstrating that current drift was unguarded. The inventory found six
reverse-engineering documents with retired package/service names, callable
`target()`, variant/selector workflows, and four captured shell-error entries;
the active state block also described an obsolete contract.

### Step 2: Add scoped drift regression first

- [x] Define the closed current reverse-engineering document set.
- [x] Reject retired package/service claims and shell-error contamination there.
- [x] Inspect only the active `Current Status` block in state so historical audit
  records remain legitimate.
- [x] Run the new guard and record its expected pre-fix failure.

**Step 2 evidence**: the new guard failed before documentation changes on the
retired facade/package claims and malformed shell-output inventory entries. Its
scope deliberately excludes audit and named historical sections.

### Step 3: Refresh active state and reverse-engineering documents

- [x] Make the active state describe the repository-remediation workflow.
- [x] Document root-only workflow entry points and concrete owner modules.
- [x] Correct architecture, code structure, API, component, and dependency
  descriptions without compatibility or retired-package guidance.
- [x] Preserve audit and explicitly historical records.

**Step 3 evidence**: the active status now describes this remediation; six
current reverse-engineering artifacts describe root workflow entry points and
the application/domain/contracts/analysis/optimisation/adapters/presentation
owners. Historical state sections and the append-only audit remain preserved.

### Step 4: Validate content and focused contracts

- [x] Validate Mermaid syntax and provide text alternatives.
- [x] Run the scoped stale-symbol guard and existing documentation consistency
  suite.
- [x] Run architecture and root API boundary checks.

**Step 4 evidence**: five Mermaid diagrams passed the required syntax/fence/tab
validation and each has a text alternative. The combined documentation,
architecture, and root-entrypoint suite passed 70 tests.

### Step 5: Run style and structural review

- [x] Run Ruff on the modified test.
- [x] Run patch hygiene and scan current artifacts for retired claims.
- [x] Confirm no runtime code, root export, compatibility page, or dependency was
  added by Unit 3.

**Step 5 evidence**: Ruff lint/format and `git diff --check` passed. The scoped
stale-symbol scan returned no matches. Unit 3 changed only AI-DLC documentation
and one existing documentation-contract test.

### Step 6: Generate implementation summary and enter Build and Test

- [x] Create the Unit 3 implementation summary.
- [x] Record requirement completion and focused evidence.
- [x] Mark Unit 3 complete and begin Build and Test under the user's completion
  authorization.

**Step 6 evidence**: implementation summary, state, audit record, and Build and
Test handoff completed after all Unit 3 checks passed.

## Build and Test Boundary

A clean warning-free Sphinx build, complete fixed-seed non-solver suite,
distribution build, isolated wheel smoke, and final repository checks execute
in the following Build and Test stage.
