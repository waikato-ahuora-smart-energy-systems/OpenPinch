# Exact OpenHENS Checkout Loading Code Generation Plan

This document is the single source of truth for Unit 2 Code Generation. Steps
execute in numerical order, and each checkbox is marked in the same interaction
in which its work completes.

## Unit Context

- **Unit**: Exact OpenHENS Checkout Loading.
- **Project type**: Brownfield Python package; modify the existing comparison
  script and its prerequisite tests in place.
- **Workspace root**: `/Users/timothyw/Github_Local/OpenPinch`.
- **Requirements**: FR-3; deterministic checkout identity, maintainability, no
  new dependency, and bounded-overhead NFRs; acceptance criteria 1 and 4.
- **User stories**: N/A; bounded correctness requirements provide traceability.
- **Dependencies**: approved inception artifacts and Unit 2 Functional Design.
- **Downstream consumer**: Unit 3 current documentation and final gates.
- **Public interfaces**: none; root exports remain exactly `PinchProblem` and
  `PinchWorkspace`.
- **Database, frontend, infrastructure, and deployment entities**: none.

## Existing Files to Modify

- `scripts/compare_openhens_openpinch_top5.py`
- `tests/packaging/test_openhens_comparison_prerequisite.py`
- `aidlc-docs/aidlc-state.md`
- `aidlc-docs/audit.md`

## Generation Steps

### Step 1: Confirm focused baseline and ownership

- [x] Re-read the script and focused prerequisite tests.
- [x] Run the focused baseline before implementation.
- [x] Confirm no external checkout or solver execution is required.

**Step 1 evidence**: the three focused prerequisite tests passed in 8.24
seconds. They use only temporary paths and fake module objects; no external
checkout or solver is required.

### Step 2: Add exact-checkout regressions first

- [x] Prove a cached foreign module cannot satisfy requested imports.
- [x] Prove wrong or missing module origins fail closed.
- [x] Prove success and import/capability failure restore path and cache state.
- [x] Prove source execution receives the verified factory explicitly.
- [x] Run regressions and record expected pre-fix failures.

**Step 2 evidence**: five regressions failed before implementation because the
scoped loader and injected execution helper did not exist. The three original
prerequisite controls continued to pass.

### Step 3: Implement the scoped exact-checkout loader

- [x] Snapshot and isolate OpenHENS cache entries.
- [x] Prioritize the resolved requested checkout and invalidate import caches.
- [x] Import the closed module set and validate source origins and capabilities.
- [x] Restore exact caller state in `finally`.

**Step 3 evidence**: `_supported_openhens_checkout` now treats the required
module set and import path as a bounded transaction, rejects missing/foreign
origins, and restores the original path and cached object identities.

### Step 4: Inject the verified factory into execution

- [x] Split source model execution from scoped module acquisition.
- [x] Pass the verified `OpenHENS` callable explicitly.
- [x] Preserve model arguments, ranking, statistics, and failure-before-output.

**Step 4 evidence**: `_run_source_openhens` owns the verified import scope and
passes its factory to `_execute_source_openhens`; the execution helper contains
no OpenHENS import and retains the original model and result logic.

### Step 5: Run focused verification

- [x] Run the complete comparison prerequisite tests.
- [x] Run related packaging/architecture tests.
- [x] Run Ruff check, Ruff format check, and patch hygiene.

**Step 5 evidence**: all 8 prerequisite tests passed. The combined prerequisite,
API-boundary, dependency-rule, repository-entrypoint, and workspace regression
selection passed 123 tests with Hypothesis seed `20260715`. Ruff lint/format and
patch hygiene passed.

### Step 6: Perform structural review

- [x] Confirm no ambient OpenHENS import remains in source execution.
- [x] Confirm no upstream mutation, fallback, dependency, root export, duplicate
  file, or solver/ranking behavior change was introduced.
- [x] Confirm exact restoration on all tested exit paths.

**Step 6 evidence**: structural inspection found only the planned script and test
owners. No production import of OpenHENS exists outside the scoped loader, and
all state-restoration assertions pass.

### Step 7: Generate implementation summary and handoff

- [x] Create the Unit 2 implementation summary.
- [x] Record fulfilled requirements and focused verification.
- [x] Mark Unit 2 complete and continue to Unit 3 under the user's completion
  authorization.

**Step 7 evidence**: the implementation summary, state, audit record, and Unit 3
handoff were completed after all focused checks passed.

## Deferred to Build and Test

The complete fixed-seed non-solver suite, clean warning-free Sphinx build,
distribution build, isolated wheel smoke, and final stale-symbol checks execute
after Unit 3. No real external OpenHENS solve is required by this unit.
