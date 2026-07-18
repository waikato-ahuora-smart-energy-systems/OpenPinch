# Unit 1 Implementation Summary

## Outcome

Application State and Filesystem Contracts is implemented and focused-
verified. The unit closes four confirmed behavioral findings without adding
dependencies, compatibility paths, root exports, solver changes, or schema
migrations.

## Modified Application Files

- `OpenPinch/contracts/workspace.py`
  - added one strict portable workspace case-name validator;
  - applied it to workspace bundle `baseline_name` and all case keys;
  - retained workspace schema version `3`.
- `OpenPinch/application/workspace.py`
  - validates runtime case-creation/storage boundaries;
  - resolves and enforces batch export-root containment without introducing a
    concrete filesystem dependency forbidden by the application architecture;
  - preserves original case keys and batch failure isolation.
- `OpenPinch/application/problem.py`
  - returns deep detached `problem_data` snapshots;
  - uses the prepared-root guard for multiplier mutation;
  - preserves existing warning, rebuilding, and cache invalidation behavior.
- `OpenPinch/presentation/reporting/workbook.py`
  - reserves workbook paths atomically with exclusive creation;
  - uses readable microsecond timestamps and numeric collision suffixes;
  - removes reserved/partial artifacts on unsuccessful writes;
  - preserves successful workbook contents and return type.

## Modified Test Files

- `tests/application/test_pinch_problem.py`
- `tests/application/test_pinch_workspace.py`
- `tests/presentation/test_workbook_reporting.py`

Coverage added for mapping and `TargetInput` snapshots, workspace delegation,
unloaded and lazy multiplier paths, accepted/rejected case identifiers, bundle
keys, fixed-seed generated identifiers, corrupted private state, symlink
containment, repeated/concurrent allocation, writer failure cleanup, and
successful workbook readability.

## Requirement Completion

- [x] FR-1 workspace identity and export containment.
- [x] FR-2 detached problem-input observation.
- [x] FR-4 collision-free workbook allocation.
- [x] FR-5 consistent unloaded-problem error.
- [x] Assigned safety, reliability, state consistency, portability,
  maintainability, dependency, and bounded-overhead NFRs.
- [x] Unit 1 portions of acceptance criteria 1, 2, 3, 5, and 6.

## Verification Evidence

- Pre-change focused baseline: 128 passed.
- Problem-state regressions: 5 expected failures confirmed before fixes.
- Identity/containment regressions: 51 expected failures and 7 accepted controls
  before fixes with Hypothesis seed `20260715`.
- Workbook regressions: 4 expected failures confirmed before fixes.
- Bundle/accepted identifier checkpoint: 30 passed.
- Runtime/property/containment checkpoint: 58 passed.
- Snapshot/multiplier checkpoint: 11 passed.
- Workbook reporting module: 14 passed.
- Integrated application, workspace, usability, and reporting suite: 203 passed
  with Hypothesis seed `20260715`.
- HEN serialization, general round-trip, and root API boundary tests: 34 passed.
- Ruff lint: passed.
- Ruff format check: passed.
- `git diff --check`: passed.

## Structural Review

- No duplicate `*_new.py` or `*_modified.py` files.
- No compatibility alias or sanitizing fallback.
- No schema migration or new dependency.
- No package-root export change.
- No database, frontend, deployment, solver, or HEN behavior change.
- Unit 2 comparison tooling and Unit 3 current-documentation sources remain
  outside this implementation unit.

## Deferred Gates

The complete fixed-seed non-solver suite, clean warning-free Sphinx build,
distribution build, isolated-wheel smoke, and final stale-symbol scans execute
in Build and Test after Units 2 and 3 complete.
