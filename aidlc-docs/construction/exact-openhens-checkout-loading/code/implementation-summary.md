# Unit 2 Implementation Summary

## Outcome

Exact OpenHENS Checkout Loading is implemented and focused-verified. The
comparison utility now proves that every required OpenHENS module and the model
factory originate from the requested checkout, independent of ambient import
state.

## Modified Files

- `scripts/compare_openhens_openpinch_top5.py`
  - added a scoped exact-checkout loader;
  - isolates and restores all `openhens` cache entries and the complete import
    path sequence;
  - validates required module source origins and callable capabilities;
  - injects the verified factory into a separate source-execution helper.
- `tests/packaging/test_openhens_comparison_prerequisite.py`
  - covers cached foreign modules, foreign origins, partial import failure,
    exact restoration, verified-factory handoff, and factory-only execution.
- `OpenPinch/application/workspace.py`
  - corrected the Unit 1 containment implementation discovered by the broader
    architecture gate so the application layer no longer imports `pathlib`.

## Requirement Completion

- [x] FR-3 exact OpenHENS checkout identity.
- [x] Deterministic checkout identity, maintainability, no-new-dependency, and
  bounded-overhead NFRs.
- [x] Unit 2 portions of acceptance criteria 1 and 4.

## Verification Evidence

- Focused baseline: 3 passed.
- Regression-first checkpoint: 5 expected pre-fix failures, 3 controls passed.
- Exact-checkout prerequisite suite: 8 passed.
- Combined prerequisite, API-boundary, dependency-rule, repository-entrypoint,
  and workspace selection: 123 passed with seed `20260715`.
- Ruff lint and format checks: passed.
- `git diff --check`: passed.

## Structural Review

- No ambient OpenHENS import remains in model execution.
- No fallback to installed or cached modules.
- No upstream mutation, compatibility shim, new dependency, root export,
  duplicate file, solver change, ranking change, or output-format change.
- Interpreter state is restored on tested success and failure paths.

## Deferred Gates

The complete fixed-seed non-solver suite, clean warning-free Sphinx build,
distribution build, isolated-wheel smoke, and final stale-symbol checks execute
after Unit 3.
