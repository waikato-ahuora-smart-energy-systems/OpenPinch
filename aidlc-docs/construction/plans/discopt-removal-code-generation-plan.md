# Experimental Discopt Integration Removal Code Generation Plan

## Status

- **Stage**: Code Generation, Part 1 - Planning
- **Approval**: Approved on 2026-07-15
- **Single source of truth**: This checklist controls the removal implementation.
- **Workspace root**: `/Users/timothyw/Github_Local/OpenPinch`
- **Application code location**: Existing package, script, test, and developer
  documentation paths at the workspace root; never under `aidlc-docs/`.

## Unit Context

- **Unit**: Experimental Discopt integration removal.
- **User stories**: None; the approved workflow skips user stories for this
  internal experimental-code removal.
- **Requirements**: FR-01 through FR-08 and NFR-01 through NFR-06 in
  `aidlc-docs/inception/requirements/discopt-removal-requirements.md`.
- **Dependencies**: Existing HEN solver backend, GEKKO/Pyomo conversion boundary,
  unit-model solver annotations, benchmark infrastructure, and Sphinx docs.
- **Preserved interfaces**: Couenne, `ipopt-pyomo`, `ipopt-GEKKO`, and APOPT.
- **Removed interface**: The private `discopt` solver selection and bridge.
- **Database entities**: None.
- **Deployment artifacts**: None.

## Step 1 - Establish the Surgical Removal Baseline

- [x] Re-read the approved requirements and workflow plan.
- [x] Capture the current Git status and exact Discopt references in package
  code, scripts, tests, developer docs, package metadata, and lockfile.
- [x] Distinguish unrelated heat-pump, Read the Docs, stream-segmentation, and
  AI-DLC changes from the Discopt experiment.
- [x] Confirm ignored benchmark results and historical AI-DLC artifacts are
  preservation-only evidence and outside the executable removal set.

## Step 2 - Restore the Supported HEN Solver Backend

- [x] Remove the private `_discopt` import and restore the supported Pyomo solver
  set in
  `OpenPinch/services/heat_exchanger_network_synthesis/common/solver/backend.py`.
- [x] Remove Discopt dependency checks, configuration branches, execution
  branches, and optional solver-result extraction.
- [x] Remove `SolverRun` fields introduced only for Discopt bounds, nodes,
  termination, and native timing.
- [x] Preserve existing Couenne, IPOPT, and APOPT option-file, availability,
  working-directory, status, timing, and failure behavior.

## Step 3 - Restore Unit-Model Solver Contracts

- [x] Remove `discopt` from solver `Literal` annotations in `base.py`,
  `pinch_design.py`, and `stagewise.py`.
- [x] Remove `discopt` from integer-capable piecewise-profile classifications in
  `base.py`.
- [x] Confirm the remaining annotation and classification sets agree with the
  backend-supported solver names.

## Step 4 - Remove Experimental-Only Files

- [x] Delete
  `OpenPinch/services/heat_exchanger_network_synthesis/common/solver/_discopt.py`.
- [x] Delete `scripts/benchmark_hen_solvers.py`.
- [x] Delete
  `tests/heat_exchanger_network_synthesis/test_benchmark_hen_solvers.py`.
- [x] Delete `tests/strategies/hen_benchmarks.py`.
- [x] Confirm no duplicate or renamed replacement files are created.

## Step 5 - Clean Tests, Telemetry, and Developer Documentation

- [x] Remove Discopt-specific backend and bridge tests from
  `tests/heat_exchanger_network_synthesis/test_pinch_design_method.py`.
- [x] Remove Discopt-only solver-result telemetry from
  `scripts/benchmark_performance.py`.
- [x] Retain the solver-agnostic effective-backend tracer behavior using a
  supported solver in its regression test, because it independently improves
  trace accuracy without adding a Discopt contract.
- [x] Retain solver-agnostic verification and network duty/count metrics in the
  general performance benchmark, because they remain useful for supported HEN
  solvers and do not depend on Discopt.
- [x] Remove the experimental three-stack benchmark section from
  `docs/developer/synthesis-dependency-policy.rst`.
- [x] Remove unused imports or fixtures left by the deleted tests and run scoped
  formatting where mechanically required.

## Step 6 - Perform Static Generation Checks

- [x] Parse every modified Python file successfully.
- [x] Search package code, scripts, tests, developer docs, `pyproject.toml`, and
  `uv.lock` for active Discopt references; allow only historical AI-DLC and
  ignored benchmark-result evidence.
- [x] Confirm `pyproject.toml` and `uv.lock` were not modified by the removal.
- [x] Run `git diff --check`.
- [x] Inspect the scoped diff for accidental changes to supported solver behavior
  or unrelated working-tree content.
- [x] Confirm deleted experimental files are absent and no duplicate files exist.

## Step 7 - Record the Generated-Code Summary

- [x] Create
  `aidlc-docs/construction/discopt-removal/code/code-generation-summary.md`.
- [x] List modified and deleted files, preserved generic benchmark telemetry,
  static checks, and deferred Build and Test commands.
- [x] Record extension compliance: Security and Resiliency disabled; PBT Partial
  with the deleted experimental strategy requiring no replacement.
- [x] Mark every completed checklist item immediately in this plan and update
  `aidlc-docs/aidlc-state.md` in the same interaction.

## Deferred to Build and Test

- Focused backend, PDM, stagewise, and benchmark-performance tests.
- Ruff formatting and lint checks.
- CI-selected non-solver suite with the 95% coverage threshold.
- Solver-marked HEN tests.
- Sphinx documentation build with warnings treated as errors.
- Wheel and source-distribution builds.

## Completion Criteria

- All checklist items above are marked complete.
- Active repository surfaces contain no Discopt integration.
- Supported solver behavior is preserved by construction and ready for tests.
- Unrelated worktree changes and historical benchmark evidence remain intact.
- Generated code is explicitly approved before Build and Test begins.
