# HEN Tier 0 and Tier 1 Exact Regression Plan

## Objective

Confirm whether current HEN synthesis produces the same deterministic outputs as
the code immediately before the segmented-stream HEN refactor.

## Baseline

- **Baseline revision**: `973d2322`, the parent of commit `a4e5f2f8` that adopted
  segmented streams and changed HEN arrays, equations, extraction, verification,
  and area handling.
- **Current revision**: current workspace based on `e5c0a539`, including the
  segmented-stream implementation and later fixes.
- Existing JSON results are supporting evidence only because their case matrices,
  grids, timeouts, and schemas are incomplete or inconsistent.

## Controlled Matrix

- Seven non-reordered OpenHENS fixtures with at most nine process streams.
- Quality tiers 0 and 1.
- The benchmark's fixed one-point settings: approach temperature 10, derivative
  threshold 0.5, and stage selection 3.
- One HEN task worker to remove parallel completion-order effects.
- Identical Python 3.14 environment, solver binaries, fixture files, benchmark
  harness, case order, and timeout for baseline and current code.
- Baseline package code is loaded from a read-only Git archive under `/tmp`;
  current fixture and harness files are used for both runs.

## Exact Comparison Contract

Exclude elapsed times, solver-call durations, run IDs, and diagnostic model paths.
Compare these fields exactly for every case and tier:

- completion status and error category;
- selected method, stage count, task count, and candidate counts;
- total annual, utility, and capital costs;
- active exchanger count and recovery/hot-utility/cold-utility duties;
- selected manifest settings and fallback flags.

For numeric fields, report both Python-value exact equality and absolute/relative
differences. A tolerance match must not be described as exact equality.

## Execution Checklist

- [x] Record the user request and reopen Build and Test for this regression.
- [x] Export baseline revision `973d2322` to a unique temporary directory.
- [x] Prove baseline and current runs import their intended OpenPinch source trees.
- [x] Run a paired Tier 0/1 smoke case and confirm the comparison contract works.
- [ ] Run all seven cases at Tier 0 against the baseline revision.
- [ ] Run all seven cases at Tier 1 against the baseline revision.
- [ ] Run all seven cases at Tier 0 against the current workspace.
- [ ] Run all seven cases at Tier 1 against the current workspace.
- [ ] Validate 28 unique version/case/tier attempts with no missing records.
- [ ] Produce an exact field-by-field comparison and classify every difference.
- [ ] Relate the result to multiperiod and segmented-stream coverage limits.
- [ ] Save the raw baseline/current JSON and comparison report under `results/`.
- [ ] Update the Build and Test summary, state, audit, and this checklist.

## Extension Compliance

- **Security Baseline**: Disabled; not applicable.
- **Resiliency Baseline**: Disabled; not applicable.
- **Property-Based Testing**: Partial; existing generated invariants remain
  covered, while this task adds deterministic real-solver regression evidence.
