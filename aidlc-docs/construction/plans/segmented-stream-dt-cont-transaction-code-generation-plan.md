# Segmented Stream dt_cont Transaction Code Generation Plan

## Unit Context

- **Unit**: Domain and Input - segmented stream mutation contract
- **Change type**: Focused brownfield correctness correction
- **Application code**: `OpenPinch/classes/stream.py`
- **Primary tests**: `tests/test_classes/test_stream_segments.py`
- **Domain strategy**: `tests/strategies/stream_segments.py`
- **Dependencies**: Existing `StreamSegment` ownership, detached-candidate cloning,
  complete-profile validation, `replace_segments`, numeric revisions, and
  `StreamCollection.segment_numeric_view`
- **Public contract**: Assigning `dt_cont` on a segmented parent applies the
  requested value to every ordered child atomically; flat-stream behavior is
  unchanged.
- **Story traceability**: N/A - this is an internal domain consistency fix under
  the approved segmented variable-CP stream requirements.

This document is the single source of truth for this focused Code Generation
continuation.

## Execution Steps

### Step 1 - Pin the transactional mutation contract

- [x] Add example-based regressions proving scalar parent `dt_cont` assignment
  propagates to every child and updates the parent aggregate, effective shifted
  temperatures, and expanded numeric view.
- [x] Add a multiperiod regression for `set_value_attr_at_idx("dt_cont", ...)`
  proving only the selected period changes and all children remain aligned.
- [x] Add a rollback regression proving an invalid parent assignment leaves the
  parent, every child, revisions-visible state, and numeric results unchanged.
- [x] Confirm flat streams retain their current scalar and indexed mutation
  behavior.

### Step 2 - Implement atomic child propagation

- [x] Add a private `Stream` helper that clones every child, applies one full or
  indexed `dt_cont` mutation to the detached candidates, and commits through
  `replace_segments` only after all candidates validate.
- [x] Route segmented-parent `_dt_cont` changes from both `set_value_attr` and
  `set_value_attr_at_idx` through the helper while avoiding recursion during
  aggregate synchronization.
- [x] Preserve explicit segment-level mutation, units, multiperiod context,
  ordered ownership, aggregate derivation, and numeric cache invalidation.
- [x] Keep flat-stream behavior and all other segmented aggregate-field guards
  unchanged.

### Step 3 - Verify the general invariant with Property-Based Testing

- [x] Add a Hypothesis invariant using the reusable domain-specific segmented
  stream strategy: after valid parent assignment, every child and the parent
  expose the same base contribution and the expanded numeric view contains that
  contribution for every segment.
- [x] Use bounded thermal values, normal Hypothesis shrinking, and the existing
  reproducible CI seed configuration.

### Step 4 - Run focused and regression verification

- [x] Run the stream and stream-collection unit tests.
- [x] Run direct integration, indirect integration, HPR targeting, problem-table,
  and segmented PDM tests that consume the expanded numeric view.
- [x] Run Ruff formatting/lint checks and `git diff --check` on the modified
  scope.
- [x] Run broader tests and coverage in proportion to the final diff, retaining
  the repository coverage target.

### Step 5 - Complete traceability and handoff

- [x] Update the segmented-stream implementation summary with the transactional
  `dt_cont` contract and verification evidence.
- [x] Update `aidlc-docs/aidlc-state.md` and `aidlc-docs/audit.md` with completed
  checkboxes, test results, and extension compliance.
- [x] Confirm no duplicate application files or unrelated worktree changes were
  introduced.

## Extension Compliance Plan

- **Security Baseline**: Disabled; not enforced.
- **Resiliency Baseline**: Disabled; not enforced.
- **Property-Based Testing (Partial)**: PBT-03, PBT-07, PBT-08, and PBT-09 apply
  to the propagation invariant and will be covered in Step 3. PBT-02 is N/A
  because this change adds no serialization or inverse operation.

## Content Validation

- Markdown headings, lists, paths, inline code, and checkboxes were checked for
  CommonMark-compatible syntax.
- No Mermaid, ASCII diagram, JSON, YAML, or other embedded structured block is
  present.
