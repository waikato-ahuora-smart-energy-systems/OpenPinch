# Domain and Input Implementation Summary

## Outcome

OpenPinch now represents a piecewise sensible-duty profile as one parent `Stream` with an immutable ordered tuple of `StreamSegment` children. Parent aggregates are derived, child thermal mutation is transactional, and continuity is validated in every period using the configured thermal tolerance.

## Delivered

- Public segment and temperature-heat profile schemas with mutually exclusive nested input contracts.
- Parent construction from explicit segments and authoritative cumulative-heat profiles.
- Ordered ownership, metadata and period propagation, aggregate protection, atomic mutation, and recursive revisions.
- Parent-default and explicitly expanded collection/reporting projections.
- Structured-input preparation with nested validation paths.
- Segment-expanded problem-table inputs with parent identity retained for counts.
- Focused example and property tests, including JSON round trips and invariant checks.

## Verification

The final non-solver suite passed 1,946 tests with four external-solver tests deselected. Five segmented HEN synthesis tests passed. Ruff, formatting, warning-free Sphinx, notebook/resource, wheel/sdist, pickle/deepcopy, and patch checks passed.

## Maintainability Follow-up

Segmented parent construction, temperature-heat profile conversion, and supplied-parent aggregate validation now live in the private `_stream_segment_preparation.py` helper. `data_preparation.py` retains orchestration and delegates nested stream construction without changing input behavior or public APIs.

The `Stream` model now delegates three stateless calculation boundaries:

- `_stream_value_state.py` handles value coercion, period normalization, broadcasting, copying, and period-count validation.
- `_stream_thermodynamics.py` handles missing-state completion and derived thermal/economic calculations through frozen result records.
- `_stream_profile.py` handles profile-point validation, detached cloning, ordered continuity checks, and aggregate duty/effective-HTC calculations.

`Stream` and `StreamSegment` remain physically defined in `stream.py`. Mutation, ownership, rollback, revisions, public and private wrapper methods, serialization, deepcopy, and pickle module identity are preserved. The main module decreased from 1,388 to 1,144 lines.

## Extension Compliance

- Security Baseline: disabled; not enforced.
- Resiliency Baseline: disabled; not enforced.
- Property-Based Testing (Partial): compliant for the domain unit through shared generators, bounded examples, serializable schemas, invariant properties, shrinking, and a reproducible CI seed.

## Pre-Release Corrective PR 1

### Outcome

The domain and input review findings are closed with clean pre-release contract
changes. No aliases, workspace migrations, or mutation compatibility shims were
added.

### Delivered

- Stream- and segment-owned values are immutable views; explicit domain APIs
  perform transactional mutation and revision/cache invalidation.
- One period-weight resolver pads trailing values and rejects excess,
  non-finite, negative, and all-zero vectors across zones, utilities, HEN
  arrays, and summaries.
- Default utility temperatures use all authoritative child-segment shifted
  extrema in all periods.
- Process-stream inputs reject extras and retired field spellings; all packaged
  and regression example inputs use canonical fields.
- Workspace bundles are strict schema version 2 and store variant data under
  `case_input` only.
- Segmented process streams and utilities use one semantic validation path in
  reporting and preparation, including parent aggregate checks.

### Extension Compliance

- Security Baseline: disabled; N/A.
- Resiliency Baseline: disabled; N/A.
- Property-Based Testing (Partial): compliant for immutable ownership,
  transactional rollback, revision/cache invalidation, schema serialization,
  and period-weight expansion.
