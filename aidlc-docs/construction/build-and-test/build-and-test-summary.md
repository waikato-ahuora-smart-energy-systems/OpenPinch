# Build and Test Summary

## Acceptance Result

The post-implementation quality corrections, area-slice cleanup, private-helper
extractions, and Stream model refactor passed the non-solver, solver,
documentation, notebook/resource, lint, formatting, example, and packaging
checks listed below.

## Results

- Non-solver tests: 1,947 passed, 4 external-solver tests deselected.
- Coverage: 99% across 22,444 statements; the CI floor is 95%.
- Segmented HEN synthesis tests: 5 passed, 7 unrelated tests deselected.
- Ruff: full lint passed; all changed Python files are formatted.
- Documentation: Sphinx warning-as-error build passed.
- Notebook/example: 43 notebook, documentation-consistency, and packaging-metadata tests passed; segmented targeting smoke coverage passed in the full suite.
- Packaging: wheel and source distribution built successfully.
- Aggregate-CP audit: segmented HEN parents use a zero legacy-CP sentinel; the
  only service-level ``stream.CP`` lookup is the guarded ordinary-stream path.

## Final Staged-Change Audit Corrections

- Expanded segment numeric views now replace stale cache signatures instead
  of retaining one cache entry per stream revision; a regression proves both
  invalidation and bounded cache size.
- The direct-MVR conversion no longer retains an unused private segment-duty
  helper or a test coupled only to that dead implementation.
- Staged Markdown files have canonical single-newline endings, and the staged
  HEN source is Ruff-formatted.

## Stream Refactor Result

- `stream.py` decreased from 1,388 to 1,144 lines.
- Value/period, thermodynamic, and segmented-profile calculations each have one private stateless implementation.
- Both public classes remain defined in `OpenPinch.classes.stream`.
- Existing private wrapper methods, exceptions, units, tolerance behavior, revisions, transactions, deepcopy, pickle, and workspace serialization remain compatible.
- A regression covers multiperiod broadcasting when pressure or enthalpy establishes the parent period count.

## Accepted HEN Area Strategy

HEN topology optimization intentionally uses the existing smooth Chen area
surrogate inside the nonlinear total-cost objective. Exact ordered
segment-summed areas are applied after solution for exchanger outputs, cost
verification, TDM derivatives, and EVM ranking. The user confirmed this
two-level treatment is the correct and appropriate behavior; it is not an
outstanding implementation gap.

## Deferred Final Polish

The user deferred the optional exact-LMTD refinement for later consideration.
If revisited, evaluate an exact logarithmic LMTD expression in the continuous
NLP formulation only. The NLP expression should:

- use the exact counter-current LMTD for positive terminal approaches;
- implement the analytic equal-approach limit to avoid the logarithmic `0/0`
  singularity;
- leave non-NLP and integer-capable formulations on the Chen surrogate; and
- be accepted only after regression comparisons confirm solver convergence,
  feasibility, topology, and post-processed segment-area consistency.

The current Chen-based topology objective remains the accepted baseline, and
the deferred refinement is not part of the immediate next steps.

## Extension Compliance

- Security Baseline: disabled; N/A.
- Resiliency Baseline: disabled; N/A.
- Property-Based Testing (Partial): compliant.
  - PBT-02: generated nested exchanger JSON round trips pass.
  - PBT-03: generated period aggregation and design-area invariants pass.
  - PBT-07: reusable constrained heat-exchanger and stream strategies are used.
  - PBT-08: Hypothesis shrinking and CI seed reporting remain enabled.
  - PBT-09: Hypothesis remains configured through pytest and project dependencies.

## GitHub CI Heat-Pump Zero-Duty Follow-Up

### Build Status

- **Build**: Success.
- **Artifacts**: OpenPinch 0.4.5 wheel (504 KiB) and sdist (391 KiB).
- **Artifact location**: Isolated temporary directory outside the worktree.

### Test Execution Summary

- **Reported plus new regression tests**: 6 passed in 2.37 seconds.
- **Heat-pump/profile integration suite**: 79 passed in 4.81 seconds.
- **Full CI-selected non-solver suite**: 1,964 passed, 4 deselected.
- **Coverage**: 98% across 22,521 statements; required floor is 95%.
- **Ruff formatting**: 3 scoped files already formatted.
- **Ruff lint**: All scoped checks passed.
- **Failures**: 0.

### Property-Based Testing Compliance

- **PBT-02**: N/A; no round-trip operation changed.
- **PBT-03**: Compliant; generated cases verify zero-duty stream omission.
- **PBT-07**: Compliant; the reusable strategy constrains side, duty, mass flow,
  and finite one-ULP profiles.
- **PBT-08**: Compliant; shrinking is enabled and the CI seed `20260715` was
  used in focused and full runs.
- **PBT-09**: Compliant; Hypothesis 6.156.6 ran through pytest.
- **Blocking PBT findings**: None.

### Additional Test Categories

- **Performance**: N/A; no performance-affecting feature was introduced.
- **Contract/API**: N/A; no public contract changed.
- **Security**: N/A; Security Baseline is disabled and no security surface changed.
- **End-to-end**: Covered by the full repository non-solver suite.
- **Resiliency**: N/A; Resiliency Baseline is disabled.

### Overall Status

- **Build**: Success.
- **Tests**: Pass.
- **Coverage gate**: Pass.
- **Ready for Operations**: Yes; no deployment or hosted-state mutation was
  requested. The prior GitHub run will remain failed until these local changes
  are committed and pushed to trigger a new run.

## Segment Batch Update and Pricing Acceptance

### Build Status

- Build: success.
- Documentation: warning-free Sphinx HTML build.
- Distribution artifacts: OpenPinch 0.4.6 wheel and source distribution in an
  isolated temporary directory.

### Test Execution Summary

- Non-solver suite: 1,978 passed, 4 deselected, 0 failed in 120.22 seconds.
- Solver-marked HEN suite: 3 passed, 1 intentional environment-dependent skip,
  0 failed in 85.49 seconds.
- Coverage: 98% across 22,658 statements; required floor is 95%.
- Static quality: repository Ruff lint passed; all 15 changed Python files were
  already formatted; `git diff --check` passed.
- Notebook, end-to-end, structured-input, direct/indirect integration, HPR, and
  documentation smoke coverage passed within the complete suite.

### Extension Compliance

- Security Baseline: disabled; N/A.
- Resiliency Baseline: disabled; N/A.
- Property-Based Testing (Partial): compliant. Domain-specific transaction and
  price/cost invariants ran with normal shrinking and seed `20260715`.
- Performance: no blocking threshold introduced; parent solver axes are unchanged.

### Overall Status

- Build: success.
- Tests: pass.
- Coverage gate: pass.
- Ready for Operations placeholder: yes; no deployment work was requested.

## Pre-Release Corrective PR 1 Evidence

### Results

- Sandbox-safe non-solver matrix: 1,985 passed and 6 deselected in 137.19
  seconds. The deselections were four solver-marked cases plus the two
  environment-dependent documentation/image gates.
- Environment-dependent gates: warning-free Sphinx build passed with official
  intersphinx inventories; Kaleido grid-image export passed with headless Chrome.
- Post-matrix period-context regressions: 2 passed.
- Ruff: repository-wide `OpenPinch` and `tests` lint passed.
- Structured JSON: every packaged sample, example input, and mirrored utility
  targeting fixture parsed with `jq` after canonical-field regeneration.
- Packaging: OpenPinch 0.4.6 wheel and source distribution built successfully in
  an isolated temporary directory.
- Patch hygiene: `git diff --check` passed; generated documentation and
  distribution outputs remained outside the worktree.

### Extension Compliance

- Security Baseline: disabled; N/A for this stage.
- Resiliency Baseline: disabled; N/A for this stage.
- Property-Based Testing (Partial): compliant; generated invariants cover
  immutable owned values, transaction rollback, revisions/cache invalidation,
  serialization, and canonical period-weight expansion.

### Status

PR 1 Domain and Input Correctness is independently green and ready to be
committed as the first stacked pre-release change.
