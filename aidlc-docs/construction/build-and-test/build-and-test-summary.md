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

## Pre-Release Corrective PR 2 Evidence

### Results

- Focused and broad HEN regressions: 443 passed with four solver cases
  deselected.
- Complete non-solver repository matrix: 1,996 passed and four solver-marked
  cases deselected in 140.69 seconds.
- Solver-marked matrix: three passed and one intentionally skipped in 80.33
  seconds.
- Canonical HEN tier 0/1 matrix: thirteen successful case/tier results and the
  established bounded timeout for Nine-stream Linnhoff/Ahmad tier 1. Both
  Spray Dryer tiers passed after correcting acceptance of a legitimate
  utility-only zero-stage PDM side.
- Ruff: repository-wide lint passed; every PR 2 Python file passed formatting.
- Documentation: the 59-source Sphinx HTML build completed with warnings as
  errors and official intersphinx inventories.
- Packaging: OpenPinch 0.4.6 wheel and source distribution built successfully
  in an isolated temporary directory.
- Patch hygiene: `git diff --check` passed and temporary benchmark, Sphinx, and
  distribution outputs remained outside the worktree.

### Extension Compliance

- Security Baseline: disabled; N/A for this stage.
- Resiliency Baseline: disabled; N/A for this stage.
- Property-Based Testing (Partial): compliant through the period-native,
  exact-boundary, asymmetric warm-start, and multiperiod regression matrix;
  no new generated-domain requirement was introduced in this unit.

### Status

PR 2 Period-Native PDM and Utility Constraints is independently green and
ready to be committed as the second stacked pre-release change.

## Pre-Release Corrective PR 3 Evidence

### Results

- Focused HEN result suites: 307 passed with nine intentional deselections;
  reporting added 52 passes and focused controllability/thermal coverage added
  25 passes.
- Complete non-solver repository matrix: 1,999 passed and four solver-marked
  cases deselected in 127.17 seconds.
- Solver-marked matrix: three passed and one intentionally skipped in 79.24
  seconds.
- Canonical live-solver baseline: Four-stream Linnhoff/Ahmad passed in 69.80
  seconds.
- Canonical tier 0/1 benchmark: twelve case/tier results succeeded. Nine-stream
  tier 1 and Six-stream Yee tier 1 reached their bounded solve timeouts. The
  latter also timed out in isolated 180-second serial and parallel retries;
  the solve did not return a network to the PR 3 extraction/result code.
- Benchmark harness regressions: 10 passed.
- Ruff: repository-wide lint passed and all 34 changed Python files were already
  formatted. A whole-repository format check reported six unrelated existing
  formatting drifts outside this change.
- Documentation and artifacts: notebook JSON parsing, a warning-free 59-source
  Sphinx build, and isolated OpenPinch 0.4.6 wheel/source builds passed.
- Patch hygiene: `git diff --check` passed and all benchmark, Sphinx, and build
  outputs remained outside the worktree.

### Extension Compliance

- Security Baseline: disabled; N/A for this stage.
- Resiliency Baseline: disabled; N/A for this stage.
- Property-Based Testing (Partial): compliant. Existing generated transaction,
  serialization, and multiperiod invariants remained green; PR 3 adds explicit
  deterministic period-state contract and serialization coverage.

### Status

PR 3 Period-Native HEN Results is independently green and ready to be committed
as the third stacked pre-release change. The two bounded tier-1 solve timeouts
are recorded as performance observations, not correctness passes.

## Pre-Release Corrective PR 4 Evidence

### Results

- Regression-first baseline: five expected failures and seventeen passes;
  post-implementation focused result: 22 passed.
- Expanded HPR, reporting, target schema, and unit-aware result coverage: 228
  passed.
- PinchProblem, workspace, export, report-unit, and packaged notebook coverage:
  156 passed.
- Complete seeded non-solver repository matrix: 2,004 passed and four
  solver-marked cases deselected in 125.52 seconds.
- Solver-marked matrix: three passed and one intentionally skipped in 83.99
  seconds.
- Coverage: 98% across 23,201 statements, above the required 95% floor.
- Ruff: repository-wide lint passed; all five changed Python files passed format
  checks.
- Documentation and artifacts: the changed notebook parsed as JSON, all 59
  Sphinx sources built with warnings as errors, and OpenPinch 0.4.6 wheel/source
  distributions built in isolated temporary directories.
- Patch hygiene: `git diff --check` passed and generated artifacts remained
  outside the worktree.

### Extension Compliance

- Security Baseline: disabled; N/A for this stage.
- Resiliency Baseline: disabled; N/A for this stage.
- Property-Based Testing (Partial): compliant. Existing generated transaction,
  serialization, period-weight, and multiperiod invariants passed in the full
  suite; deterministic regressions cover replay restoration and HPR economics.

### Status

PR 4 Summary Isolation and HPR Economics is independently green and ready to be
committed as the fourth stacked pre-release change.

## Pre-Release Corrective Final Build and Test

### Closure

- All fifteen review findings are represented by regression coverage and passed
  in the final complete matrix.
- The four dependency-ordered commits are isolated on
  `codex/pr1-domain-input-correctness`, `codex/pr2-period-native-pdm`,
  `codex/pr3-period-native-hen-results`, and
  `codex/pr4-summary-hpr-economics`.
- The final worktree audit found no untracked files, duplicate application
  copies, unintended generated artifacts, or patch-integrity errors.
- No compatibility aliases, migration loaders, period-zero shims, or deprecated
  scalar HEN operating fields were added.

### Final Matrix

- Seeded non-solver suite: 2,004 passed and four solver cases deselected.
- Solver suite: three passed and one intentional environment-dependent skip.
- Coverage: 98%, above the required 95% floor.
- Ruff lint and changed-file formatting: pass.
- Packaged notebook JSON and notebook smoke tests: pass.
- Sphinx with warnings as errors: 59 sources passed.
- Wheel and source distribution: pass.
- Canonical HEN solver/result evidence remains the PR 3 matrix: twelve accepted
  tier 0/1 networks, plus bounded pre-extraction timeouts for Nine-stream tier 1
  and Six-stream Yee tier 1; the Four-stream live baseline passed.

### Publication Status

Publication is the sole remaining plan item. The GitHub publishing workflow
requires `gh`, which is not installed. The restricted environment also cannot
resolve GitHub, and local `develop` is four approved baseline commits ahead of
`origin/develop`. No remote branch was pushed and no pull request was opened
because doing so without identifying the intended baseline would produce an
incorrect stack or mutate `develop` without authorization.

## Residual Compatibility Shim Removal Evidence

### Build Status

- Build tool: Hatchling through `scripts/build_dist.py`.
- Build status: passed.
- Artifacts: OpenPinch 0.5.0 wheel and source distribution under
  `/private/tmp/openpinch-residual-shims-20260717`.
- Documentation: a fresh 60-source Sphinx build passed with `-E -W` and
  `--keep-going`.

### Test Execution Summary

- Focused HPR, configuration, StreamCollection, architecture, and protected-main
  gate: 277 passed.
- Affected cycle tests after attribute-only migration: 37 passed.
- Complete non-solver suite: 2,063 passed; four solver-tagged tests deselected.
- Ruff lint: passed repository-wide.
- Ruff format: all 459 Python files formatted.
- Stale-surface searches: no helper retry test, alias table, old method export,
  HPR dictionary-emulation method, legacy pickle repair, or noncanonical runtime
  optimiser input remains.
- Patch hygiene: `git diff --check` passed.

### Additional Test Classification

- Contract and end-to-end tests: passed through the focused and complete suites.
- Performance tests: N/A because no numerical or scaling algorithm changed.
- Security tests: N/A because the Security extension is disabled.
- Resiliency tests: N/A because the Resiliency extension is disabled.
- Property-Based Testing additions: N/A because no numerical algorithm changed;
  direct deterministic regressions cover every removed behaviour.
- Solver tests: unaffected and excluded by marker; they are not counted as
  passes in this unit.

### Overall Status

Build and Test passes for the approved residual compatibility-shim cleanup.
`OpenPinch.main.pinch_analysis_service` remains unchanged, and the generated
code is ready for explicit review. Operations is N/A because no deployment or
infrastructure work was requested.

## Post-Implementation Import and Type Correction Evidence

- Regression-first baseline: five expected failures.
- Focused architecture, total-site, and heat-transfer gate: 96 passed.
- Runtime import sweep: all 301 discoverable package modules imported.
- Targeted Pylint errors: none for the five corrected modules.
- Complete non-solver suite: 2,067 passed; four solver-tagged tests deselected.
- Ruff lint and formatting: passed for all 459 Python files.
- Patch hygiene: passed.
- Protected main service: unchanged.

## Serialized HEN Target Input Evidence

- Expanded focused contracts: 177 passed; the final serialized-HEN contract
  file passes 19 tests including canonical input retention.
- Complete non-solver HEN gate: 463 passed, four solver cases deselected.
- Architecture gate: 43 passed.
- Complete non-solver repository suite: 2,091 passed, four solver cases
  deselected.
- Ruff lint: passed repository-wide.
- Ruff formatting: all 460 Python files formatted.
- Documentation: warning-as-error Sphinx build passed for 60 sources.
- Stale symbol guard: no `HeatExchangerStreamRole` or old enum-member spelling
  remains in source, tests, authored docs, or rebuilt generated docs.
- Patch hygiene: `git diff --check` passed.
- Property-based testing: compliant with fixed seed `20260717`, 30 bounded
  generated aligned networks, shrinking, deterministic discovery, and exact
  mapping/ordering invariants.
- Security: N/A, extension disabled.
- Resiliency: N/A, extension disabled.
- Performance: N/A, no numerical or scaling algorithm changed.

The serialized HEN contract is ready for generated-code review. Operations is
N/A because no deployment or infrastructure work was requested.

## Serialized HEN JSON-Safety Fix Evidence

- Regression-first result: three expected `StreamID` JSON serialization
  failures across runtime, canonical-input, and public workspace paths.
- Post-fix targeted result: three passed.
- Expanded HEN, contract, workspace, property, and architecture gate: 574
  passed; four solver-marked cases deselected.
- Complete non-solver suite: 2,093 passed; four solver-marked cases deselected.
- Ruff lint and formatting: passed across all 460 Python files.
- Documentation: warning-as-error Sphinx HTML build passed.
- Stale-symbol and patch-hygiene checks: passed.
- Compatibility: no alias, lowercase legacy value, migration layer, or field
  serializer was added; `StreamID` is directly string-backed.
- Security and Resiliency: N/A, disabled.
- Partial PBT: compliant; the existing seeded serialized-network property ran
  in both the focused and complete suites.
