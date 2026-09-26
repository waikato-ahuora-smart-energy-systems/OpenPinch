# Build and Test Summary

## Delivery Workflow Overhaul - 2026-09-26

- Complete ordinary lane: 3451 passed, 6 expected skips, 65 deselected in
  382.68 seconds; branch-aware coverage report passed the 95-percent gate.
- Final packaging suite after the last recovery audit: 273 passed, 6 expected
  skips in 63.17 seconds. Includes 28 additional finalization cases added after
  the complete ordinary run and a successful warning-strict documentation build.
- Specialized suites: 64 passed, 1 existing environment-dependent solver skip
  in 214.30 seconds; 58 TESPy, 2 performance, 1 docs, 4 solver cases represented.
- Final collection: 3485 ordinary cases (including six expected skips), 65
  specialized cases, and no overlapping specialized markers.
- OpenPinch 0.6.10 wheel/sdist built; separate temporary core/TESPy environments
  passed installed-artifact smoke checks, with imports outside the checkout.
- Ruff, changed-file formatting, actionlint 1.7.12 with ShellCheck 0.11.0,
  lock validation, and patch hygiene passed. Existing unrelated formatting
  differences and Notebook 10 edits are not included.
- Initial failures were diagnosed, not retried blindly: sandbox blocked
  Chrome image export; filesystem-property timing needed a finite 2-second
  allowance rather than the pure-function 200-ms default. Both were verified
  after correction. Numerical assertions and budgets were not weakened.
- All PBT-01 through PBT-10 obligations are compliant; detailed evidence is in
  `../delivery-workflow/code/implementation-summary.md`. Security and Resiliency
  extensions remain disabled (N/A).
- Local implementation/build/test complete. Hosted PR checks, required-check
  activation, trusted-publisher configuration, and actual publication remain
  separate verification layers. No merge, dispatch, publishing, legacy release
  recovery, or remote settings change was performed.

The instructions in this directory cover reproducible build, unit, integration,
performance, and E2E verification. The historical results below are retained
for comparison, not presented as current suite size or a new speedup claim.

## Historical Test-Suite Runtime Reduction

## Build Status

- Build tool: uv with Hatchling through PEP 517.
- Runtime: CPython 3.14.2.
- Project version: OpenPinch 0.6.9.
- Status: success.
- Local distribution build time: 1.4 seconds.
- Artifacts: `openpinch-0.6.9-py3-none-any.whl` and
  `openpinch-0.6.9.tar.gz`.
- Artifact verification: both archives contain Notebook 19; a clean external
  virtual environment imported the wheel rather than the checkout and passed
  the core artifact smoke.

## Test Execution Summary

### Ordinary Coverage and Regression Lane

- 3,390 passed.
- 6 expected optional tutorial-profile skips.
- 65 intentional specialized-lane deselections.
- Branch-aware 95 percent coverage threshold: passed.
- Coverage-instrumented runtime: 434.68 seconds.
- Uninstrumented serial runtime: 383.17 seconds.
- Baseline: 600.89 seconds; improvement: 217.72 seconds or 36.2 percent.

### Integration and End-to-End Evidence

- CoolProp audit: 14 passed.
- HPR/MVR E2E: 55 passed, including all 54 standard cases and direct MVR.
- Utility placement: 50 main tests and 5 batch tests passed in focused runs;
  process/site real optimization and copy isolation remain covered.
- Notebook 19: generator drift and real execution passed; final profile 6.06
  seconds with finite feasible process and site results.
- Fresh-process architecture/API selection: 39 passed in 18.53 seconds.
- Final workflow and notebook contracts: 72 passed.

### Specialized Lanes

- TESPy: 58 passed in 58.23 seconds.
- Performance: 2 convergence benchmarks passed.
- Documentation: 1 warning-strict Sphinx build passed and validated output.
- Solver: 3 passed and 1 existing environment-dependent case skipped.
- No retry, expected-failure conversion, unexpected deselection, or orphaned
  marker was introduced.

### Quality Gates

- Ruff: passed.
- Git patch hygiene: passed.
- GitHub Actions YAML parsing and workflow contracts: passed.
- Canonical notebook regeneration: passed.
- Isolated installed-wheel smoke: passed.

## Overall Status

- Build: success.
- Tests: pass.
- Runtime requirement: pass with 96.83 seconds of margin.
- Coverage requirement: pass.
- HPR/MVR robustness evidence: preserved.
- Production API changes: none.
- Ready for Operations placeholder review: yes, subject to explicit approval of
  this Build and Test stage.

## Generated Instructions

- `build-instructions.md`
- `unit-test-instructions.md`
- `integration-test-instructions.md`
- `performance-test-instructions.md`
- `e2e-test-instructions.md`
- `build-and-test-summary.md`

## Extension Compliance

- Property-Based Testing: compliant. Domain strategies, shrinking, fixed seed,
  and example-based integration evidence remain active.
- Security Baseline: disabled; N/A. No authentication, authorization, secret,
  network-service, or protected-data boundary changed.
- Resiliency Baseline: disabled; N/A. No deployed service, failover, or runtime
  recovery behavior changed.
