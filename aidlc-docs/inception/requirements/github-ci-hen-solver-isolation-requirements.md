# GitHub CI HEN Solver-Isolation Requirements

## Intent Analysis

- **User request**: Explain and resolve a GitHub Actions test failure that passes
  locally.
- **Request type**: Bug fix in test isolation.
- **Scope estimate**: One test in one HEN design-workflow module.
- **Complexity estimate**: Simple and low risk.
- **Requirements depth**: Minimal. The traceback, CI marker configuration, local
  solver discovery, and existing fake executor establish the expected repair
  without clarification.

## Diagnosed Cause

The CI test job runs pytest with `-m "not solver"`, but
`test_design_options_are_validated_at_their_owner_boundary` is unmarked and
invokes the real default HEN synthesis executor. GitHub's Ubuntu runner has
neither Couenne nor IPOPT, so Couenne falls back to the network-evolution stage
and all resulting tasks fail because `ipopt` is absent.

The test passes locally because the uv environment discovers both executables
under `/Users/timothyw/.idaes/bin`. This local dependency masks the test's
undeclared external-solver requirement.

## Functional Requirements

- **FR-CI-HEN-01**: The owner-boundary test must exercise configuration routing
  and restoration without invoking an external HEN solver.
- **FR-CI-HEN-02**: The test must retain its invalid runtime-options and invalid
  configuration assertions.
- **FR-CI-HEN-03**: The successful design call must still return a result with a
  manifest and prove that call-local derivative thresholds do not mutate the
  stored problem configuration.
- **FR-CI-HEN-04**: The test must use the existing `FakeSynthesisExecutor`
  isolation helper already used by the other public HEN workflow tests.

## Non-Functional Requirements

- **NFR-CI-HEN-01**: The non-solver CI suite must remain independent of Couenne,
  IPOPT, IDAES solver installations, and platform-specific PATH contents.
- **NFR-CI-HEN-02**: Production code and HEN solver behavior must not change.
- **NFR-CI-HEN-03**: GitHub workflow dependency installation and marker policy
  must not change.
- **NFR-CI-HEN-04**: Verification must include the exact failing test and the
  containing non-solver test module with the fixed Hypothesis seed used by CI.

## Recommended Repair

Change the test signature to accept pytest's `monkeypatch` fixture and call
`_use_fake_default_executor(monkeypatch)` before the successful design workflow.
This keeps the test focused on its declared contract. Marking it as `solver`
would hide contract coverage from normal CI, while installing solver binaries
would make the general test job slower and contradict the repository's current
solver marker policy.

## Success Criteria

- The exact reported test passes without consulting external solver binaries.
- All tests in `test_design_workflow.py` pass under `-m "not solver"`.
- Ruff accepts the changed test file.
- No production, dependency, lockfile, fixture, or GitHub Actions file changes.

## Extension Compliance

- **Security Baseline**: Disabled in `aidlc-state.md`; skipped.
- **Resiliency Baseline**: Disabled in `aidlc-state.md`; skipped.
- **Property-Based Testing (Partial)**:
  - **PBT-02**: N/A; no inverse or round-trip operation changes.
  - **PBT-03**: N/A; the repair isolates one fixed contract example and adds no
    general data-transform invariant.
  - **PBT-07**: N/A; no generated domain input is added.
  - **PBT-08**: Compliant; the CI fixed seed remains unchanged.
  - **PBT-09**: Compliant; Hypothesis remains the configured Python framework.

## Non-Goals

- Installing Couenne or IPOPT in the general GitHub Actions test job.
- Reclassifying the owner-boundary test as an external-solver test.
- Changing missing-solver fallback behavior.
- Altering the production design accessor or synthesis service.
