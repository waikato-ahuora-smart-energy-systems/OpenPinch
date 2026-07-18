# GitHub CI HEN Solver-Isolation Build and Test Plan

## Execution Checklist

- [x] Record generated-code approval and confirm the scoped test diff.
- [x] Run the exact GitHub Actions failure with the local uv environment.
- [x] Run the complete containing module with CI's non-solver marker and fixed
  Hypothesis seed.
- [x] Run Ruff lint and format checks for the modified test file.
- [x] Run scoped and repository patch-hygiene checks.
- [x] Generate build, unit-test, integration-test, and performance-test
  instructions.
- [x] Generate the Build and Test summary with observed evidence.
- [x] Update state and audit records and present the review gate.

## Scope Rules

- Test the isolated HEN owner-boundary repair without modifying production code.
- Preserve unrelated working-tree changes.
- Do not install or invoke external HEN solver binaries for this workflow.
- Treat performance, security, end-to-end, and deployment testing as N/A unless
  the focused verification reveals broader impact.

## Extension Compliance

- **Security Baseline**: Disabled; skipped.
- **Resiliency Baseline**: Disabled; skipped.
- **Property-Based Testing (Partial)**: Preserve Hypothesis shrinking and use
  the existing fixed CI seed `20260715` for the containing module.
