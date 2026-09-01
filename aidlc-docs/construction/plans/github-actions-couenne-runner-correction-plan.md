# GitHub Actions Couenne Runner Correction Plan

## Scope and authorization

The user explicitly authorized correction and commit of the release workflow's
Couenne solver-test failure reproduced in GitHub Actions run 33466355219.

## Execution plan

- [x] Step 1: Add a failing workflow contract for an IDAES-supported solver
  runner and an explicit Couenne/IPOPT runtime probe.
- [x] Step 2: Pin only the release solver-test job to Ubuntu 22.04 and add the
  post-install runtime probe before pytest.
- [x] Step 3: Run focused workflow tests, YAML parsing, and the runtime probe
  against the configured local solvers.
- [x] Step 4: Run the complete configured-solver suite, Ruff, and patch hygiene
  checks.
- [x] Step 5: Review scope, update AI-DLC evidence and checkboxes, and commit
  the verified workflow correction to `develop`.

## Extension applicability

- Property-Based Testing: N/A because the correction is a fixed GitHub runner
  compatibility contract with no meaningful generated input domain.
- Security Baseline: disabled and N/A.
- Resiliency Baseline: disabled and N/A.

## Completion evidence

- RED: the new workflow contract failed against `ubuntu-latest` before reaching
  the missing runtime-probe assertions.
- GREEN: all 25 packaging/workflow tests pass and all three workflow files parse.
- Solver integration: the exact post-install probe reports Couenne and IPOPT
  available in the configured local solver environment.
- Regression: 2,519 tests pass with 4 expected skips under the complete
  configured-solver environment and fixed Hypothesis seed.
- Quality: repository Ruff, changed-file formatting, and patch hygiene pass.
- Property-Based Testing is N/A for this fixed runner-image contract; Security
  and Resiliency remain disabled and N/A.
