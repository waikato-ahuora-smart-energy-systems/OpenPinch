# Code Generation Summary: GitHub CI HEN Solver Isolation

## Outcome

The HEN design owner-boundary test now installs the existing fake synthesis
executor before invoking the successful design workflow. The test therefore
validates option ownership and temporary configuration restoration without
consulting Couenne, IPOPT, IDAES solver binaries, or platform PATH contents.

## Modified File

- `tests/analysis/heat_exchanger_networks/test_design_workflow.py` now accepts
  pytest's `monkeypatch` fixture and calls `_use_fake_default_executor` before
  constructing the public example problem.

## Created File

- This summary records Code Generation scope, rationale, and traceability.

## Preserved Contracts

- Non-dictionary internal runtime options still raise `TypeError`.
- HEN configuration keys passed at the wrong ownership boundary still raise
  `ValueError`.
- The successful public design call still returns a result with a manifest.
- Call-local derivative thresholds still leave stored problem configuration
  unchanged.
- The test remains in the normal non-solver CI suite.

## Requirements Traceability

- **FR-CI-HEN-01 / FR-CI-HEN-04**: The existing fake executor isolates the
  successful design call from external solvers.
- **FR-CI-HEN-02 / FR-CI-HEN-03**: All original behavioral assertions remain
  unchanged.
- **NFR-CI-HEN-01**: The affected test no longer depends on local solver
  provisioning.
- **NFR-CI-HEN-02 / NFR-CI-HEN-03**: No production code, dependencies, solver
  behavior, markers, or GitHub Actions workflow changed.

## Generated-Code Checks

- The scoped diff contains only the fixture parameter and helper invocation.
- `git diff --check` passes for the modified test.
- Ruff reports the file is already formatted.
- Only the canonical `test_design_workflow.py` file exists.

## Verification Boundary

Executable pytest and Ruff lint verification belongs to the subsequent Build
and Test stage required by the approved workflow.

## Extension Compliance

- **Security Baseline**: Disabled; skipped.
- **Resiliency Baseline**: Disabled; skipped.
- **Property-Based Testing (Partial)**: PBT-02, PBT-03, and PBT-07 are N/A;
  PBT-08 and PBT-09 remain compliant through the unchanged fixed-seed
  Hypothesis CI configuration.
