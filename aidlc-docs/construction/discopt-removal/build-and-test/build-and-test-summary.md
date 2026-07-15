# Experimental Discopt Removal Build and Test Summary

## Overall Status

- **Build**: Success.
- **Tests**: Pass.
- **Coverage**: 99%, above the 95% project target.
- **Active Discopt integration**: None found.
- **Ready for Operations placeholder review**: Yes.

## Test Results

### Focused Unit and Integration Tests

- 138 focused HEN backend, model-helper, segmented-stream, and benchmark tests
  passed.
- Ruff lint passed for the repository.
- Ruff formatting passed for all five currently modified Python files.

### CI-Selected Test and Coverage Suite

- 1,956 tests collected.
- 1,952 non-solver tests passed.
- Four solver tests were deselected by the CI marker expression.
- Total OpenPinch line coverage was 99%.
- The required 95% coverage gate passed.
- Hypothesis ran with the reproducible seed `20260715`, satisfying Partial PBT
  enforcement for the retained pure-function and serialization tests.

### Real-Solver Tests

- Three solver-marked HEN tests passed.
- One solver-marked case was intentionally skipped.
- Couenne and IPOPT/APOPT support remained functional.

## Build Results

- Sphinx 9.1.0 built all 58 sources with warnings treated as errors.
- A clean documentation build in `/tmp/openpinch-docs-discopt-removal` contained
  no Discopt references.
- Built `openpinch-0.4.5-py3-none-any.whl` successfully.
- Built `openpinch-0.4.5.tar.gz` successfully.
- Neither distribution contains the removed Discopt adapter, benchmark harness,
  benchmark test, or Hypothesis strategy.

## Removal Verification

- Active package code, scripts, tests, developer documentation, package metadata,
  and lockfile contain no Discopt references.
- `pyproject.toml` and `uv.lock` hashes match the pre-removal baseline.
- `git diff --check` passed.
- Historical AI-DLC and ignored result records remain intact.
- Unrelated heat-pump and Read the Docs worktree changes remain untouched.

## Generated Instructions

- `build-instructions.md`
- `unit-test-instructions.md`
- `integration-test-instructions.md`
- `performance-test-instructions.md`
- `build-and-test-summary.md`

## Extension Compliance

- **Security Baseline**: Disabled; skipped.
- **Resiliency Baseline**: Disabled; skipped.
- **Property-Based Testing**: Partial; compliant. The deleted experimental
  benchmark strategy required no replacement, while the complete retained
  Hypothesis suite passed with a reproducible seed.
