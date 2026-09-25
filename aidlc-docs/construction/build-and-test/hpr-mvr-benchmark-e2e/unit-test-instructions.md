# HPR and MVR Benchmark E2E Unit Test Instructions

## Purpose

Exercise the pure benchmark helpers, generated convergence properties, and the
two duplicate-temperature production corrections independently of the complete
54-problem numerical matrix.

## Run Helper and Property Tests

```bash
uv run --no-sync pytest --hypothesis-seed=20260715 -q \
  tests/e2e/test_hpr_benchmark_helpers.py
```

Expected result: 11 passed and no failures. These tests cover immutable corpus
discovery, stable round-robin assignment, three-state outcome classification,
atomic state checks, observer delegation, diagnostic bounds, convergence
oracles, repeated candidate points, and tolerance boundaries.

## Run Duplicate-Temperature Regressions

```bash
uv run --no-sync pytest --hypothesis-seed=20260715 -q \
  tests/analysis/heat_pumps/test_targeting.py \
  tests/analysis/heat_pumps/test_hpr_residual_regressions.py
```

The relevant assertions verify:

- ambient-cascade values align to every target problem-table row, including
  repeated exact temperatures;
- equivalent duplicate residual-profile rows collapse at the detached contract
  boundary; and
- conflicting duplicate values fail explicitly rather than being discarded.

## Review Test Results

- All tests must pass under the fixed Hypothesis seed `20260715`.
- Unexpected warnings, raw broadcasting errors, and untyped optimization
  exceptions are failures.
- Property failures should be reproduced with the shrunk example printed by
  Hypothesis before changing generators or tolerances.

The full repository run recorded 3,443 passed tests, including these unit and
property cases, with 96% branch-aware project coverage.
