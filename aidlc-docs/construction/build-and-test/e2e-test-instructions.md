# End-to-End Test Instructions - HPR and MVR Services

## Standard Benchmark Corpus

Run the public HPR/MVR service against the standard benchmark/profile matrix:

```bash
uv run --no-sync pytest --hypothesis-seed=20260715 -q \
  tests/e2e/test_hpr_mvr.py
```

Expected result: 55 tests pass. This consists of all 54 standard benchmark
cases plus the dedicated direct process-MVR workflow. Each case must either
return a valid bounded result with convergence evidence or the explicitly
allowed structured diagnostic outcome; no case may corrupt prior state.

## Broader Public Workflow Corpus

```bash
uv run --no-sync pytest --hypothesis-seed=20260715 -q tests/e2e/test_main.py
```

This validates the ordinary OpenPinch pipeline for every shipped standard
problem and the root import boundary.

## Optional Tutorial Profiles

Slow HPR, solver, and interactive notebook profiles are opt-in and explain the
six expected skips in the ordinary lane:

```bash
OPENPINCH_TUTORIAL_PROFILES=slow-hpr uv run --no-sync pytest \
  --hypothesis-seed=20260715 -q tests/packaging/test_notebooks.py
```

Install the declared optional dependencies before selecting a profile. Solver
and interactive profiles should be run independently when their external
requirements are available.

## Result Review

- Confirm all 54 standard HPR/MVR parameter IDs were collected.
- Confirm the direct MVR case passed.
- Treat an unexpected skip, retry, crash, non-finite objective, or state leak as
  a failure requiring investigation.
- Preserve the fixed seed and complete corpus when reproducing a failure.
