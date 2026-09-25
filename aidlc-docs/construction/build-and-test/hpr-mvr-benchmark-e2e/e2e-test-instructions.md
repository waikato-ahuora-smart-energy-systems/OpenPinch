# HPR and MVR Benchmark End-to-End Test Instructions

## Scope

The end-to-end suite contains 55 distinct workflows:

- 54 unique standard benchmark inputs, each assigned to exactly one of six HPR
  profiles through a stable round-robin mapping; and
- one separate packaged direct process-MVR workflow.

Within the 54-case matrix, 27 cases use direct process integration and 27 use
utility/site integration. Including the separate process-MVR workflow gives 28
process-level and 27 site-level cases.

The six convergence sentinels are members of the 54-case matrix and are not
additional workflows.

## Execute the Complete Focused Gate

```bash
uv run --no-sync pytest --hypothesis-seed=20260715 -q \
  tests/e2e/test_hpr_benchmark_helpers.py \
  tests/e2e/test_hpr_mvr.py
```

Expected result: 66 passed tests:

- 11 pure helper and property tests;
- 54 real CoolProp HPR benchmark assignments; and
- one direct process-MVR and downstream-targeting workflow.

The verified focused run completed in 47.37 seconds on the test host.

## Acceptance Checks

Every standard case must produce exactly one allowed outcome:

1. a strict solved result with finite thermodynamic and economic evidence;
2. a documented no-op where no target is applicable; or
3. a bounded `HPRTargetingError` with structured diagnostics.

Unexpected exceptions fail the test. Failure and no-op paths must not mutate
the target collection. Successful paths must commit only the expected target.

Every sentinel additionally requires at least two distinct viable observations,
a material objective improvement, a non-increasing incumbent sequence, and a
final selected objective equal to the best viable observed value within the
scale-aware tolerance.

## Observed Classification

The verified environment produced:

- 40 solved cases;
- five legitimate no-ops;
- nine bounded typed failures; and
- zero unexpected outcomes.

These are environment-specific observations, not a universal feasibility
guarantee for every fluid or thermodynamic problem.
