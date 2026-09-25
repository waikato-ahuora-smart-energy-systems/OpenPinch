# HPR and MVR Notebook Reliability Performance Test Instructions

## Scope

The approved performance contract is budget-bounded completion, not a portable
wall-clock SLA. Machine-dependent timings are recorded as diagnostic evidence.

## Hard Work Limits

- Notebook 08 required Carnot targets: one stage, one restart, 20 iterations,
  and 50 evaluations.
- Notebook 09 required CoolProp VC targets: one stage, one restart,
  20 iterations, and 50 evaluations.
- Notebook 10 required shared designs: one stage, one restart, 20 iterations,
  and 50 evaluations for each fresh technology-specific problem.
- Notebook 10 optional advanced cascade: one restart, five iterations, and
  20 evaluations.
- Notebook 11 required VC+MVR target: one stage, one restart, three iterations,
  and 50 evaluations.

## Execute the Guarded Performance Evidence

```bash
OPENPINCH_TUTORIAL_PROFILES=slow-hpr \
uv run --no-sync pytest --durations=10 -q \
  tests/packaging/test_notebooks.py::test_slow_hpr_notebook_executes

uv run --no-sync pytest --hypothesis-seed=20260715 -q \
  tests/e2e/test_hpr_mvr.py
```

The tests assert configured budgets and bounded observation counts. The E2E
sentinels must show objective improvement and select the best viable observed
candidate within tolerance.

## Interpretation

- A budget overrun is a failure even if the final thermodynamic point solves.
- A fast typed failure is not success for a required notebook path.
- A bounded typed failure is an accepted robust outcome only for an explicitly
  optional screen or a benchmark assignment not designated as a convergence
  sentinel.
- Global optimality is not claimed.

## Verified Timing Snapshot

- Four real HPR/MVR notebooks together: 29.71 seconds.
- Focused 192-test thermodynamic and integration selection: 56.94 seconds.
- CI-equivalent 3,461-test non-solver selection with branch coverage:
  762.76 seconds.

These values are informational results from the verified macOS CPython 3.14.2
environment and must not be promoted to cross-platform thresholds.
