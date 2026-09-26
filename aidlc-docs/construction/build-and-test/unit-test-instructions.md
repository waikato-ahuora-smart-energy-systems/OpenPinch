# Unit Test Execution - Delivery Workflow

## Delivery regression contracts

```bash
uv run --no-sync pytest tests/packaging --hypothesis-seed=20260715 -m "not docs" -q
```

Includes parsed YAML, executable aggregate gates, complete evidence reuse,
fake-clock index verification, manifest/archive round trips, real staging
logic with fake external commands, and interrupted-recovery properties.
No index calls, real sleeps, uploads, or tag mutations occur in these tests.
Temporary-file properties use a bounded two-second per-example allowance
because filesystem latency is not a pure-function performance requirement;
shrinking and the fixed seed remain enabled.

Current verification results are in `build-and-test-summary.md`. Historical
runtime-reduction counts below are retained as baseline evidence, not current
suite size.

## Scope

OpenPinch does not use a single `unit` marker. Inner-layer unit, property, and
contract tests live primarily under `tests/domain`, `tests/contracts`,
`tests/optimisation`, and `tests/analysis`. Hypothesis uses the reproducible CI
seed `20260715`; shrinking remains enabled.

## Run Fast Unit and Property Tests

```bash
uv run --no-sync pytest --hypothesis-seed=20260715 -q \
  tests/domain tests/contracts tests/optimisation tests/analysis \
  -m "not solver and not tespy and not performance and not docs"
```

Expected result: all selected tests pass with no unexpected skip or retry.

## Run the Complete Ordinary Coverage Gate

```bash
uv run --no-sync coverage erase
uv run --no-sync coverage run --branch --source=OpenPinch -m pytest \
  --hypothesis-seed=20260715 \
  -m "not solver and not tespy and not performance and not docs"
uv run --no-sync coverage report --fail-under=95
```

Verified result for this change:

- 3,390 passed;
- 6 expected optional tutorial-profile skips;
- 65 intentional specialized-lane deselections;
- the configured 95 percent coverage threshold passed.

The coverage report is written to the terminal; `.coverage` contains the raw
data unless `COVERAGE_FILE` selects another path.

## Failure Triage

1. Re-run the exact failing node with `-vv --showlocals` and the same Hypothesis
   seed.
2. For generated failures, retain the shrunk reproducer and do not disable
   shrinking.
3. Confirm that the failing test belongs to the ordinary lane before installing
   an optional engine or external solver.
4. Re-run the complete ordinary coverage gate after the focused correction.
