# Performance Test Instructions - Delivery Workflow

## Delivery measurement boundary

The numerical budgets and marker selectors below are unchanged. New delivery
helpers use fake clocks and bounded generated inputs, not network timing.
Compare hosted runs of the same profile and candidate class, separating
queue/setup time, test time, and reused evidence. Local concurrent diagnostic
runs are not evidence of a pipeline speedup. Historical measurements below
belong to the earlier runtime-reduction change.

## Requirements

- Ordinary uninstrumented serial selection: at most 480 seconds on the
  profiling host with seed `20260715`.
- Notebook 19: at most 20 seconds.
- Utility-placement file: target 75 seconds or less.
- CoolProp audit: at most 30 seconds.
- Focused fresh-process architecture/API set: at most 20 seconds.
- Coverage-instrumented time is reported separately from the uninstrumented
  600.89-second baseline.

Network throughput, concurrent-user load, and request stress testing are N/A:
OpenPinch is a local scientific Python library and this change concerns test
execution cost rather than a deployed service.

## Run the Dedicated Convergence Benchmarks

```bash
timeout 900s uv run --no-sync pytest --hypothesis-seed=20260715 \
  -m performance --durations=20
```

Expected result: both four-backend convergence benchmarks pass under the fixed
2-run, 25-iteration, 5,000-evaluation profile. A timeout is a hard failure.

## Profile the Ordinary Lane

```bash
uv run --no-sync pytest --hypothesis-seed=20260715 \
  -m "not solver and not tespy and not performance and not docs" \
  --durations=100 --junitxml=/tmp/openpinch-profile.xml -q
```

Run serially and without coverage for comparison with the baseline. Do not use
parallel workers or compare a coverage-instrumented duration to this target.

## Verified Results

| Area | Baseline | Result | Status |
|---|---:|---:|---|
| Ordinary selection | 600.89 s | 383.17 s | Pass, 36.2% faster |
| Notebook 19 | 44.51 s | 6.06 s | Pass |
| Utility placement | 105.13 s | 65.63 s | Pass |
| CoolProp audit | 43.87 s | 22.82 s | Pass |
| Fresh-process focused set | 34.58 s | 18.53 s | Pass |

The dedicated TESPy selection completed in 58.23 seconds. The performance and
documentation selections together completed in 45.41 seconds. These are
separate CI lanes and are intentionally excluded from the ordinary critical
path.

## Regression Response

If the ordinary lane exceeds 480 seconds, use JUnit testcase durations to group
time by `classname`, compare the affected hotspot with this table, and optimize
repeated setup rather than removing assertions or benchmark cases.
