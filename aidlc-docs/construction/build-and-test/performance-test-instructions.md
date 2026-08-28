# Performance Test Instructions

## Purpose

Validate bounded local numerical kernels and evaluation replay. This is a
library-level optimizer, so concurrent-user throughput and network error-rate
targets are not applicable.

## Performance Requirements

- **Entropy kernel**: p95 for 4,000 representative entropy calculations is at
  most 50 ms.
- **Bound reduction**: p95 for 40 utility levels across 100 periods is at most
  150 ms.
- **Complete pure model**: p95 for build, decode, and verify over the same
  representative study is at most 250 ms.
- **Cold evaluation replay**: p95 is at most 1 second for the approved detached
  fixture.
- **Memoized evaluation**: p95 is at most 1 ms.
- **Scaling**: increasing the fixture from 20 to 100 periods remains within the
  approved linear-growth guard.
- **Bounded solve**: iteration and evaluation limits are enforced by contract.

## Setup Performance Test Environment

Use the locked project environment on an otherwise normally loaded local or CI
machine. No service, database, load balancer, or credential is required.

## Run Performance Tests

```bash
uv run pytest tests/analysis/utility_placement/test_performance.py tests/analysis/utility_placement/test_unit2_performance.py -q
```

## Analyze Results

- **Expected**: all five tests pass their embedded wall-clock and scaling
  thresholds.
- **Observed status**: pass in the completed construction run.
- **Results location**: pytest terminal output; the assertions contain the
  authoritative thresholds.

## Performance Optimization

If a threshold fails, rerun the single test on an idle machine, distinguish
environment jitter from reproducible regression, profile the owning pure
kernel or allocation adapter, preserve numerical tolerances and deterministic
ordering, and rerun all specialist and oracle tests after optimization.
