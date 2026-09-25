# HPR and MVR Benchmark Performance Test Instructions

## Purpose

Verify deterministic search-work bounds and useful convergence evidence. This
is an in-process numerical library, so request throughput, concurrent users,
and network latency are not applicable. No wall-clock service-level objective
was approved.

## Search Budgets

| Profile | Cases | Selected-cycle maximum | Public search-observation maximum |
|---|---:|---:|---:|
| Direct cascade VC heat pump | 9 | 12 | 24 |
| Utility parallel VC heat pump | 9 | 12 | 24 |
| Direct cascade VC refrigeration | 9 | 12 | 24 |
| Utility parallel VC refrigeration | 9 | 12 | 24 |
| Direct optimized VC+MVR heat pump | 9 | 20 | 40 |
| Utility optimized VC+MVR heat pump | 9 | 16 | 32 |

All public calls therefore remain below the approved ceiling of 50 observed
search evaluations.

## Execute the Bounded Search Test

```bash
/usr/bin/time -p uv run --no-sync pytest --hypothesis-seed=20260715 -q \
  tests/e2e/test_hpr_benchmark_helpers.py \
  tests/e2e/test_hpr_mvr.py
```

The assertions, rather than elapsed time, are the performance acceptance
boundary. Each assignment checks its selected objective count and total public
search observations. Each sentinel also checks convergence toward the best
viable observed candidate.

## Verified Results

- All 54 HPR assignments stayed within their profile-specific bounds.
- All six sentinels demonstrated strict material improvement.
- Every sentinel selected the best viable observed objective within tolerance.
- The selected-to-best-observed gap was zero for all six sentinels.
- The complete 66-test focused gate took 47.37 seconds of pytest time and 48.39
  seconds wall time on the verification host.

Elapsed time is recorded for regression comparison only. It is not a portable
performance guarantee because CoolProp, CPU architecture, and system load vary.
The evidence establishes bounded search and convergence toward an optimal
observed solution; it does not prove a global optimum.

## Investigating Regressions

If an evaluation cap fails, report the exact problem filename, profile ID,
selected objective count, total observed search count, and structured failure
diagnostics. Do not increase a cap until the extra evaluations are explained
and a lower calibrated bound has been tested sequentially.
