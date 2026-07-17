# Package Architecture Modernization Performance Test Instructions

## Applicability

This refactor adds no network service, concurrency target, throughput target,
or new performance requirement. A load/stress gate is therefore N/A. Numerical
correctness and deterministic ordering are blocking; wall-clock timing is a
diagnostic and must not replace behavioural assertions.

## Diagnostic Checks

- Compare fresh-process `OpenPinch.main` import time and imported-module count
  with the recorded Step 1 baseline when investigating a startup regression.
- Run the HEN benchmark-performance tests and canonical fixtures when changing
  solver equations or execution paths.
- Use fixed inputs, fixed seed `20260715`, the same optional solver versions,
  and repeated runs before interpreting timing differences.

```bash
uv run pytest -q tests/analysis/heat_exchanger_networks/test_benchmark_performance.py
```

The long seven-case external solver benchmark is an explicit release or solver
change gate, not a routine architecture-refactor load test. Timeouts and skips
must be reported as classifications rather than converted to successes.
