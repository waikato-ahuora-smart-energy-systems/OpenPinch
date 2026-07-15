# Experimental Discopt Removal Performance Test Instructions

## Applicability

No load, stress, or solver-comparison performance target applies to this removal.
The change restores the previously supported solver backend and does not add a
runtime path.

The focused `test_benchmark_performance.py` suite verifies that supported-solver
telemetry remains functional. The deleted three-stack benchmark must not be run
as a project command; its ignored result artifacts are historical evidence only.
