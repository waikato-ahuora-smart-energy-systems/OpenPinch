# Performance Test Instructions: GitHub CI HEN Solver Isolation

## Applicability

Performance testing is N/A for this change.

The patch adds one pytest fixture parameter and replaces a live numerical solve
with the repository's deterministic fake executor inside a test. It does not
change runtime production code, algorithmic complexity, memory behavior,
throughput, concurrency, or user-facing latency.

## Regression Guard

The focused test should remain a short non-solver test suitable for the general
CI job. The observed local execution completed in 3.85 seconds including pytest
startup and fixture loading. This observation is diagnostic, not a performance
service-level objective.

## Extension Compliance

- Security and Resiliency extensions are disabled.
- Partial Property-Based Testing does not impose a performance test requirement
  for this isolated example-based repair.
