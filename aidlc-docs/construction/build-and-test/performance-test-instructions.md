# Performance Test Instructions

## Applicability

Dedicated load, stress, throughput, and response-time testing is N/A. OpenPinch
is an in-process engineering package, and this remediation adds no server,
concurrent service, latency objective, database query, network boundary, or
numerical algorithm.

## Proportional Cost Evidence

- Case-name validation is linear in the short identifier and occurs only at
  case boundaries.
- Export containment performs a bounded number of local path operations per
  selected case.
- Workbook allocation normally succeeds on the first exclusive-create attempt
  and retries only on a real collision.
- OpenHENS module isolation is limited to the comparison utility and the fixed
  required module set; solver execution dominates its cost.
- Detached input observation intentionally copies one validated input graph per
  property access.

The complete non-solver suite and focused concurrent workbook test provide
proportional execution evidence. Future benchmarks should record hardware,
Python/dependency versions, case size, period count, worker count, solver, and
backend before comparing elapsed time.
