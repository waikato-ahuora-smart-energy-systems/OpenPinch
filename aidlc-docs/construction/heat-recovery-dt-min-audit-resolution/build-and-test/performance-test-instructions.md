# Performance Test Instructions

No optimizer, persistent service, or new external dependency is introduced, so
a standalone load benchmark is not required. Guard performance proportionally
through the bounded property suite and the full configured test duration.

- Keep generated stream sets bounded to at most three active streams per side
  and two segments per generated segmented stream.
- Keep deterministic bisection capped at 100 iterations.
- Confirm selected-period and all-period calls terminate under Hypothesis with
  deadlines disabled only because numerical test duration varies by platform.
- Compare sequential and two-worker all-period results for exact equality.

The accepted complete suite finished in 490.88 seconds; the CI-equivalent
coverage suite finished in 320.87 seconds on the local Python 3.14 environment.
