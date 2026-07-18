# Performance Test Instructions

## Applicability

Dedicated load, throughput, concurrency, and response-time testing is N/A. This change does not introduce a service, persistence layer, user-interface event loop, or new computational algorithm.

## Proportional Performance Evidence

- The fixed-seed non-solver suite completed in 250.82 seconds.
- The final real-solver profile completed in 223.96 seconds.
- Penalty equations were preserved exactly and their scalar-versus-array aggregation property is tested.
- Unit-group lookup remains a short ordered tuple scan and introduces no new asymptotic behavior.
- The OpenHENS prerequisite check imports and inspects four modules once before comparison work.

If future work changes numerical algorithms or solver task generation, use `scripts/benchmark_performance.py` and compare against checked-in HEN benchmark artifacts. No performance regression was observed or accepted as part of this cleanup.
