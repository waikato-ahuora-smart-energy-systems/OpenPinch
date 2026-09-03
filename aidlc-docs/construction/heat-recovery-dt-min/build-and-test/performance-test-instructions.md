# Performance Test Instructions

No optimizer or unbounded search was introduced. The inverse solver has a hard
100-iteration ceiling and normally converges in fewer than 30 cascade
evaluations for the supported temperature ranges.

A separate benchmark threshold is not required for this cohesive feature.
Retain performance confidence by running the seeded property suites and the
complete repository test suite; investigate any material increase in those
stable gate times before release.
