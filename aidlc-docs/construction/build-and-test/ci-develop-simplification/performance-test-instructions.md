# Performance applicability

Application load testing is N/A: this change only schedules CI jobs. The reuse
preflight has a five-minute job timeout and bounded API subprocess timeouts.
Measure CI time savings on remote runs; no measured speedup is claimed locally.
