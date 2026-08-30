# Performance Test Instructions

No dedicated performance test is required. The change replaces two local square
operations with one vectorized call over exactly two residuals, so asymptotic
candidate-evaluation complexity is unchanged. Existing Utility Placement
performance tests remain the applicable gate.
