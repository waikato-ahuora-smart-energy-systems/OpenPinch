# Residual utility workflow implementation

R6; HPR-US-5. problem.target.residual_utility(base_target=heat_pump) creates an
unsolved detached PinchProblem. Canonical input stores a frozen HPR residual and
selected-period provenance. Utility definitions preserve units, prices, capacity
and segmented shape. Residual cases reject mixed physical inputs, process-shift
changes and analyses requiring original process streams.

Direct targeting allocates utilities on the frozen profiles. Utility placement
uses the same data for every candidate, with separate physical temperatures and
thermal slices for entropy calculations. The resulting case preserves the basis
through serialization and final allocation. Source problems and HPR handles
remain unchanged. This is sequential optimization, with HPR fixed.

Owners: contracts/input.py, domain/targets.py, analysis/targeting/residual.py,
application/residual_utility.py, utility_placement.py and target/input lifecycle.
A regression exposed premature deactivation of segmented utility input; utility
construction now marks the consumed schema only after creating the active stream.

Real bounded placement, allocation reconciliation, foreign/modified selections,
round trips, generated operation sequences, zero load and segmented proportions
pass in the 68-test correction gate. Shared-vector HPR aggregates are explicitly
unsupported by conversion; independently solved scalar periods remain eligible.

Final affected verification after the indirect ambient correction: 731 tests
passed in 145.60 seconds. See the integrated Build and Test summary for final
aggregate coverage and the preserved notebook exclusions.

## Extension compliance

PBT-01 through PBT-10: compliant across the integrated change. Designs identify
properties; generated residual cases exercise the independent offset oracle,
nonnegative/finite profiles and idempotence. Graph and frozen-input JSON round
trips preserve data. Generated operation sequences compare allocation, roundtrip
and price edits with the unchanged reference basis after every operation,
including empty sequences. Constrained Hypothesis generators retain shrinking;
seed 20260715 matches CI. Real solver, graph and segmented-utility examples
complement properties. Commutativity/induction are N/A for these operations.
Security and Resiliency are disabled in aidlc-state.md and were skipped.
