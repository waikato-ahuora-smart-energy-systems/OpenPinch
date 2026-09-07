# HPR graph implementation

R4; HPR-US-3. Solved HP and refrigeration targets emit their own GCC and net-load
profiles. Ordinary direct integration no longer emits an empty HP placeholder.
Graph transport retains graph names and series identity. Named plot methods
accept explicit target selection; default selection requires one matching result.
Reading plots does not run analysis. Local provenance and a numerical digest
reject foreign, stale or modified target handles.

Owners: analysis/graphs/specifications.py, analysis/heat_pumps/service.py,
analysis/targeting/direct.py, contracts/graphs.py, domain/enums.py,
application/hpr_selection.py and presentation/graphs/problem.py.

Fresh notebook execution gives four NLP traces and seven GCC traces per mode.
The focused graph integration also verifies that implicit and explicit selection
return identical data while each mode remains distinct. The final default-plot
regression passed; the broader focused correction gate passed 68 tests.

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
